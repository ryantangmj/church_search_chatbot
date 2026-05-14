"""
app.py — FastAPI backend for the church sermon search chatbot
Run: venv/bin/uvicorn app:app --reload --port 5000
"""

import os
import string
from openai import AsyncOpenAI
from dotenv import load_dotenv
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from rank_bm25 import BM25Okapi

# ── Config ─────────────────────────────────────────────────────────────────────
load_dotenv()

CHROMA_DB_PATH  = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sermons")
EMBED_MODEL     = os.getenv("EMBED_MODEL", None)
N_RESULTS       = 5
OPENAI_MODEL    = "gpt-4o"

app = FastAPI(title="Church Sermon Chatbot")

# ── Embedding Function Helper ──────────────────────────────────────────────────
def get_embedding_function(model: str = None):
    """Returns embedding function. None = ChromaDB default (all-MiniLM-L6-v2)."""
    if model is None:
        print("ℹ️  Using default embedding model (all-MiniLM-L6-v2)")
        return None

    from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
    print(f"🔗  Using Ollama embedding model: {model}")
    return OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name=model,
    )

embedding_function = get_embedding_function(EMBED_MODEL)

# ── ChromaDB client (lazy singleton) ───────────────────────────────────────────
_collection = None

def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = client.get_collection(
            name=COLLECTION_NAME,
            **({"embedding_function": embedding_function} if embedding_function else {})
        )
    return _collection

# ── BM25 Lexical Index (Lazy singleton) ────────────────────────────────────────
_bm25_index = None
_bm25_docs = []

def tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25: lowercase, remove punctuation, split by spaces."""
    return text.lower().translate(str.maketrans('', '', string.punctuation)).split()

def get_bm25_index():
    global _bm25_index, _bm25_docs
    if _bm25_index is None:
        print("⏳ Building BM25 keyword index from ChromaDB...")
        collection = get_collection()
        all_data = collection.get(include=["documents", "metadatas"])
        
        tokenized_corpus = []
        for doc_id, doc, meta in zip(all_data["ids"], all_data["documents"], all_data["metadatas"]):
            _bm25_docs.append({"id": doc_id, "text": doc, "meta": meta})
            tokenized_corpus.append(tokenize(doc))
            
        if tokenized_corpus:
            _bm25_index = BM25Okapi(tokenized_corpus)
        print(f"✅ BM25 index built with {len(_bm25_docs)} documents.")
    return _bm25_index, _bm25_docs

# ── OpenAI client ──────────────────────────────────────────────────────────────
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("⚠️ WARNING: OPENAI_API_KEY is not set. The /api/chat endpoint will fail.")
openai_client = AsyncOpenAI(api_key=api_key)

# ── Request / Response models ───────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: Optional[list[dict]] = []

class SearchRequest(BaseModel):
    query: str
    n: Optional[int] = N_RESULTS

# ── Helpers ────────────────────────────────────────────────────────────────────

def retrieve_chunks(query: str, n: int = N_RESULTS) -> list[dict]:
    collection = get_collection()
    bm25_index, bm25_docs = get_bm25_index()

    # We retrieve more chunks initially to ensure good overlap for Reciprocal Rank Fusion
    pool_size = max(n * 2, 10)
    
    # 1. Semantic Search (ChromaDB)
    results = collection.query(query_texts=[query], n_results=pool_size)
    semantic_ranks = {}
    for rank, (doc_id, doc, meta, dist) in enumerate(zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
    )):
        semantic_ranks[doc_id] = {
            "rank": rank + 1, "text": doc, "meta": meta, "semantic_score": round(1 - dist, 3)
        }

    # 2. Lexical Search (BM25)
    lexical_ranks = {}
    if bm25_index:
        tokenized_query = tokenize(query)
        bm25_scores = bm25_index.get_scores(tokenized_query)
        top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:pool_size]
        
        for rank, idx in enumerate(top_indices):
            if bm25_scores[idx] > 0: # Only include if there's an actual keyword match
                doc_info = bm25_docs[idx]
                lexical_ranks[doc_info["id"]] = {"rank": rank + 1, **doc_info}

    # 3. Reciprocal Rank Fusion (RRF)
    rrf_k = 60
    combined = {}
    all_ids = set(semantic_ranks.keys()).union(set(lexical_ranks.keys()))

    for doc_id in all_ids:
        rrf_score = 0.0
        doc_data = semantic_ranks.get(doc_id) or lexical_ranks.get(doc_id)
        
        if doc_id in semantic_ranks: rrf_score += 1.0 / (rrf_k + semantic_ranks[doc_id]["rank"])
        if doc_id in lexical_ranks:  rrf_score += 1.0 / (rrf_k + lexical_ranks[doc_id]["rank"])
            
        # Calculate a display-friendly percentage score for the UI
        display_score = semantic_ranks[doc_id]["semantic_score"] if doc_id in semantic_ranks else 0.0
        if doc_id in lexical_ranks and doc_id not in semantic_ranks:
            display_score = max(0.50, 0.90 - (lexical_ranks[doc_id]["rank"] * 0.05))
        if doc_id in semantic_ranks and doc_id in lexical_ranks:
            display_score = min(0.99, display_score + 0.08) # Hybrid synergy boost!
            
        meta = doc_data["meta"]
        combined[doc_id] = {
            "text":           doc_data["text"],
            "date":           meta.get("date", ""),
            "title":          meta.get("title", ""),
            "chunk_type":     meta.get("chunk_type", "sermon"),
            "chunk_index":    meta.get("chunk_index", 0),
            "total_chunks":   meta.get("total_chunks", 0),
            "scripture_refs": meta.get("scripture_refs", ""),
            "score":          round(display_score, 3),
            "rrf_score":      rrf_score # Used internally for sorting
        }
        
    # 4. Sort by highest RRF rank and clean up
    sorted_results = sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)
    for res in sorted_results: del res["rrf_score"]
        
    return sorted_results[:n]


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        label = f"[{i}] {c['date']} — {c['title']} ({c['chunk_type'].upper()})"
        if c["scripture_refs"]:
            label += f" | Scriptures: {c['scripture_refs']}"
        parts.append(f"{label}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


SYSTEM_PROMPT = """You are a warm, knowledgeable assistant for a church congregation.
You help members search and understand sermon messages.

You will be given relevant excerpts from sermon transcripts as context.
The retrieved excerpts are your ONLY source of truth — answer strictly from them.

## Core Rules
- Never guess, infer, or draw on knowledge outside the provided excerpts.
- If the context does not explicitly answer the question, say clearly:
  "I couldn't find anything on that in the sermon transcripts. Try searching with different keywords."
- Do not combine or connect excerpts unless they explicitly reference the same topic or event.
- Accuracy over completeness — an honest "I don't know" is always better than a plausible-sounding answer.

## Tone & Format
- Be warm, pastoral, and encouraging — you are speaking to congregation members.
- Be concise by default (2–4 sentences). Elaborate only if the question genuinely requires it.
- Use Markdown formatting where it improves readability.
- IMPORTANT: Use inline brackets like [1] or [2] to quote the source excerpt you are drawing from, just like a search engine overview.
- For scripture references, always cite the book and verse explicitly.

## Process
- For multi-part questions, address each part separately.
- If sources in the context conflict with each other, note the discrepancy rather than merging them.
- If coverage is incomplete, explicitly say so rather than filling the gap."""


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def stats():
    collection = get_collection()
    return {"total_chunks": collection.count(), "collection": COLLECTION_NAME}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    # 1. Retrieve relevant chunks
    chunks = retrieve_chunks(req.message)

    # 2. Build context
    context = build_context(chunks)

    # 3. Build messages for OpenAI (system message goes first)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in (req.history or [])[-6:]:   # keep last 3 exchanges
        messages.append({"role": h["role"], "content": h["content"]})

    messages.append({
        "role": "user",
        "content": f"Context from sermon transcripts:\n\n{context}\n\n---\n\nUser question: {req.message}"
    })

    # 4. Call OpenAI
    response = await openai_client.chat.completions.create(
        model      = OPENAI_MODEL,
        max_tokens = 1024,
        messages   = messages,
    )

    answer = response.choices[0].message.content

    # 5. Return answer + sources
    sources = [
        {
            "date":       c["date"],
            "title":      c["title"],
            "chunk_type": c["chunk_type"],
            "score":      c["score"],
            "preview":    c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"],
        }
        for c in chunks
    ]

    return {"answer": answer, "sources": sources}


@app.post("/api/search")
async def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Empty query")

    chunks = retrieve_chunks(req.query, n=req.n)
    return {"results": chunks}


# ── Serve frontend ─────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def index():
    return FileResponse("static/index.html")