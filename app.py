"""
app.py — FastAPI backend for the church sermon search chatbot
Run: venv/bin/uvicorn app:app --reload --port 5000
"""

import os
import math
import string
import re
from openai import AsyncOpenAI
from dotenv import load_dotenv
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from rank_bm25 import BM25Okapi

try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    print("⚠️  sentence-transformers not installed. Re-ranking will be disabled.")
    print("   Install with: pip install sentence-transformers")
    CROSSENCODER_AVAILABLE = False
    CrossEncoder = None

# ── Config ─────────────────────────────────────────────────────────────────────
load_dotenv(override=True)

CHROMA_DB_PATH  = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sermons")
EMBED_MODEL     = os.getenv("EMBED_MODEL", None)
N_RESULTS           = 5
MIN_SEMANTIC_SCORE  = 0.25
OPENAI_MODEL        = "gpt-4o"

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

# ── Cross-Encoder Re-ranker (Lazy singleton) ───────────────────────────────────
_reranker = None

def get_reranker():
    global _reranker
    if not CROSSENCODER_AVAILABLE:
        return None
    if _reranker is None:
        print("⏳ Loading cross-encoder re-ranker model...")
        _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        print("✅ Re-ranker model loaded.")
    return _reranker

# ── Query Classification & Expansion ────────────────────────────────────────────

# Scripture reference patterns
SCRIPTURE_PATTERN = re.compile(
    r'\b(Genesis|Exodus|Leviticus|Numbers|Deuteronomy|Joshua|Judges|Ruth|'
    r'1 Samuel|2 Samuel|1 Kings|2 Kings|1 Chronicles|2 Chronicles|Ezra|Nehemiah|Esther|Job|'
    r'Psalms?|Proverbs?|Ecclesiastes|Song of Solomon|Isaiah|Jeremiah|Lamentations|Ezekiel|Daniel|'
    r'Hosea|Joel|Amos|Obadiah|Jonah|Micah|Nahum|Habakkuk|Zephaniah|Haggai|Zechariah|Malachi|'
    r'Matthew|Mark|Luke|John|Acts|Romans|1 Corinthians|2 Corinthians|Galatians|Ephesians|'
    r'Philippians|Colossians|1 Thessalonians|2 Thessalonians|1 Timothy|2 Timothy|Titus|Philemon|'
    r'Hebrews|James|1 Peter|2 Peter|1 John|2 John|3 John|Jude|Revelation)\s+\d+[:\d\-]*',
    re.IGNORECASE
)

# Theological synonyms for query expansion
EXPANSION_MAP = {
    "prayer": ["praying", "intercession", "communion with God", "talking to God"],
    "faith": ["believe", "believing", "trust", "confidence in God"],
    "salvation": ["saved", "redemption", "being born again", "conversion"],
    "forgiveness": ["forgiving", "mercy", "pardoning", "grace"],
    "love": ["loving", "charity", "compassion", "agape"],
    "worship": ["worshiping", "praise", "adoration", "glorifying God"],
    "sin": ["sinning", "transgression", "iniquity", "wrongdoing"],
    "repentance": ["repent", "turning from sin", "changing", "transformation"],
    "grace": ["unmerited favor", "God's kindness", "divine favor"],
    "sanctification": ["holiness", "being made holy", "spiritual growth", "transformation"],
}

def classify_query(query: str) -> Dict[str, any]:
    """
    Classify query type and extract relevant features.
    Returns: {
        "type": "scripture" | "topical" | "general",
        "scripture_refs": [...],
        "is_date_query": bool,
        "keywords": [...]
    }
    """
    query_lower = query.lower()

    # Check for scripture references
    scripture_refs = SCRIPTURE_PATTERN.findall(query)

    # Check for date mentions
    date_keywords = ["date", "when", "month", "year", "2023", "2024", "2025", "january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
    is_date_query = any(kw in query_lower for kw in date_keywords)

    # Determine query type
    if scripture_refs:
        query_type = "scripture"
    elif is_date_query:
        query_type = "date"
    else:
        query_type = "topical"

    return {
        "type": query_type,
        "scripture_refs": scripture_refs,
        "is_date_query": is_date_query,
        "keywords": tokenize(query)
    }

def expand_query(query: str) -> List[str]:
    """
    Expand query with theological synonyms and related terms.
    Returns list of expanded query variations.
    """
    expansions = [query]  # Always include original

    query_lower = query.lower()
    for term, synonyms in EXPANSION_MAP.items():
        if term in query_lower:
            # Create variations with synonyms
            for syn in synonyms[:2]:  # Use top 2 synonyms to avoid explosion
                expanded = query_lower.replace(term, syn)
                if expanded not in expansions:
                    expansions.append(expanded)

    return expansions[:3]  # Return max 3 variations to keep reasonable

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

def deduplicate_chunks(chunks: List[Dict]) -> List[Dict]:
    """
    Merge overlapping chunks from the same sermon.
    If consecutive chunks from the same sermon have high text overlap, merge them.
    """
    if not chunks:
        return []

    # Group by sermon (title + date)
    from collections import defaultdict
    sermon_groups = defaultdict(list)

    for chunk in chunks:
        key = f"{chunk['title']}|{chunk['date']}"
        sermon_groups[key].append(chunk)

    deduplicated = []

    for sermon_key, sermon_chunks in sermon_groups.items():
        # Sort by chunk_index to maintain order
        sermon_chunks.sort(key=lambda c: c.get("chunk_index", 0))

        merged = []
        for chunk in sermon_chunks:
            if not merged:
                merged.append(chunk)
                continue

            last = merged[-1]

            # Check if chunks are consecutive and from same sermon
            if (abs(chunk.get("chunk_index", 0) - last.get("chunk_index", 0)) <= 1 and
                chunk["title"] == last["title"]):

                # Check text overlap
                last_words = set(last["text"].split()[-50:])  # Last 50 words
                curr_words = set(chunk["text"].split()[:50])  # First 50 words
                overlap = len(last_words & curr_words)

                # If significant overlap (>30%), merge
                if overlap > 30:
                    # Merge texts, avoiding duplication
                    last["text"] = last["text"] + " [...] " + chunk["text"]
                    last["score"] = max(last["score"], chunk["score"])  # Keep best score
                    continue

            merged.append(chunk)

        deduplicated.extend(merged)

    return deduplicated


def retrieve_chunks(query: str, n: int = N_RESULTS) -> list[dict]:
    """
    Enhanced retrieval with:
    - Query classification and expansion
    - Hybrid search (semantic + BM25)
    - Cross-encoder re-ranking
    - Context deduplication
    """
    collection = get_collection()
    bm25_index, bm25_docs = get_bm25_index()
    reranker = get_reranker()

    # 1. Classify query
    query_info = classify_query(query)

    # 2. Expand query for better recall
    query_variations = expand_query(query)

    # We retrieve more chunks initially for re-ranking
    pool_size = max(n * 4, 20)  # Increased pool for re-ranking

    # 3. Semantic Search (ChromaDB) - search with multiple query variations
    semantic_ranks = {}
    for q_variation in query_variations:
        results = collection.query(query_texts=[q_variation], n_results=pool_size // len(query_variations))
        for rank, (doc_id, doc, meta, dist) in enumerate(zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
        )):
            if doc_id not in semantic_ranks:
                semantic_ranks[doc_id] = {
                    "rank": len(semantic_ranks) + 1,
                    "text": doc,
                    "meta": meta,
                    "semantic_score": round(1 - dist, 3)
                }

    # 4. Apply query-specific boosting
    if query_info["type"] == "scripture":
        # Boost scripture-type chunks
        for doc_id, data in semantic_ranks.items():
            if data["meta"].get("chunk_type") == "scripture":
                data["semantic_score"] = min(0.99, data["semantic_score"] * 1.15)

    # 5. Lexical Search (BM25)
    lexical_ranks = {}
    if bm25_index:
        tokenized_query = tokenize(query)
        bm25_scores = bm25_index.get_scores(tokenized_query)
        top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:pool_size]

        for rank, idx in enumerate(top_indices):
            if bm25_scores[idx] > 0:
                doc_info = bm25_docs[idx]
                lexical_ranks[doc_info["id"]] = {"rank": rank + 1, **doc_info}

    # 6. Reciprocal Rank Fusion (RRF)
    rrf_k = 60
    combined = {}
    all_ids = set(semantic_ranks.keys()).union(set(lexical_ranks.keys()))

    for doc_id in all_ids:
        rrf_score = 0.0
        doc_data = semantic_ranks.get(doc_id) or lexical_ranks.get(doc_id)

        if doc_id in semantic_ranks: rrf_score += 1.0 / (rrf_k + semantic_ranks[doc_id]["rank"])
        if doc_id in lexical_ranks:  rrf_score += 1.0 / (rrf_k + lexical_ranks[doc_id]["rank"])

        meta = doc_data["meta"]
        combined[doc_id] = {
            "id":             doc_id,
            "text":           doc_data["text"],
            "date":           meta.get("date", ""),
            "title":          meta.get("title", ""),
            "chunk_type":     meta.get("chunk_type", "sermon"),
            "chunk_index":    meta.get("chunk_index", 0),
            "total_chunks":   meta.get("total_chunks", 0),
            "scripture_refs": meta.get("scripture_refs", ""),
            "topics":         meta.get("topics", ""),
            "section_type":   meta.get("section_type", "body"),
            "rrf_score":      rrf_score,
            "semantic_score": semantic_ranks[doc_id]["semantic_score"] if doc_id in semantic_ranks else 0.0
        }

    # Filter out chunks with insufficient semantic similarity before re-ranking
    combined = {k: v for k, v in combined.items() if v["semantic_score"] >= MIN_SEMANTIC_SCORE}
    if not combined:
        return []

    # 7. Cross-encoder re-ranking on top RRF results
    sorted_by_rrf = sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)
    rerank_candidates = sorted_by_rrf[:pool_size // 2]  # Re-rank top half

    if rerank_candidates and reranker is not None:
        # Prepare pairs for cross-encoder
        pairs = [[query, candidate["text"]] for candidate in rerank_candidates]

        # Get cross-encoder scores
        ce_scores = reranker.predict(pairs)

        # Normalize CE scores to [0,1] via sigmoid (raw range is roughly -10 to +10).
        # Normalize RRF scores to [0,1] via max so both are on the same scale before blending.
        ce_normalized = [1.0 / (1.0 + math.exp(-float(s))) for s in ce_scores]
        max_rrf = max(c["rrf_score"] for c in rerank_candidates) or 1.0

        for candidate, ce_raw, ce_norm in zip(rerank_candidates, ce_scores, ce_normalized):
            candidate["rerank_score"] = float(ce_raw)
            rrf_norm = candidate["rrf_score"] / max_rrf
            candidate["final_score"] = 0.6 * ce_norm + 0.4 * rrf_norm

        # Sort by final score
        reranked = sorted(rerank_candidates, key=lambda x: x["final_score"], reverse=True)

        # Calculate display scores
        for i, res in enumerate(reranked):
            # Normalize rerank score to 0-1 range for display
            display_score = min(0.99, max(0.5, (res["rerank_score"] + 1) / 12))  # CE scores roughly -10 to +10
            res["score"] = round(display_score, 3)
            del res["rrf_score"]
            del res["semantic_score"]
            del res["rerank_score"]
            del res["final_score"]
            del res["id"]

        # 8. Deduplicate overlapping chunks
        deduplicated = deduplicate_chunks(reranked[:n * 2])

        return deduplicated[:n]

    # Fallback if re-ranking fails
    for res in sorted_by_rrf:
        res["score"] = round(res["semantic_score"], 3)
        del res["rrf_score"]
        del res["semantic_score"]
        del res["id"]

    return sorted_by_rrf[:n]


def build_context(chunks: list[dict]) -> str:
    """Build context string with source citations."""
    parts = []
    for i, c in enumerate(chunks, 1):
        label = f"[{i}] {c['date']} — {c['title']} ({c['chunk_type'].upper()})"
        if c["scripture_refs"]:
            label += f" | Scriptures: {c['scripture_refs']}"
        if c.get("topics"):
            label += f" | Topics: {c['topics']}"
        parts.append(f"{label}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def filter_and_renumber_citations(answer: str, chunks: list) -> tuple[str, list]:
    """
    Keep only sources the LLM actually cited, renumbered sequentially.
    E.g. if the answer uses [3][4][5], they become [1][2][3] and only
    those three chunks are returned as sources.
    """
    num_sources = len(chunks)

    # Unique valid citation numbers used in the answer, in document order
    cited_nums = sorted({
        int(c) for c in re.findall(r'\[(\d+)\]', answer)
        if 1 <= int(c) <= num_sources
    })

    if not cited_nums:
        return answer, []

    old_to_new = {old: new for new, old in enumerate(cited_nums, start=1)}

    def replace(m):
        n = int(m.group(1))
        return f"[{old_to_new[n]}]" if n in old_to_new else ""

    rewritten = re.sub(r'\[(\d+)\]', replace, answer)
    cited_chunks = [chunks[i - 1] for i in cited_nums]

    return rewritten, cited_chunks


GUARDRAIL_MODEL = "gpt-4o-mini"

GUARDRAIL_PROMPT = """You are a strict guardrail for a church sermon chatbot. Your job is to review a draft answer and ensure it is:
1. SAFE — appropriate, respectful, and on-topic for a church congregation. Not harmful, offensive, or misleading.
2. GROUNDED — every factual claim is directly supported by the provided sermon excerpts. No guessing, inferring, or drawing on outside knowledge.

You will receive:
- The user's question
- The sermon excerpts used as context (numbered [1], [2], etc.)
- The draft answer

## Your Task
Review the draft answer carefully. If it is both safe and fully grounded, return it unchanged.
If there are problems, rewrite the answer to fix them:
- Remove or correct any claims not supported by the excerpts.
- Remove any harmful, offensive, or off-topic content.
- Keep inline citations [1], [2], etc. intact and accurate.
- Preserve the warm, pastoral tone.
- If the answer cannot be fixed (e.g. the entire answer is ungrounded), replace it with:
  "I couldn't find reliable information on that in the sermon transcripts. Try rephrasing or using different keywords."

## Output Format
Respond with ONLY valid JSON (no markdown, no code fences):
{
  "safe": true or false,
  "grounded": true or false,
  "issues": ["brief description of each problem found, or empty list if none"],
  "answer": "the final answer to return to the user"
}"""


SYSTEM_PROMPT = """You are a warm, knowledgeable assistant for a church congregation.
You help members search and understand sermon messages using an advanced retrieval system.

You will be given relevant excerpts from sermon transcripts as context. Each excerpt includes:
- Date and sermon title
- Chunk type (SERMON, SCRIPTURE, or VIDEO)
- Scripture references (if applicable)
- Key topics extracted from the content

The retrieved excerpts are your ONLY source of truth — answer strictly from them.

## Core Rules
- Never guess, infer, or draw on knowledge outside the provided excerpts.
- If the context does not explicitly answer the question, say clearly:
  "I couldn't find specific information on that in the sermon transcripts. Try rephrasing or using different keywords."
- Do not combine or connect excerpts unless they explicitly reference the same topic or event.
- Accuracy over completeness — an honest "I don't know" is always better than speculation.

## Citation Requirements
- ALWAYS use inline citations [1], [2], etc. to reference which excerpt you're drawing from.
- Only use citation numbers that exist in the provided sources (e.g., if given 5 sources, only use [1] through [5]).
- Place citations immediately after the relevant statement, before punctuation.
- Multiple citations are allowed: "The pastor taught about prayer and fasting [1][3]."

## Tone & Format
- Be warm, pastoral, and encouraging — you are speaking to congregation members.
- Be concise (2–4 sentences for simple questions). Elaborate for complex theological topics.
- Use Markdown **bold** for emphasis on key spiritual concepts.
- For scripture references, always include book, chapter, and verse.
- Structure multi-part answers with clear sections.

## Process
- For multi-part questions, address each part in order.
- If sources conflict, note: "The sermons present different perspectives on this..."
- If coverage is partial, say: "Based on the excerpts, here's what the pastor addressed..."
- Prioritize SCRIPTURE-type chunks when answering biblical interpretation questions."""


# ── Guardrail ──────────────────────────────────────────────────────────────────

async def guardrail_check(question: str, context: str, answer: str) -> str:
    """
    Runs a fast guardrail LLM pass to verify the answer is safe and grounded.
    Returns the (possibly rewritten) answer.
    """
    import json

    user_content = (
        f"USER QUESTION:\n{question}\n\n"
        f"SERMON EXCERPTS:\n{context}\n\n"
        f"DRAFT ANSWER:\n{answer}"
    )

    try:
        response = await openai_client.chat.completions.create(
            model=GUARDRAIL_MODEL,
            max_tokens=1200,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": GUARDRAIL_PROMPT},
                {"role": "user",   "content": user_content},
            ],
        )
        result = json.loads(response.choices[0].message.content)

        if result.get("issues"):
            print(f"⚠️  Guardrail flagged issues: {result['issues']}")

        return result.get("answer", answer)

    except Exception as e:
        # If guardrail fails for any reason, log and pass original answer through
        print(f"⚠️  Guardrail check failed ({e}); returning original answer.")
        return answer


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

    if not chunks:
        return {
            "answer": "I couldn't find any relevant information in the sermon transcripts for your question. Try rephrasing or using different keywords.",
            "sources": []
        }

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

    # 5. Guardrail: verify answer is safe and grounded in the retrieved chunks
    answer = await guardrail_check(req.message, context, answer)

    # 6. Filter to only cited sources and renumber citations sequentially
    answer, cited_chunks = filter_and_renumber_citations(answer, chunks)

    # 7. Return answer + sources with enhanced metadata
    sources = [
        {
            "date":       c["date"],
            "title":      c["title"],
            "chunk_type": c["chunk_type"],
            "score":      c["score"],
            "preview":    c["text"][:250] + "..." if len(c["text"]) > 250 else c["text"],
            "topics":     c.get("topics", ""),
            "section":    c.get("section_type", ""),
        }
        for c in cited_chunks
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