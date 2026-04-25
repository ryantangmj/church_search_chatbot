"""
app.py — FastAPI backend for the church sermon search chatbot
Run: venv/bin/uvicorn app:app --reload --port 5000
"""

import os
from openai import OpenAI
import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

# ── Config ─────────────────────────────────────────────────────────────────────
CHROMA_DB_PATH  = "./chroma_db"
COLLECTION_NAME = "sermons"
N_RESULTS       = 5
OPENAI_MODEL    = "gpt-4o"

app = FastAPI(title="Church Sermon Chatbot")

# ── ChromaDB client (lazy singleton) ───────────────────────────────────────────
_collection = None

def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection

# ── OpenAI client ──────────────────────────────────────────────────────────────
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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
    results = collection.query(query_texts=[query], n_results=n)
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text":           doc,
            "date":           meta.get("date", ""),
            "title":          meta.get("title", ""),
            "chunk_type":     meta.get("chunk_type", "sermon"),
            "chunk_index":    meta.get("chunk_index", 0),
            "total_chunks":   meta.get("total_chunks", 0),
            "scripture_refs": meta.get("scripture_refs", ""),
            "score":          round(1 - dist, 3),
        })
    return chunks


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
- Use Markdown formatting where it improves readability (e.g. bullet points for multi-part answers).
- When referencing a sermon, always cite the date and title.
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
    response = openai_client.chat.completions.create(
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