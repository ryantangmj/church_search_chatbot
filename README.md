---
title: Church Sermon Chatbot
emoji: ⛪
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Church Sermon Search Chatbot

A RAG-powered chatbot for searching and querying church sermon transcripts. Uses hybrid retrieval (semantic + BM25 + cross-encoder re-ranking) backed by ChromaDB and GPT-4o.

## Project Structure

```
church_search_chatbot/
├── app.py                        # FastAPI backend (the running server)
├── scripts/
│   ├── scrape.py                 # Step 0: scrape .docx files from church website (gitignored)
│   ├── post_process_doc.py       # Step 1: .docx → cleaned .txt
│   ├── chunk_sermons.py          # Step 2: .txt → chunks → ChromaDB
│   └── compare_embeddings.py     # Dev utility: compare embedding model quality
├── static/
│   └── index.html                # Frontend
├── sermon_documents/             # Raw .docx sermon files (gitignored)
├── cleaned_sermons/              # Cleaned .txt files (gitignored)
├── chroma_db/                    # ChromaDB vector store (gitignored)
├── .env                          # API keys and config (gitignored)
├── .env.example                  # Template for .env
└── requirements.txt
```

---

## Setup (first time only)

### 1. Create and activate virtual environment

```bash
/opt/homebrew/bin/python3.11 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
venv/bin/pip install -r requirements.txt
```

First install may take 5–10 minutes as it downloads ML models (PyTorch, sentence-transformers, etc.).

### 3. Configure environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
OPENAI_API_KEY="your-openai-api-key"
CHROMA_DB_PATH="./chroma_db"
COLLECTION_NAME="sermons"

# Optional: use a local Ollama embedding model instead of the default all-MiniLM-L6-v2
# EMBED_MODEL="qwen3-embedding:4b"
```

---

## Pipeline (run in order when adding new sermons)

### Step 1 — Clean raw .docx files

```bash
venv/bin/python scripts/post_process_doc.py
```

Reads from `sermon_documents/`, writes cleaned `.txt` files to `cleaned_sermons/`. Extracts dates, titles, scripture references, and strips formatting noise.

### Step 2 — Chunk and index into ChromaDB

```bash
venv/bin/python scripts/chunk_sermons.py --input ./cleaned_sermons --collection sermons
```

The collection name must match `COLLECTION_NAME` in your `.env`.

Useful flags:

```bash
# Preview chunks without writing to ChromaDB
venv/bin/python scripts/chunk_sermons.py --input ./cleaned_sermons --dry-run

# Export all chunks as JSON
venv/bin/python scripts/chunk_sermons.py --input ./cleaned_sermons --export-json chunks.json

# Test retrieval against an existing collection
venv/bin/python scripts/chunk_sermons.py --query "What did the pastor say about prayer?" --collection sermons
```

### Step 3 (Optional) — Use a local embedding model via Ollama

By default the app uses ChromaDB's built-in `all-MiniLM-L6-v2`. To run embeddings locally:

```bash
brew install ollama
ollama serve
ollama pull qwen3-embedding:4b   # ~2.5 GB
```

Update `.env`:

```bash
EMBED_MODEL="qwen3-embedding:4b"
COLLECTION_NAME="sermons_qwen"   # use a new collection name — don't mix embedding models
```

Re-run Step 2 to build the new collection. The embedding model used at ingestion **must match** `EMBED_MODEL` at runtime.

---

## Start the app

```bash
venv/bin/uvicorn app:app --reload --port 5000
```

Open: **http://localhost:5000**

API docs: **http://localhost:5000/docs**

---

## API Endpoints

| Method | Endpoint      | Description                          |
| ------ | ------------- | ------------------------------------ |
| GET    | `/`           | Serves the frontend                  |
| GET    | `/api/stats`  | Returns total chunk count            |
| POST   | `/api/chat`   | Chat with sermon context via GPT-4o  |
| POST   | `/api/search` | Raw semantic search, no LLM          |

---

## Retrieval Pipeline

Each query goes through:

1. **Query classification** — detects scripture references, date queries, or topical searches
2. **Query expansion** — adds theological synonyms (e.g. "prayer" → "intercession")
3. **Semantic search** — ChromaDB cosine similarity, filtered to `semantic_score ≥ 0.25`
4. **Lexical search** — BM25 keyword matching
5. **Reciprocal Rank Fusion** — combines semantic and lexical rankings
6. **Cross-encoder re-ranking** — neural re-scoring of top candidates (`cross-encoder/ms-marco-MiniLM-L-6-v2`)
7. **Deduplication** — merges overlapping chunks from the same sermon
8. **Citation filtering** — response only shows sources the LLM actually cited, renumbered sequentially

---

## Troubleshooting

**Wrong Python being used**
```bash
venv/bin/python --version
venv/bin/pip list
```

**ModuleNotFoundError (yake, sentence_transformers, etc.)**
```bash
venv/bin/pip install sentence-transformers yake-keyword torch
```

**ChromaDB collection not found**

Check that `COLLECTION_NAME` in `.env` matches what you ingested:
```bash
venv/bin/python -c "import chromadb; c = chromadb.PersistentClient('./chroma_db'); print([col.name for col in c.list_collections()])"
```

If missing, re-run the pipeline:
```bash
venv/bin/python scripts/post_process_doc.py
venv/bin/python scripts/chunk_sermons.py --input ./cleaned_sermons --collection sermons
```

**Slow first query**

First request initializes the BM25 index and loads the cross-encoder model (~8–15 seconds). Subsequent queries are ~1–3 seconds. To pre-load at startup, add to `app.py`:

```python
@app.on_event("startup")
async def startup_event():
    get_bm25_index()
    get_reranker()
```
