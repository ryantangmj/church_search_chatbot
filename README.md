# Church Sermon Search — Command Reference

## Project Structure

```
church_search_chatbot/
├── app.py                  # FastAPI backend
├── chunk_sermons.py        # Chunking + ChromaDB ingestion
├── clean_sermons.py        # Docx → cleaned .txt
├── static/
│   └── index.html          # Frontend
├── sermon_documents/       # Raw .docx files (gitignored)
├── cleaned_sermons/        # Cleaned .txt files (gitignored)
├── chroma_db/              # ChromaDB vector store (gitignored)
└── venv/                   # Python virtual environment (gitignored)
```

---

## Setup (first time only)

### 1. Create and activate virtual environment

```bash
/opt/homebrew/bin/python3.11 -m venv venv
```

### 2. Install dependencies

```bash
venv/bin/pip install fastapi uvicorn openai chromadb python-docx python-dateutil python-dotenv rank-bm25
```

### 3. Set environment variable

```bash
export OPENAI_API_KEY="your-key-here"
```

> Tip: Add this to your `~/.zshrc` so you don't have to run it every session.

---

## Pipeline (run in order when adding new sermons)

### Step 1 — Clean raw .docx files → .txt

```bash
venv/bin/python clean_sermons.py
```

Reads from `sermon_documents/`, outputs to `cleaned_sermons/`.

### Step 2 — Chunk and ingest into ChromaDB

```bash
venv/bin/python chunk_sermons.py --input ./cleaned_sermons --collection sermons
```

#### Optional flags

```bash
# Dry run — preview chunks without writing to ChromaDB
venv/bin/python chunk_sermons.py --input ./cleaned_sermons --dry-run

# Also export chunks as JSON
venv/bin/python chunk_sermons.py --input ./cleaned_sermons --export-json chunks.json

# Query the collection from command line
venv/bin/python chunk_sermons.py --query "What did the pastor say about prayer?" --collection sermons
```

---

## Start the app

```bash
export OPENAI_API_KEY="your-key-here"
venv/bin/uvicorn app:app --reload --port 5000
```

Then open: **http://localhost:5000**

API docs (FastAPI auto-generated): **http://localhost:5000/docs**

---

## API Endpoints

| Method | Endpoint      | Description                         |
| ------ | ------------- | ----------------------------------- |
| GET    | `/`           | Serves the frontend                 |
| GET    | `/api/stats`  | Returns total chunk count           |
| POST   | `/api/chat`   | Chat with sermon context via OpenAI |
| POST   | `/api/search` | Raw semantic search, no LLM         |

---

## Troubleshooting

### Wrong Python being used

```bash
# Always use explicit venv paths
venv/bin/python --version
venv/bin/pip list
```

### ChromaDB collection not found

```bash
# Re-run the full pipeline from Step 1
venv/bin/python clean_sermons.py
venv/bin/python chunk_sermons.py --input ./cleaned_sermons --collection sermons
```

### Check what's in ChromaDB

```bash
venv/bin/python chunk_sermons.py --query "test" --collection sermons
```
