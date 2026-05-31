"""
app.py — FastAPI backend for the church sermon search chatbot
Run: venv/bin/uvicorn app:app --reload --port 5000
"""

import os
import math
import string
import re
import datetime
import calendar
from openai import AsyncOpenAI
from dotenv import load_dotenv
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from rank_bm25 import BM25Okapi

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    print("⚠️  sentence-transformers not installed. Embeddings and re-ranking will be disabled.")
    print("   Install with: pip install sentence-transformers")
    CROSSENCODER_AVAILABLE = False
    SentenceTransformer = None
    CrossEncoder = None

# ── Config ─────────────────────────────────────────────────────────────────────
load_dotenv(override=True)

CHROMA_DB_PATH  = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "sermons")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "")  # e.g. "yourname/church-chroma-db"
HF_TOKEN        = os.getenv("HF_TOKEN", "")
N_RESULTS           = 5   # default for specific questions
N_RESULTS_BROAD     = 10  # for year/week/month range queries
MIN_SEMANTIC_SCORE  = 0.25
OPENAI_MODEL        = "gpt-4o"

def _ensure_chroma_db():
    if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
        return
    if not HF_DATASET_REPO:
        return
    print(f"⏳ ChromaDB not found locally, downloading from {HF_DATASET_REPO}...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        local_dir=CHROMA_DB_PATH,
        token=HF_TOKEN or None,
    )
    print("✅ ChromaDB downloaded.")
    files = []
    for root, _, fs in os.walk(CHROMA_DB_PATH):
        for f in fs:
            files.append(os.path.relpath(os.path.join(root, f), CHROMA_DB_PATH))
    print(f"   Files in ChromaDB dir: {files}")

_ensure_chroma_db()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(_):
    _get_embed_model()
    get_bm25_index()
    get_reranker()
    yield

app = FastAPI(title="Church Sermon Chatbot", lifespan=lifespan)

# ── Embedding Function (nomic-embed-text-v1.5) ─────────────────────────────────
_embed_model_instance = None

def _get_embed_model():
    global _embed_model_instance
    if _embed_model_instance is None:
        print("⏳ Loading nomic-embed-text-v1.5...")
        _embed_model_instance = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
        )
        print("✅ Embedding model loaded.")
    return _embed_model_instance

class NomicEmbeddingFunction(EmbeddingFunction[Documents]):
    def name(self) -> str:
        return "nomic-embedding-function"

    def __call__(self, input: Documents) -> Embeddings:
        return _get_embed_model().encode(
            ["search_document: " + t for t in input],
            normalize_embeddings=True
        ).tolist()

embedding_function = NomicEmbeddingFunction()

# ── ChromaDB client (lazy singleton) ───────────────────────────────────────────
_collection = None

def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        _collection = client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
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

# ── Source type filter detection ───────────────────────────────────────────────

SERMON_CHUNK_TYPES = {"sermon", "scripture"}
LESSON_CHUNK_TYPES = {"30 lessons"}

def detect_source_filter(message: str, history: list | None = None) -> dict | None:
    """Return a ChromaDB where-filter if the message or recent history targets a specific source type."""
    def _check(text: str) -> dict | None:
        t = text.lower()
        if re.search(r'\bsermon\b|\bmessage\b', t):
            return {"chunk_type": {"$in": list(SERMON_CHUNK_TYPES)}}
        if re.search(r'\blesson\b', t):
            return {"chunk_type": {"$in": list(LESSON_CHUNK_TYPES)}}
        return None

    found = _check(message)
    if found:
        return found

    # Fall back to scanning the last few history turns for a source signal
    for turn in reversed((history or [])[-6:]):
        content = turn.get("content", "") if isinstance(turn, dict) else ""
        found = _check(content)
        if found:
            return found

    return None


def merge_where_filters(*filters) -> dict | None:
    """Combine multiple ChromaDB where-filters with $and, flattening nested $and lists."""
    active = [f for f in filters if f]
    if not active:
        return None
    if len(active) == 1:
        return active[0]
    conditions = []
    for f in active:
        if list(f.keys()) == ["$and"]:
            conditions.extend(f["$and"])
        else:
            conditions.append(f)
    return {"$and": conditions}


def year_filter_from_year(year: int | None) -> tuple[dict | None, str | None]:
    """Return a ChromaDB integer-equality filter for the given year."""
    if not year:
        return None, None
    return {"year": {"$eq": year}}, str(year)


_WEEKDAY_MAP = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "friday": 4, "fri": 4, "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

# The only days services are held
_SERVICE_WEEKDAYS = {6, 2}  # Sunday=6, Wednesday=2

_MONTH_MAP = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}


def most_recent_service_date() -> datetime.date:
    """Return the most recent Sunday or Wednesday (today inclusive)."""
    today = datetime.date.today()
    for offset in range(8):
        d = today - datetime.timedelta(days=offset)
        if d.weekday() in _SERVICE_WEEKDAYS:
            return d
    return today

def resolve_date_expression(expr: str | None) -> datetime.date | None:
    """
    Convert a raw date expression extracted by the LLM (e.g. "last sunday",
    "may 18 2025", "2025-05-18") to a datetime.date using Python datetime math.
    """
    if not expr:
        return None
    expr_lower = expr.strip().lower()
    today = datetime.date.today()

    # Normalise aliases before matching
    expr_lower = re.sub(r'\b(most recent|recent|latest)\b', 'last', expr_lower)
    expr_lower = re.sub(r'\bpast\b', 'last', expr_lower)

    # "last service/message/sermon" with no specific day → most recent Sunday or Wednesday
    if re.search(r'\blast\s+(service|message|sermon)\b', expr_lower):
        return most_recent_service_date()

    # Relative: "last wednesday", "this sunday", etc.
    m = re.match(
        r'(last|this)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun)',
        expr_lower,
    )
    if m:
        qualifier, day_name = m.group(1), m.group(2)
        target_wd = _WEEKDAY_MAP[day_name]
        days_since = (today.weekday() - target_wd) % 7
        if qualifier == "last":
            days_since = days_since or 7
        return today - datetime.timedelta(days=days_since)

    # Explicit date strings
    for fmt in ["%Y-%m-%d", "%B %d, %Y", "%B %d %Y", "%b %d, %Y", "%b %d %Y"]:
        try:
            return datetime.datetime.strptime(expr_lower, fmt).date()
        except ValueError:
            continue

    # Month + day without year ("may 18") — assume current year, roll back if future
    for fmt in ["%B %d", "%b %d"]:
        try:
            d = datetime.datetime.strptime(expr_lower, fmt).date().replace(year=today.year)
            return d if d <= today else d.replace(year=today.year - 1)
        except ValueError:
            continue

    return None


def date_filter_from_date(d: datetime.date | None) -> tuple[dict | None, str | None]:
    """Build a ChromaDB equality filter on the date_numeric integer field."""
    if not d:
        return None, None
    date_numeric = d.year * 10000 + d.month * 100 + d.day
    return {"date_numeric": {"$eq": date_numeric}}, d.strftime("%B %-d %Y")


def resolve_date_range(expr: str | None) -> tuple[datetime.date, datetime.date] | None:
    """
    Convert a range expression to (start, end) dates. Handles:
    - Relative weeks/months: "last week", "this month", etc.
    - Month names with optional year: "may", "may 2023", "2023 may"
      → defaults to most recent occurrence of that month when no year is given.
    """
    if not expr:
        return None
    t = expr.strip().lower()
    today = datetime.date.today()

    if "last week" in t:
        last_monday = today - datetime.timedelta(days=today.weekday() + 7)
        return last_monday, last_monday + datetime.timedelta(days=6)

    if "this week" in t:
        this_monday = today - datetime.timedelta(days=today.weekday())
        return this_monday, today

    if "last month" in t:
        first_of_this_month = today.replace(day=1)
        last_month_end = first_of_this_month - datetime.timedelta(days=1)
        return last_month_end.replace(day=1), last_month_end

    if "this month" in t:
        return today.replace(day=1), today

    # Month name (with optional year): "may", "may 2023", "2023 may", "february"
    month_names = "|".join(sorted(_MONTH_MAP, key=len, reverse=True))
    m = re.search(rf'\b({month_names})\b', t)
    if m:
        month_num = _MONTH_MAP[m.group(1)]
        year_m = re.search(r'\b(\d{4})\b', t)
        if year_m:
            year = int(year_m.group(1))
        else:
            # Default to most recent occurrence: use current year unless month is in the future
            year = today.year
            if month_num > today.month:
                year -= 1
        last_day = calendar.monthrange(year, month_num)[1]
        end = min(datetime.date(year, month_num, last_day), today)
        return datetime.date(year, month_num, 1), end

    return None


def date_range_filter(start: datetime.date, end: datetime.date) -> tuple[dict, str]:
    """Build a ChromaDB $and range filter on date_numeric."""
    start_num = start.year * 10000 + start.month * 100 + start.day
    end_num   = end.year   * 10000 + end.month   * 100 + end.day
    flt = {"$and": [
        {"date_numeric": {"$gte": start_num}},
        {"date_numeric": {"$lte": end_num}},
    ]}
    hint = (
        start.strftime("%B %Y")
        if start.day == 1 and end.day == calendar.monthrange(end.year, end.month)[1]
        else f"{start.strftime('%B %-d')} to {end.strftime('%B %-d %Y')}"
    )
    return flt, hint

# ── Query Classification & Expansion ────────────────────────────────────────────

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


def retrieve_chunks(
    query: str,
    n: int = N_RESULTS,
    where: dict | None = None,
    date_filter: dict | None = None,
    date_hint: str | None = None,
) -> list[dict]:
    """
    Try date-filtered retrieval first; fall back to unfiltered if no results.
    Year filtering is handled via `where` (ChromaDB integer equality) before this call.
    Specific-date filtering uses `date_filter` with a fallback to unfiltered.
    """
    augmented_query = f"{query} {date_hint}" if date_hint else query

    if date_filter:
        combined_where = merge_where_filters(where, date_filter)
        results = _retrieve_chunks_core(augmented_query, n, combined_where)
        if not results:
            results = _retrieve_chunks_core(augmented_query, n, where)
    else:
        results = _retrieve_chunks_core(augmented_query, n, where)

    return results


def _retrieve_chunks_core(query: str, n: int = N_RESULTS, where: dict | None = None) -> list[dict]:
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

    # 1. Expand query for better recall
    query_variations = expand_query(query)

    # We retrieve more chunks initially for re-ranking
    pool_size = max(n * 4, 20)  # Increased pool for re-ranking

    # Resolve allowed chunk types for BM25 filtering (handles flat and $and-wrapped filters)
    allowed_types: set | None = None
    if where:
        conditions = where.get("$and", [where])
        for condition in conditions:
            ct = condition.get("chunk_type", {})
            if isinstance(ct, str):
                allowed_types = {ct}
                break
            elif isinstance(ct, dict) and "$in" in ct:
                allowed_types = set(ct["$in"])
                break

    # 3. Semantic Search (ChromaDB) - search with multiple query variations
    semantic_ranks = {}
    for q_variation in query_variations:
        query_kwargs = {
            "query_texts": ["search_query: " + q_variation],
            "n_results": pool_size // len(query_variations),
        }
        if where:
            query_kwargs["where"] = where
        results = collection.query(**query_kwargs)
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

    # 4. Lexical Search (BM25)
    lexical_ranks = {}
    if bm25_index:
        tokenized_query = tokenize(query)
        bm25_scores = bm25_index.get_scores(tokenized_query)
        if allowed_types:
            for idx, doc_info in enumerate(bm25_docs):
                if doc_info["meta"].get("chunk_type") not in allowed_types:
                    bm25_scores[idx] = 0.0
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
2. GROUNDED — every factual claim must be a direct quote or clear paraphrase of the provided sermon excerpts. Logical inferences, theological elaborations, and anything "implied by" the text are NOT grounded — they must be explicitly stated in an excerpt.

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
- If the answer cites the provided excerpts with [1], [2], etc. while acknowledging it couldn't fully answer the question, treat it as grounded — it is acceptable to say what was found and note it doesn't match the request.
- If the answer cannot be fixed (e.g. it makes specific factual claims not in any excerpt), replace it with:
  "I couldn't find reliable information on that in the sermon transcripts. Try rephrasing or using different keywords."

Populate all four fields: safe, grounded, issues (empty list if none), and answer."""


SYSTEM_PROMPT = """You are a warm, knowledgeable assistant for a church congregation.
You help members search and understand sermon messages using an advanced retrieval system.

You will be given relevant excerpts from sermon transcripts and bible lessons as context. Each excerpt includes:
- Date and sermon title
- Chunk type (SERMON, SCRIPTURE, 30 LESSONS, or VIDEO)
- Scripture references (if applicable)
- Key topics extracted from the content

The retrieved excerpts are your ONLY source of truth — answer strictly from them.

## Core Rules
- The retrieved excerpts are your ONLY source of truth. Your own theological training, biblical knowledge, and background understanding are NOT valid sources — treat them as if they do not exist.
- A claim is only grounded if it is explicitly stated in an excerpt. Logical inferences, theological elaborations, and anything "implied by" or "consistent with" the text do not qualify.
- If the context does not explicitly answer the question, you MUST still cite the retrieved excerpts with [1], [2], etc. and explain what they do cover. For example: "The excerpts I retrieved are from [sermon titles] [1][2], but they don't appear to cover [topic]. Try rephrasing or using different keywords."
- Never say "I couldn't find information" without citing at least one retrieved excerpt to acknowledge what was found.
- Do not combine or connect excerpts unless they explicitly reference the same topic or event.
- Accuracy over completeness — an honest "I don't know" is always better than speculation.

## Citation Requirements
- ALWAYS use inline citations [1], [2], etc. to reference which excerpt you're drawing from.
- Only use citation numbers that exist in the provided sources (e.g., if given 5 sources, only use [1] through [5]).
- Place citations immediately after the relevant statement, before punctuation.
- Multiple citations are allowed: "The pastor taught about prayer and fasting [1][3]." but try to keep it as lean as possible, if chunk 1 already contains the relevant information no need to add chunk 3 for example.

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
- Prioritize SERMON-type chunks when answering biblical interpretation questions.

## Context
- Seonsaengnim is referred to as teacher in korean and he is the head pastor of our church, the sermons are given by him and when referring to the pastor, you can call him "Seongsaengim" or "the teacher" or "SSN"."""


# ── Guardrail ──────────────────────────────────────────────────────────────────

class GuardrailResult(BaseModel):
    safe: bool
    grounded: bool
    issues: List[str]
    answer: str


async def guardrail_check(question: str, context: str, answer: str) -> str:
    """
    Runs a fast guardrail LLM pass to verify the answer is safe and grounded.
    Returns the (possibly rewritten) answer.
    """
    user_content = (
        f"USER QUESTION:\n{question}\n\n"
        f"SERMON EXCERPTS:\n{context}\n\n"
        f"DRAFT ANSWER:\n{answer}"
    )

    try:
        response = await openai_client.beta.chat.completions.parse(
            model=GUARDRAIL_MODEL,
            max_tokens=2048,
            temperature=0,
            response_format=GuardrailResult,
            messages=[
                {"role": "system", "content": GUARDRAIL_PROMPT},
                {"role": "user",   "content": user_content},
            ],
        )
        result: GuardrailResult = response.choices[0].message.parsed

        if result.issues:
            print(f"⚠️  Guardrail flagged issues: {result.issues}")

        return result.answer

    except Exception as e:
        # If guardrail fails for any reason, log and pass original answer through
        print(f"⚠️  Guardrail check failed ({e}); returning original answer.")
        return answer


# ── Query rewriter (structured: search query + optional date) ──────────────────

class QueryRewrite(BaseModel):
    query: str
    date_expression: Optional[str] = None  # specific day: "last sunday", "may 18 2025"
    date_range: Optional[str] = None       # week/month range: "last week", "this month"
    year: Optional[int] = None             # whole year: 2026


async def rewrite_query_for_retrieval(message: str, history: list) -> QueryRewrite:
    """
    Rewrites the user's message into a standalone search query and extracts any
    date reference as a raw expression. Python resolves the expression to an
    actual date so the LLM never has to do date arithmetic.
    """
    recent = history[-4:] if history else []
    history_text = "\n".join(
        f"{h['role'].upper()}: {h['content'][:300]}" for h in recent
    ) if recent else "(no prior conversation)"

    current_year = datetime.date.today().year
    system_content = (
        f"The current year is {current_year}.\n"
        "You process user questions for a church sermon search system. "
        "Return JSON with four mutually exclusive date fields (set at most one):\n"
        "- `query`: standalone search query with any date/time phrase removed. "
        "Resolve pronouns using conversation history.\n"
        "- `date_expression`: a specific day reference — normalise 'recent/most recent/latest/past' to 'last'. "
        "If a day of the week is named: e.g. 'recent sunday' → 'last sunday', 'most recent wednesday' → 'last wednesday'. "
        "If no day is named but the message clearly refers to the most recent single service/sermon/message: output 'last service'. "
        "Other examples: 'this wednesday', 'may 18', 'may 18 2025'. "
        "Use for single-day references only.\n"
        "- `date_range`: a multi-day range reference — copy verbatim in lowercase. "
        "Covers relative spans ('last week', 'this week', 'last month', 'this month') "
        "AND bare month names with optional year ('may', 'february', 'may 2023', '2023 may'). "
        "When a month name has no year, copy it as-is ('may', 'february'); Python will default to the most recent occurrence.\n"
        f"- `year`: an entire-year reference — return as integer "
        f"(e.g. 'this year' → {current_year}, 'last year' → {current_year - 1}, '2024 sermons' → 2024). "
        "Only set one of the four fields; set the rest to null."
    )

    try:
        response = await openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            max_tokens=120,
            temperature=0,
            response_format=QueryRewrite,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Conversation so far:\n{history_text}\n\nUser message: {message}"},
            ],
        )
        return response.choices[0].message.parsed
    except Exception as e:
        print(f"⚠️  Query rewrite failed ({e}); using original message.")
        return QueryRewrite(query=message, date_expression=None)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def stats():
    collection = get_collection()
    return {"total_chunks": collection.count(), "collection": COLLECTION_NAME}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    # 1. Rewrite query and extract date/year in one LLM call, then retrieve
    rewrite = await rewrite_query_for_retrieval(req.message, req.history or [])
    source_filter = detect_source_filter(req.message, req.history)

    if rewrite.year:
        # Whole-year: pre-filter in ChromaDB, no fallback to all years
        year_where, date_hint = year_filter_from_year(rewrite.year)
        combined_where = merge_where_filters(source_filter, year_where)
        chunks = retrieve_chunks(rewrite.query, n=N_RESULTS_BROAD, where=combined_where, date_hint=date_hint)
    elif rewrite.date_range:
        # Week/month range: pre-filter in ChromaDB via date_numeric, fallback to unfiltered
        date_range = resolve_date_range(rewrite.date_range)
        if date_range:
            range_flt, date_hint = date_range_filter(*date_range)
            combined_where = merge_where_filters(source_filter, range_flt)
            chunks = retrieve_chunks(rewrite.query, n=N_RESULTS_BROAD, where=combined_where, date_hint=date_hint)
            if not chunks:
                chunks = retrieve_chunks(rewrite.query, n=N_RESULTS_BROAD, where=source_filter, date_hint=date_hint)
        else:
            chunks = retrieve_chunks(rewrite.query, where=source_filter)
    else:
        # Specific day (or no date): try date filter, fall back to unfiltered
        resolved_date = resolve_date_expression(rewrite.date_expression)
        date_filter, date_hint = date_filter_from_date(resolved_date)
        chunks = retrieve_chunks(
            rewrite.query,
            where=source_filter,
            date_filter=date_filter,
            date_hint=date_hint,
        )

    if not chunks:
        if rewrite.year:
            return {
                "answer": (
                    f"I couldn't find any sermons or messages from {rewrite.year} in the database. "
                    f"The collection may not have been updated with {rewrite.year} content yet — "
                    "try asking about a specific topic without a year filter."
                ),
                "sources": []
            }
        if rewrite.date_range:
            return {
                "answer": (
                    f"I couldn't find any sermons from '{rewrite.date_range}' in the database. "
                    "The collection may not cover that period yet — try a broader time frame or different keywords."
                ),
                "sources": []
            }
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