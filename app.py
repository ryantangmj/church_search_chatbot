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
N_RESULTS           = 7   # default for specific questions
N_RESULTS_BROAD     = 12  # for year/week/month range queries
MIN_SEMANTIC_SCORE  = 0.35
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
    import sys
    sys.stdout.flush()
    print("⏳ Loading embedding model...", flush=True)
    _get_embed_model()
    print("✅ Embedding model loaded.", flush=True)
    print("⏳ Building BM25 index...", flush=True)
    get_bm25_index()
    print("✅ BM25 index built.", flush=True)
    print("⏳ Loading reranker...", flush=True)
    get_reranker()
    print("✅ Reranker loaded.", flush=True)
    print("⏳ Building scripture index...", flush=True)
    get_scripture_index()
    print("✅ Scripture index built.", flush=True)
    print("✅ All startup tasks complete!", flush=True)
    sys.stdout.flush()
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

def fetch_sermons_in_range(where: dict | None, max_sermons: int = 20) -> tuple[list[dict], int]:
    """
    Listing-mode retrieval: one representative chunk per unique sermon matching `where`.
    Bypasses semantic ranking — coverage, not relevance, is the goal.

    For each (date, title) group, picks the chunk closest to chunk_index=1 (post-intro
    body content, which usually contains the sermon's main theme).

    Returns (sermons, total_count) where:
      - sermons is at most max_sermons, sorted newest-first
      - total_count is the total number of unique sermons matching the filter
        (so total_count > len(sermons) means the result is truncated)
    """
    collection = get_collection()
    try:
        all_data = collection.get(where=where, include=["documents", "metadatas"])
    except Exception as e:
        print(f"⚠️  fetch_sermons_in_range failed ({e})")
        return [], 0

    print(f"   📋 listing where: {where}")
    print(f"   📋 raw chunks returned: {len(all_data['ids'])}")

    if not all_data["ids"]:
        return [], 0

    from collections import defaultdict
    grouped: dict[tuple[str, str], list[tuple[str, dict]]] = defaultdict(list)
    for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
        key = (meta.get("date", ""), meta.get("title", ""))
        grouped[key].append((doc, meta))

    total_count = len(grouped)
    print(f"   📋 unique sermons grouped: {total_count}")

    sermons: list[dict] = []
    for chunks in grouped.values():
        # Closest to chunk_index=1 = first body chunk after intro
        chunks.sort(key=lambda x: abs(x[1].get("chunk_index", 0) - 1))
        doc, meta = chunks[0]
        sermons.append({
            "text":           doc,
            "date":           meta.get("date", ""),
            "title":          meta.get("title", ""),
            "chunk_type":     meta.get("chunk_type", "sermon"),
            "chunk_index":    meta.get("chunk_index", 0),
            "total_chunks":   meta.get("total_chunks", 0),
            "scripture_refs": meta.get("scripture_refs", ""),
            "topics":         meta.get("topics", ""),
            "section_type":   meta.get("section_type", "body"),
            "score":          0.99,
        })

    sermons.sort(key=lambda s: s["date"], reverse=True)
    return sermons[:max_sermons], total_count


def day_of_week_filter(dow: str | None) -> dict | None:
    """Build a ChromaDB equality filter for day_of_week."""
    if not dow:
        return None
    return {"day_of_week": {"$eq": dow}}


def get_latest_dates_for_dow(day_of_week: str, n: int = 1) -> list[datetime.date]:
    """Return the N most recent distinct sermon dates for the given day_of_week."""
    collection = get_collection()
    try:
        results = collection.get(
            where={"day_of_week": {"$eq": day_of_week}},
            include=["metadatas"],
        )
    except Exception:
        return []
    if not results["metadatas"]:
        return []
    unique_numerics = sorted(
        {m["date_numeric"] for m in results["metadatas"] if m.get("date_numeric")},
        reverse=True,
    )
    dates = []
    for dn in unique_numerics[:n]:
        y, rest = divmod(dn, 10000)
        mo, d = divmod(rest, 100)
        try:
            dates.append(datetime.date(y, mo, d))
        except ValueError:
            continue
    return dates


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
    "obedience": ["obeying", "submitting", "following God's commands", "faithful living"],
    "righteousness": ["righteous", "right standing with God", "justified", "justification"],
    "discipleship": ["disciple", "following jesus", "commitment", "dedication"],
    "humility": ["humble", "humbleness", "lowliness", "meekness"],
    "word of god": ["scripture", "bible", "god's word", "the word"],
    "kingdom": ["kingdom of god", "kingdom of heaven", "god's kingdom", "eternal kingdom"],
    "suffering": ["trials", "tribulation", "hardship", "persecution", "endurance"],
    "hope": ["hopeful", "expectation", "assurance", "confidence in God"],
    "peace": ["peaceful", "shalom", "rest in God", "comfort"],
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

# ── Scripture reference index ─────────────────────────────────────────────────

_BIBLE_BOOKS = [
    # Old Testament
    "genesis", "exodus", "leviticus", "numbers", "deuteronomy",
    "joshua", "judges", "ruth",
    "1 samuel", "2 samuel", "1 kings", "2 kings",
    "1 chronicles", "2 chronicles",
    "ezra", "nehemiah", "esther", "job", "psalms", "proverbs",
    "ecclesiastes", "song of solomon", "isaiah", "jeremiah", "lamentations",
    "ezekiel", "daniel", "hosea", "joel", "amos", "obadiah", "jonah",
    "micah", "nahum", "habakkuk", "zephaniah", "haggai", "zechariah", "malachi",
    # New Testament
    "matthew", "mark", "luke", "john", "acts", "romans",
    "1 corinthians", "2 corinthians", "galatians", "ephesians",
    "philippians", "colossians",
    "1 thessalonians", "2 thessalonians",
    "1 timothy", "2 timothy", "titus", "philemon", "hebrews", "james",
    "1 peter", "2 peter", "1 john", "2 john", "3 john", "jude", "revelation",
]
BIBLE_BOOKS_SET = set(_BIBLE_BOOKS)

BOOK_ABBREVS = {
    "gen": "genesis",
    "ex": "exodus", "exod": "exodus",
    "lev": "leviticus", "num": "numbers",
    "deut": "deuteronomy", "dt": "deuteronomy",
    "josh": "joshua", "judg": "judges",
    "1 sam": "1 samuel", "1sam": "1 samuel",
    "2 sam": "2 samuel", "2sam": "2 samuel",
    "1 kgs": "1 kings", "1kgs": "1 kings", "1 ki": "1 kings",
    "2 kgs": "2 kings", "2kgs": "2 kings", "2 ki": "2 kings",
    "1 chr": "1 chronicles", "1chr": "1 chronicles",
    "2 chr": "2 chronicles", "2chr": "2 chronicles",
    "neh": "nehemiah", "est": "esther",
    "ps": "psalms", "psa": "psalms", "psalm": "psalms",
    "prov": "proverbs", "pr": "proverbs",
    "eccl": "ecclesiastes", "ecc": "ecclesiastes",
    "song": "song of solomon", "sos": "song of solomon",
    "isa": "isaiah", "jer": "jeremiah", "lam": "lamentations",
    "ezek": "ezekiel", "ezk": "ezekiel", "dan": "daniel",
    "hos": "hosea", "obad": "obadiah", "ob": "obadiah",
    "mic": "micah", "nah": "nahum", "hab": "habakkuk",
    "zeph": "zephaniah", "hag": "haggai", "zech": "zechariah", "mal": "malachi",
    "mt": "matthew", "matt": "matthew",
    "mk": "mark", "mrk": "mark",
    "lk": "luke",
    "jn": "john", "joh": "john",
    "rom": "romans",
    "1 cor": "1 corinthians", "1cor": "1 corinthians",
    "2 cor": "2 corinthians", "2cor": "2 corinthians",
    "gal": "galatians", "eph": "ephesians",
    "phil": "philippians", "col": "colossians",
    "1 thess": "1 thessalonians", "1thess": "1 thessalonians", "1 th": "1 thessalonians",
    "2 thess": "2 thessalonians", "2thess": "2 thessalonians", "2 th": "2 thessalonians",
    "1 tim": "1 timothy", "1tim": "1 timothy",
    "2 tim": "2 timothy", "2tim": "2 timothy",
    "tit": "titus", "phlm": "philemon",
    "heb": "hebrews", "jas": "james", "jms": "james",
    "1 pet": "1 peter", "1pet": "1 peter",
    "2 pet": "2 peter", "2pet": "2 peter",
    "1 jn": "1 john", "1jn": "1 john",
    "2 jn": "2 john", "2jn": "2 john",
    "3 jn": "3 john", "3jn": "3 john",
    "rev": "revelation",
}


def _build_scripture_query_pattern():
    """Regex matching any Bible book name + chapter (with optional :verse[-verse_end])."""
    all_names = list(BIBLE_BOOKS_SET) + list(BOOK_ABBREVS.keys())
    all_names.sort(key=len, reverse=True)  # longest first so "1 samuel" beats "samuel"
    book_alt = "|".join(re.escape(n) for n in all_names)
    return re.compile(
        rf'\b({book_alt})\.?\s+(\d+)(?::(\d+))?(?:-(\d+))?\b',
        re.IGNORECASE,
    )

SCRIPTURE_QUERY_RE = _build_scripture_query_pattern()


def parse_scripture_ref(ref: str) -> tuple[str, int, int, int] | None:
    """Parse a 'Book Chapter:Verse' string from chunk metadata into structured form."""
    if not ref or not ref.strip():
        return None
    m = re.match(
        r'^\s*([1-3]?\s*[A-Za-z][A-Za-z\s]*?)\s+(\d+):(\d+)(?:-(\d+))?',
        ref.strip(),
    )
    if not m:
        return None
    book_raw = re.sub(r'\s+', ' ', m.group(1).strip().lower())
    book = BOOK_ABBREVS.get(book_raw, book_raw)
    if book not in BIBLE_BOOKS_SET:
        return None
    chapter = int(m.group(2))
    vs = int(m.group(3))
    ve = int(m.group(4)) if m.group(4) else vs
    return (book, chapter, vs, ve)


def extract_scripture_refs_from_query(query: str) -> list[tuple[str, int, int | None]]:
    """Find scripture references in a user query. Returns list of (book, chapter, verse_or_None)."""
    refs = []
    for m in SCRIPTURE_QUERY_RE.finditer(query):
        book_raw = re.sub(r'\s+', ' ', m.group(1).strip().lower())
        book = BOOK_ABBREVS.get(book_raw, book_raw)
        if book not in BIBLE_BOOKS_SET:
            continue
        chapter = int(m.group(2))
        verse = int(m.group(3)) if m.group(3) else None
        refs.append((book, chapter, verse))
    # Dedupe while preserving order
    seen = set()
    out = []
    for r in refs:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


_scripture_index = None  # {book: {chapter: [doc_id, ...]}}

def get_scripture_index() -> dict:
    global _scripture_index
    if _scripture_index is None:
        _scripture_index = _build_scripture_index()
    return _scripture_index


def _build_scripture_index() -> dict:
    """Walk every chunk's scripture_refs metadata and build a book→chapter→chunk_ids index."""
    print("⏳ Building scripture reference index...")
    collection = get_collection()
    all_data = collection.get(include=["metadatas"])
    index: dict[str, dict[int, list[str]]] = {}
    for doc_id, meta in zip(all_data["ids"], all_data["metadatas"]):
        refs_str = meta.get("scripture_refs", "") or ""
        if not refs_str:
            continue
        for raw_ref in refs_str.split(","):
            parsed = parse_scripture_ref(raw_ref)
            if not parsed:
                continue
            book, chapter, _, _ = parsed
            index.setdefault(book, {}).setdefault(chapter, []).append(doc_id)
    total_pairs = sum(len(ids) for chs in index.values() for ids in chs.values())
    print(f"✅ Scripture index built: {len(index)} books, {total_pairs} (chapter, chunk) entries.")
    return index


def lookup_chunks_by_scripture(refs: list[tuple[str, int, int | None]]) -> set[str]:
    """Given a list of (book, chapter, verse) refs, return matching chunk IDs."""
    if not refs:
        return set()
    index = get_scripture_index()
    matched: set[str] = set()
    for book, chapter, _verse in refs:
        chapter_dict = index.get(book, {})
        if chapter in chapter_dict:
            matched.update(chapter_dict[chapter])
    return matched


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


def _apply_recency_bias(candidates: list[dict], weight: float) -> None:
    """Blend recency into each candidate's final_score. Mutates in place. weight in [0,1]."""
    if weight <= 0 or not candidates:
        return
    today = datetime.date.today()
    for c in candidates:
        try:
            d = datetime.date.fromisoformat(c.get("date", ""))
            days_old = max(0, (today - d).days)
            recency_score = math.exp(-days_old / 365.0)  # 1 year half-life ~ 0.5
        except (ValueError, TypeError):
            continue
        c["final_score"] = (1 - weight) * c["final_score"] + weight * recency_score


def _apply_mmr(candidates: list[dict], lambda_: float = 0.7, n: int = 10) -> list[dict]:
    """
    Maximal Marginal Relevance reranking for diversity.
    Penalty combines sermon-level overlap (same title+date) with token-level Jaccard.
    Expects candidates pre-sorted by relevance (uses final_score).
    """
    if not candidates:
        return []
    if len(candidates) <= n:
        return list(candidates)

    token_cache: dict[int, set] = {}
    def _tokens(c):
        cid = id(c)
        if cid not in token_cache:
            token_cache[cid] = set(c["text"].lower().split())
        return token_cache[cid]

    remaining = list(candidates)
    selected = [remaining.pop(0)]

    while remaining and len(selected) < n:
        best_idx = -1
        best_score = -float("inf")
        for i, cand in enumerate(remaining):
            cand_tokens = _tokens(cand)
            max_sim = 0.0
            for sel in selected:
                same_sermon = (
                    (cand.get("title"), cand.get("date")) == (sel.get("title"), sel.get("date"))
                )
                sel_tokens = _tokens(sel)
                union = cand_tokens | sel_tokens
                jaccard = (len(cand_tokens & sel_tokens) / len(union)) if union else 0.0
                sim = max(0.5 if same_sermon else 0.0, jaccard)
                if sim > max_sim:
                    max_sim = sim
            mmr = lambda_ * cand["final_score"] - (1 - lambda_) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_idx = i
        if best_idx == -1:
            break
        selected.append(remaining.pop(best_idx))

    return selected


def retrieve_chunks(
    query: str,
    n: int = N_RESULTS,
    where: dict | None = None,
    date_filter: dict | None = None,
    date_hint: str | None = None,
    query_variations: list[str] | None = None,
    priority_chunk_ids: set | None = None,
    recency_weight: float = 0.0,
) -> list[dict]:
    """
    Try date-filtered retrieval first; fall back to unfiltered if no results.
    Year filtering is handled via `where` (ChromaDB integer equality) before this call.
    Specific-date filtering uses `date_filter` with a fallback to unfiltered.
    """
    augmented_query = f"{query} {date_hint}" if date_hint else query

    if date_filter:
        combined_where = merge_where_filters(where, date_filter)
        results = _retrieve_chunks_core(
            augmented_query, n, combined_where,
            query_variations=query_variations,
            priority_chunk_ids=priority_chunk_ids,
            recency_weight=recency_weight,
        )
        if not results:
            results = _retrieve_chunks_core(
                augmented_query, n, where,
                query_variations=query_variations,
                priority_chunk_ids=priority_chunk_ids,
                recency_weight=recency_weight,
            )
    else:
        results = _retrieve_chunks_core(
            augmented_query, n, where,
            query_variations=query_variations,
            priority_chunk_ids=priority_chunk_ids,
            recency_weight=recency_weight,
        )

    return results


def _extract_bm25_constraints(where: dict | None) -> dict:
    """Parse a ChromaDB where-clause into simple constraints for BM25 filtering."""
    c = {
        "allowed_types": None,
        "allowed_days": None,
        "required_date_numeric": None,
        "min_date_numeric": None,
        "max_date_numeric": None,
        "required_year": None,
    }
    if not where:
        return c
    conditions = where.get("$and", [where])
    for cond in conditions:
        ct = cond.get("chunk_type", {})
        if isinstance(ct, dict) and "$in" in ct:
            c["allowed_types"] = set(ct["$in"])
        elif isinstance(ct, str):
            c["allowed_types"] = {ct}

        dow = cond.get("day_of_week", {})
        if isinstance(dow, dict) and "$eq" in dow:
            c["allowed_days"] = {dow["$eq"]}

        dn = cond.get("date_numeric", {})
        if isinstance(dn, dict):
            if "$eq" in dn:
                c["required_date_numeric"] = dn["$eq"]
            if "$gte" in dn:
                c["min_date_numeric"] = dn["$gte"]
            if "$lte" in dn:
                c["max_date_numeric"] = dn["$lte"]

        yr = cond.get("year", {})
        if isinstance(yr, dict) and "$eq" in yr:
            c["required_year"] = yr["$eq"]
    return c


def _retrieve_chunks_core(
    query: str,
    n: int = N_RESULTS,
    where: dict | None = None,
    query_variations: list[str] | None = None,
    priority_chunk_ids: set | None = None,
    recency_weight: float = 0.0,
    use_mmr: bool = True,
) -> list[dict]:
    """
    Hybrid retrieval pipeline:
    - Multi-query expansion (LLM paraphrases passed in OR hardcoded synonym fallback)
    - Scripture-reference priority retrieval (deterministic book/chapter lookup)
    - Semantic + BM25 + scripture, fused via RRF
    - Cross-encoder reranking
    - Recency-weighted final scoring for time-sensitive queries
    - MMR for cross-sermon diversity
    """
    collection = get_collection()
    bm25_index, bm25_docs = get_bm25_index()
    reranker = get_reranker()

    # 1. Query variations (passed-in LLM paraphrases or fallback synonym expansion)
    if query_variations is None or not query_variations:
        query_variations = expand_query(query)

    pool_size = max(n * 4, 20)

    # 2. Extract BM25 filter constraints from the ChromaDB where-clause
    bm25_c = _extract_bm25_constraints(where)

    # 3. Semantic Search (multi-query) — each variation pre-filtered by `where`
    semantic_ranks = {}
    per_variation = max(1, pool_size // len(query_variations))
    for q_variation in query_variations:
        query_kwargs = {
            "query_texts": ["search_query: " + q_variation],
            "n_results": per_variation,
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
                    "semantic_score": round(1 - dist, 3),
                }

    # 4. Lexical Search (BM25)
    lexical_ranks = {}
    if bm25_index:
        tokenized_query = tokenize(query)
        bm25_scores = bm25_index.get_scores(tokenized_query)
        if any(v is not None for v in bm25_c.values()):
            for idx, doc_info in enumerate(bm25_docs):
                meta = doc_info["meta"]
                if bm25_c["allowed_types"] and meta.get("chunk_type") not in bm25_c["allowed_types"]:
                    bm25_scores[idx] = 0.0
                    continue
                if bm25_c["allowed_days"] and meta.get("day_of_week") not in bm25_c["allowed_days"]:
                    bm25_scores[idx] = 0.0
                    continue
                if bm25_c["required_date_numeric"] is not None and meta.get("date_numeric") != bm25_c["required_date_numeric"]:
                    bm25_scores[idx] = 0.0
                    continue
                if bm25_c["min_date_numeric"] is not None and meta.get("date_numeric", 0) < bm25_c["min_date_numeric"]:
                    bm25_scores[idx] = 0.0
                    continue
                if bm25_c["max_date_numeric"] is not None and meta.get("date_numeric", 0) > bm25_c["max_date_numeric"]:
                    bm25_scores[idx] = 0.0
                    continue
                if bm25_c["required_year"] is not None and meta.get("year") != bm25_c["required_year"]:
                    bm25_scores[idx] = 0.0
        top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:pool_size]
        for rank, idx in enumerate(top_indices):
            if bm25_scores[idx] > 0:
                doc_info = bm25_docs[idx]
                lexical_ranks[doc_info["id"]] = {"rank": rank + 1, **doc_info}

    # 5. Scripture-reference priority retrieval (also respects `where`)
    scripture_ranks = {}
    if priority_chunk_ids:
        try:
            get_kwargs = {"ids": list(priority_chunk_ids), "include": ["documents", "metadatas"]}
            if where:
                get_kwargs["where"] = where
            sr_data = collection.get(**get_kwargs)
            for rank, (doc_id, doc, meta) in enumerate(zip(
                sr_data["ids"], sr_data["documents"], sr_data["metadatas"]
            )):
                scripture_ranks[doc_id] = {
                    "rank": rank + 1,
                    "text": doc,
                    "meta": meta,
                }
        except Exception as e:
            print(f"⚠️  Scripture-priority fetch failed ({e}); continuing without it.")

    # 6. Reciprocal Rank Fusion (semantic + lexical + scripture)
    rrf_k = 60
    combined = {}
    all_ids = set(semantic_ranks.keys()) | set(lexical_ranks.keys()) | set(scripture_ranks.keys())

    for doc_id in all_ids:
        rrf_score = 0.0
        doc_data = semantic_ranks.get(doc_id) or lexical_ranks.get(doc_id) or scripture_ranks.get(doc_id)

        if doc_id in semantic_ranks:
            rrf_score += 1.0 / (rrf_k + semantic_ranks[doc_id]["rank"])
        if doc_id in lexical_ranks:
            rrf_score += 1.0 / (rrf_k + lexical_ranks[doc_id]["rank"])
        if doc_id in scripture_ranks:
            # 2× weight: a direct ref match is high-precision signal
            rrf_score += 2.0 / (rrf_k + scripture_ranks[doc_id]["rank"])

        meta = doc_data["meta"]
        sem_score = semantic_ranks[doc_id]["semantic_score"] if doc_id in semantic_ranks else 0.0
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
            "semantic_score": sem_score,
            "is_scripture_match": doc_id in scripture_ranks,
        }

    # Semantic score floor — exempt scripture matches (their relevance is structural, not semantic)
    combined = {
        k: v for k, v in combined.items()
        if v["semantic_score"] >= MIN_SEMANTIC_SCORE or v["is_scripture_match"]
    }
    if not combined:
        return []

    # 7. Cross-encoder reranking on top RRF candidates
    sorted_by_rrf = sorted(combined.values(), key=lambda x: x["rrf_score"], reverse=True)
    rerank_candidates = sorted_by_rrf[: max(pool_size // 2, n)]

    if rerank_candidates and reranker is not None:
        pairs = [[query, candidate["text"]] for candidate in rerank_candidates]
        ce_scores = reranker.predict(pairs)
        ce_normalized = [1.0 / (1.0 + math.exp(-float(s))) for s in ce_scores]
        max_rrf = max(c["rrf_score"] for c in rerank_candidates) or 1.0

        for candidate, ce_raw, ce_norm in zip(rerank_candidates, ce_scores, ce_normalized):
            candidate["rerank_score"] = float(ce_raw)
            rrf_norm = candidate["rrf_score"] / max_rrf
            candidate["final_score"] = 0.6 * ce_norm + 0.4 * rrf_norm

        # 8. Recency bias (mutates final_score; no-op when weight == 0)
        _apply_recency_bias(rerank_candidates, recency_weight)

        # Re-sort by final score after recency adjustment
        reranked = sorted(rerank_candidates, key=lambda x: x["final_score"], reverse=True)

        # Set display score early so deduplicate_chunks (which compares "score") works
        for res in reranked:
            display_score = min(0.99, max(0.5, (res["rerank_score"] + 1) / 12))
            res["score"] = round(display_score, 3)

        # 9. Dedupe adjacent overlapping chunks from the same sermon
        deduped = deduplicate_chunks(reranked[: max(n * 3, n + 5)])

        # 10. MMR for cross-sermon diversity
        selected = _apply_mmr(deduped, lambda_=0.7, n=n) if use_mmr else deduped[:n]

        # Strip internal scoring fields
        for res in selected:
            for k in ("rrf_score", "semantic_score", "rerank_score", "final_score", "id", "is_scripture_match"):
                res.pop(k, None)
        return selected

    # Fallback if reranker unavailable
    for res in sorted_by_rrf:
        res["score"] = round(res["semantic_score"], 3)
        for k in ("rrf_score", "semantic_score", "id", "is_scripture_match"):
            res.pop(k, None)
    return sorted_by_rrf[:n]


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        label = f"[{i}] {c['date']} — {c['title']} ({c['chunk_type'].upper()})"
        if c["scripture_refs"]:
            label += f" | Scriptures: {c['scripture_refs']}"
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


# Cross-encoder threshold for citation verification.
# ms-marco-MiniLM scores: ~< -2 = clearly irrelevant, -2..0 = weakly related,
# 0..5 = relevant, > 5 = strongly relevant. We're checking "did the LLM cite a
# clearly-wrong chunk?", not "is this the optimal source", so we want a permissive
# floor that catches obvious mistakes without rejecting legitimate paraphrases.
CITATION_CE_THRESHOLD = -0.5


def verify_citations(answer: str, chunks: list[dict]) -> tuple[str, int]:
    """
    Backstop verification for hallucinated citations.

    The guardrail already approved the answer as grounded; this only catches
    citations that point to clearly-wrong chunks. Per sentence, the rule is
    all-or-nothing: if AT LEAST ONE cited chunk scores above threshold, all
    citations in the sentence are kept (the LLM correctly identified a relevant
    source — minor over-citation alongside it is preserved). If NO cited chunk
    clears the threshold, every citation in the sentence is dropped (likely a
    fabricated grouping).

    Returns (cleaned_answer, citations_removed).
    """
    reranker = get_reranker()
    if reranker is None or not chunks or not answer:
        return answer, 0

    sentence_re = re.compile(r'[^.!?\n]+[.!?\n]+\s*|[^.!?\n]+\Z')

    output_parts: list[str] = []
    removed = 0

    for m in sentence_re.finditer(answer):
        sent = m.group(0)
        cite_nums = [int(x) for x in re.findall(r'\[(\d+)\]', sent)]
        if not cite_nums:
            output_parts.append(sent)
            continue

        # Strip citation markers to get the bare claim text
        bare = re.sub(r'\s*\[\d+\]\s*', ' ', sent).strip()
        bare = re.sub(r'\s+', ' ', bare)
        if len(bare) < 20:
            output_parts.append(sent)
            continue

        unique_cites = list(dict.fromkeys(c for c in cite_nums if 1 <= c <= len(chunks)))
        if not unique_cites:
            output_parts.append(sent)
            continue

        try:
            pairs = [[bare, chunks[c - 1]["text"]] for c in unique_cites]
            scores = reranker.predict(pairs)
            any_supported = any(float(s) > CITATION_CE_THRESHOLD for s in scores)
        except Exception as e:
            print(f"⚠️  Citation verification scoring failed ({e}); keeping citations.")
            output_parts.append(sent)
            continue

        if any_supported:
            # At least one cited chunk is relevant — trust the LLM's grouping
            output_parts.append(sent)
            continue

        # None of the cited chunks are relevant — strip them all
        removed += len(cite_nums)
        cleaned = re.sub(r'\[(\d+)\]', "", sent)
        cleaned = re.sub(r'\s+([.,!?])', r'\1', cleaned)
        cleaned = re.sub(r'  +', ' ', cleaned)
        output_parts.append(cleaned)

    return "".join(output_parts), removed


GUARDRAIL_MODEL = "gpt-4o-mini"

GUARDRAIL_PROMPT = """You are a guardrail for a church sermon chatbot. Review the draft answer for two things only:

1. GROUNDED — every factual claim must be a direct quote or paraphrase from the provided sermon excerpts. Logical inferences and theological elaborations beyond what is explicitly stated are not grounded.
2. SAFE — appropriate for a church congregation. Not harmful, offensive, or misleading.

Rules:
- If the answer is grounded and safe, return it unchanged.
- If there are problems, fix them with minimal changes — trim or correct specific ungrounded claims rather than rewriting the whole answer.
- Preserve the original voice. Do NOT introduce phrases like "Based on the excerpts", "Certainly!", "I hope this helps", or any closing pleasantry.
- Keep all citations [1], [2], etc. intact and accurate.
- An answer that cites excerpts while honestly acknowledging it couldn't fully answer the question is grounded — do not rewrite it.
- If the answer cannot be fixed (makes specific factual claims absent from all excerpts), replace it with: "I didn't find anything in the transcripts that directly covers that. Try rephrasing or using different keywords."

Populate all four fields: safe, grounded, issues (empty list if none), and answer."""


SYSTEM_PROMPT = """You help members of a church congregation search and understand content from Seonsaengnim's sermon transcripts and Bible lesson notes. The archive covers 2023 to the present.

Seonsaengnim means "teacher" in Korean — he is the head pastor. Refer to him as "Seonsaengnim", "the teacher", or "the pastor".

You have been given numbered excerpts from the archive. These are your only source. Do not draw on outside theological knowledge, even when you think you could add helpful context.

## Grounding (strict)
- Only state things explicitly said in the excerpts. Inferences, elaborations, and anything "implied by" the text are not grounded — don't say them.
- Cite inline with [1], [2], etc. right after the relevant statement, before punctuation. Cite the SINGLE chunk that most directly supports each claim — only bundle multiple citations like [1][2] when the claim genuinely combines distinct pieces of information from each.
- If excerpts don't answer the question, cite what was found and acknowledge the gap honestly: "The messages I found — [title][1] — cover [X], not [Y]. You might try searching for [related term]."
- Never say "I couldn't find information" without citing at least one excerpt to show what was retrieved.
- If sources say different things, say so plainly — don't smooth it over.

## Voice and format
- Write the way a knowledgeable church member would explain something to a fellow member — direct, natural, and conversational. Not a formal report.
- Short answers for simple questions (2–4 sentences). Longer only if the topic genuinely requires it.
- Prose over bullet points. No section headers (##, ###) in answers unless absolutely necessary.
- Bold only scripture references (e.g. **Ephesians 4:22**) or a term Seonsaengnim specifically highlighted. Not for general emphasis.
- Direct quotes from Seonsaengnim go in quotation marks.
- Scripture references must be complete: **John 3:16**, not just "John 3".

## Phrases never to use
- "Certainly!", "Great question!", "Absolutely!", "Of course!", "Sure!", "I'd be happy to"
- "Based on the provided excerpts", "According to the transcripts", "The context shows", "The sermon excerpts indicate"
- "I hope this helps", "Feel free to ask more", "Is there anything else I can help with?"
- "As an AI" or any variation"""


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
    date_expression: Optional[str] = None  # specific calendar day: "last sunday", "may 18 2025"
    date_range: Optional[str] = None       # week/month range: "last week", "this month", "may 2023"
    year: Optional[int] = None             # whole year: 2026
    day_of_week: Optional[str] = None      # "sunday" or "wednesday" when a service day is specified
    recency: Optional[str] = None          # "latest" or "last_n"
    n: Optional[int] = None               # used when recency is "last_n"
    listing_intent: bool = False           # True if user wants a summary/listing across multiple sermons
    assumption: Optional[str] = None       # short explanation of any temporal assumption made


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

    today = datetime.date.today()
    system_content = (
        f"Today is {today.strftime('%Y-%m-%d')} ({today.strftime('%A')}). "
        "Services are held on Sundays and Wednesdays only.\n"
        "You process user questions for a church sermon search system. "
        "Return JSON with these fields (set at most ONE of date_expression / date_range / year):\n"
        "- `query`: standalone search query with ALL date/time/recency phrases removed. Resolve pronouns using conversation history.\n"
        "- `day_of_week`: if the query specifies a service day, set to \"sunday\" or \"wednesday\". Otherwise null. "
        "Examples: 'Sunday message' → \"sunday\", 'last Wednesday sermon' → \"wednesday\", 'most recent sermon' → null.\n"
        "- `recency`: set \"latest\" when the user wants the single most recent item for a given day/type "
        "WITHOUT referencing a specific calendar date (e.g. 'the most recent Sunday message', 'latest Wednesday sermon'). "
        "Set \"last_n\" when the user wants the N most recent items. Otherwise null. "
        "Do NOT set recency when the user references a specific calendar date like 'last Sunday' or 'May 18'.\n"
        "- `n`: integer only when recency is \"last_n\" (e.g. 'the last 3 Wednesday messages' → n=3). Otherwise null.\n"
        "- `date_expression`: a specific calendar-day reference — only when the user means a KNOWN date, not just 'the latest'. "
        "Normalise 'recent/most recent/latest/past' → 'last' ONLY when paired with a specific named day. "
        "Examples: 'last sunday', 'this wednesday', 'may 18', 'may 18 2025'. "
        "If no day is named but user wants the most recent single sermon/service, use recency='latest' instead.\n"
        "- `date_range`: a multi-day range — copy verbatim in lowercase. "
        "Covers: 'last week', 'this week', 'last month', 'this month', AND any specific calendar month "
        "('may', 'february', 'may 2023', 'the month of may' → 'may', 'across may' → 'may', 'in june' → 'june'). "
        "ALWAYS use date_range for month-scoped queries, even when the user says 'the month of X' or 'across X' — "
        "extract just the bare month name. "
        "When a month name has no year, copy it as-is; Python will resolve to the most recent occurrence.\n"
        f"- `year`: an entire-year reference as integer "
        f"(e.g. 'this year' → {today.year}, 'last year' → {today.year - 1}, '2024 sermons' → 2024). "
        "Only use year for WHOLE-YEAR queries with NO month narrowing — never combine year with a month reference.\n"
        "- `listing_intent`: true if the user wants a SUMMARY, OVERVIEW, RECAP, or LISTING across multiple sermons "
        "(e.g. 'summarize May messages', 'what was preached last month', 'walk me through last quarter', "
        "'overview of recent Sunday sermons', 'list the topics covered'). "
        "False for focused/specific questions about one topic, one sermon, or one date.\n"
        "- `assumption`: if you make any temporal assumption (e.g. assumed current year for a bare month), describe it briefly. Otherwise null."
    )

    try:
        response = await openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            max_tokens=200,
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


# ── Multi-query paraphrase generator (replaces hardcoded synonym map) ──────────

class QueryParaphrases(BaseModel):
    paraphrases: List[str]


async def generate_query_paraphrases(query: str, n: int = 3) -> List[str]:
    """
    Generate N diverse paraphrases of a query for multi-query retrieval.
    Returns [original, paraphrase_1, paraphrase_2, ...] deduped, capped at n+1 total.
    Falls back to [query] if the LLM call fails.
    """
    if not query or not query.strip():
        return [query]
    try:
        response = await openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            max_tokens=250,
            temperature=0,
            response_format=QueryParaphrases,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Generate exactly {n} short paraphrases of the user's question for a church sermon search. "
                        "Each paraphrase should explore a different angle: synonyms, related theological concepts, "
                        "or a more specific/general phrasing. Keep each under 15 words. "
                        "Do not repeat the original wording verbatim. "
                        "Output strictly as JSON matching the schema."
                    ),
                },
                {"role": "user", "content": query},
            ],
        )
        result = response.choices[0].message.parsed
        candidates = [query] + [p for p in (result.paraphrases or []) if p and p.strip()]
        seen: set = set()
        unique: list[str] = []
        for q in candidates:
            key = q.strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique.append(q.strip())
        return unique[: n + 1]
    except Exception as e:
        print(f"⚠️  Paraphrase generation failed ({e}); falling back to original query.")
        return [query]


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/api/stats")
async def stats():
    collection = get_collection()
    return {"total_chunks": collection.count(), "collection": COLLECTION_NAME}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    # 1. Rewrite query and extract date/year/day-of-week in one LLM call, then retrieve
    rewrite = await rewrite_query_for_retrieval(req.message, req.history or [])
    source_filter = detect_source_filter(req.message, req.history)
    dow_filter = day_of_week_filter(rewrite.day_of_week)
    if dow_filter:
        # Day-of-week is only meaningful for service messages, not lessons
        lesson_only = (
            source_filter is not None
            and source_filter.get("chunk_type", {}).get("$in", []) == list(LESSON_CHUNK_TYPES)
        )
        if lesson_only:
            dow_filter = None  # user asked about lessons; DOW doesn't apply
        elif source_filter is None:
            # Restrict to sermon types so lesson chunks aren't pulled in
            source_filter = {"chunk_type": {"$in": list(SERMON_CHUNK_TYPES)}}

    # 1a. Multi-query paraphrase expansion (replaces hardcoded synonym map)
    query_variations = await generate_query_paraphrases(rewrite.query, n=3)

    # 1b. Detect scripture references in the query → priority chunk IDs
    scripture_refs = extract_scripture_refs_from_query(rewrite.query)
    priority_chunk_ids = lookup_chunks_by_scripture(scripture_refs) if scripture_refs else None
    if priority_chunk_ids:
        print(f"📖  Scripture refs in query: {scripture_refs} → {len(priority_chunk_ids)} matching chunks")

    # 1c. Recency weight for time-sensitive queries
    recency_weight = 0.0
    if rewrite.recency:
        recency_weight = 0.4
    elif rewrite.date_range:
        recency_weight = 0.2

    retrieval_kwargs = dict(
        query_variations=query_variations,
        priority_chunk_ids=priority_chunk_ids,
        recency_weight=recency_weight,
    )

    # 1d. Listing path: "summarize/list across a date range" — bypass relevance
    # ranking and return one chunk per sermon for full coverage.
    # The listing_intent flag is set by the rewrite LLM call (more robust than regex).
    listing_mode = False
    listing_total = 0
    listing_max = 20
    chunks: list[dict] | None = None
    if rewrite.listing_intent and (rewrite.year or rewrite.date_range):
        # Defensive: prefer date_range over year if both were set (more specific scope)
        listing_where = None
        if rewrite.date_range:
            date_range = resolve_date_range(rewrite.date_range)
            if date_range:
                range_flt, _ = date_range_filter(*date_range)
                listing_where = merge_where_filters(source_filter, dow_filter, range_flt)
        if listing_where is None and rewrite.year:
            year_where, _ = year_filter_from_year(rewrite.year)
            listing_where = merge_where_filters(source_filter, dow_filter, year_where)

        if listing_where is not None:
            # If no source filter was set, restrict to sermons (lessons aren't service-dated)
            if not source_filter:
                sermon_only = {"chunk_type": {"$in": list(SERMON_CHUNK_TYPES)}}
                listing_where = merge_where_filters(listing_where, sermon_only)

            result, total = fetch_sermons_in_range(listing_where, max_sermons=listing_max)
            if result:
                listing_mode = True
                chunks = result
                listing_total = total
                if total > len(result):
                    print(f"📋  Listing mode: showing {len(result)} of {total} sermon(s) (truncated, newest first).")
                else:
                    print(f"📋  Listing mode: {len(result)} sermon(s) in range.")

    if not listing_mode:
        if rewrite.recency == "latest" and rewrite.day_of_week:
            # Find the actual most recent date in the DB for this day of week
            latest_dates = get_latest_dates_for_dow(rewrite.day_of_week, 1)
            base_where = merge_where_filters(source_filter, dow_filter)
            if latest_dates:
                date_flt, date_hint = date_filter_from_date(latest_dates[0])
                chunks = retrieve_chunks(rewrite.query, where=base_where, date_filter=date_flt, date_hint=date_hint, **retrieval_kwargs)
            else:
                chunks = retrieve_chunks(rewrite.query, where=base_where, **retrieval_kwargs)
        elif rewrite.recency == "latest" and not rewrite.day_of_week:
            # Most recent sermon (any day): find most recent service date regardless of day
            most_recent = most_recent_service_date()
            date_flt, date_hint = date_filter_from_date(most_recent)
            base_where = merge_where_filters(source_filter, dow_filter)
            chunks = retrieve_chunks(rewrite.query, where=base_where, date_filter=date_flt, date_hint=date_hint, **retrieval_kwargs)
            if not chunks:
                chunks = retrieve_chunks(rewrite.query, where=base_where, **retrieval_kwargs)
        elif rewrite.recency == "last_n" and rewrite.day_of_week and rewrite.n:
            # Find the last N distinct dates for this day of week and cover that range
            latest_dates = get_latest_dates_for_dow(rewrite.day_of_week, rewrite.n)
            if latest_dates:
                range_flt, date_hint = date_range_filter(latest_dates[-1], latest_dates[0])
                combined_where = merge_where_filters(source_filter, dow_filter, range_flt)
            else:
                combined_where = merge_where_filters(source_filter, dow_filter)
                date_hint = None
            chunks = retrieve_chunks(rewrite.query, n=N_RESULTS_BROAD, where=combined_where, date_hint=date_hint, **retrieval_kwargs)
        elif rewrite.year:
            # Whole-year: pre-filter in ChromaDB, no fallback to all years
            year_where, date_hint = year_filter_from_year(rewrite.year)
            combined_where = merge_where_filters(source_filter, dow_filter, year_where)
            chunks = retrieve_chunks(rewrite.query, n=N_RESULTS_BROAD, where=combined_where, date_hint=date_hint, **retrieval_kwargs)
        elif rewrite.date_range:
            # Week/month range: pre-filter in ChromaDB via date_numeric, fallback to unfiltered
            date_range = resolve_date_range(rewrite.date_range)
            if date_range:
                range_flt, date_hint = date_range_filter(*date_range)
                combined_where = merge_where_filters(source_filter, dow_filter, range_flt)
                chunks = retrieve_chunks(rewrite.query, n=N_RESULTS_BROAD, where=combined_where, date_hint=date_hint, **retrieval_kwargs)
                if not chunks:
                    fallback_where = merge_where_filters(source_filter, dow_filter)
                    chunks = retrieve_chunks(rewrite.query, n=N_RESULTS_BROAD, where=fallback_where, date_hint=date_hint, **retrieval_kwargs)
            else:
                chunks = retrieve_chunks(rewrite.query, where=merge_where_filters(source_filter, dow_filter), **retrieval_kwargs)
        else:
            # Specific day (or no date): try date filter, fall back to unfiltered
            resolved_date = resolve_date_expression(rewrite.date_expression)
            date_filter, date_hint = date_filter_from_date(resolved_date)
            chunks = retrieve_chunks(
                rewrite.query,
                where=merge_where_filters(source_filter, dow_filter),
                date_filter=date_filter,
                date_hint=date_hint,
                **retrieval_kwargs,
            )

    if not chunks:
        if rewrite.recency and rewrite.day_of_week:
            return {
                "answer": (
                    f"I couldn't find any {rewrite.day_of_week} sermons in the database. "
                    "The collection may not have content for that day yet — try asking without a day filter."
                ),
                "sources": []
            }
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

    if listing_mode:
        context_header = "Context from sermon transcripts (one excerpt per sermon, newest first):"
        listing_hint = (
            "\nThe user is asking for a SUMMARY/LISTING across multiple sermons. "
            "For each excerpt above, give 1–2 sentences capturing the date, title, and main theme, "
            "citing it as [N]. Cover EVERY excerpt — don't skip any. "
            "After the per-sermon summaries, briefly note any cross-cutting themes if present."
        )
        if listing_total > len(chunks):
            listing_hint += (
                f"\nNOTE: The user's date range actually contains {listing_total} sermons total. "
                f"You're seeing the {len(chunks)} most recent. Begin your answer by noting that "
                "more sermons exist in this range and offer to narrow the search."
            )
        listing_hint += "\n"
    else:
        context_header = "Context from sermon transcripts:"
        listing_hint = ""

    messages.append({
        "role": "user",
        "content": f"{context_header}\n\n{context}\n\n---\n{listing_hint}\nUser question: {req.message}"
    })

    # 4. Call OpenAI
    response = await openai_client.chat.completions.create(
        model       = OPENAI_MODEL,
        max_tokens  = 2048 if listing_mode else 1024,
        temperature = 0,
        messages    = messages,
    )

    answer = response.choices[0].message.content

    # 5. Guardrail: verify answer is safe and grounded in the retrieved chunks
    answer = await guardrail_check(req.message, context, answer)

    # 5b. Citation verification: drop [N] citations the cross-encoder can't support
    answer, citations_removed = verify_citations(answer, chunks)
    if citations_removed:
        print(f"⚠️  Stripped {citations_removed} unsupported citation(s)")

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
    try:
        return FileResponse("static/index.html")
    except Exception as e:
        print(f"Error serving index.html: {e}", flush=True)
        return {"error": str(e)}