"""
chunk_sermons.py
----------------
Chunks cleaned sermon transcript .txt files into overlapping semantic windows
and loads them into a local ChromaDB collection.

Usage:
    # Install deps first:
    pip install chromadb sentence-transformers yake-keyword

    # Chunk and ingest all .txt files in a folder:
    python chunk_sermons.py --input ./cleaned_sermons --collection sermons

    # Dry run — prints chunks without writing to ChromaDB:
    python chunk_sermons.py --input ./cleaned_sermons --dry-run

    # Query the collection after ingestion:
    python chunk_sermons.py --query "What did the pastor say about prayer?" --collection sermons
"""

import hashlib
import re
import json
import os
import chromadb
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
from typing import List
from dotenv import load_dotenv

load_dotenv(override=True)

try:
    import yake
except ImportError:
    print("⚠️  YAKE not installed. Topic extraction will be disabled.")
    print("   Install with: pip install yake-keyword")
    yake = None


# ── Chunking config ────────────────────────────────────────────────────────────

TARGET_CHUNK_WORDS   = 400   # aim for ~400 words per chunk (reduced for better precision)
OVERLAP_WORDS        = 75    # carry over last ~75 words into next chunk (increased for better context)
MIN_PARAGRAPH_WORDS  = 5     # paragraphs shorter than this are merged upward
SCRIPTURE_CONTEXT_WORDS = 100  # words of context to include around scripture references


# ── Data model ─────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id:        str
    date:            str
    title:           str
    chunk_index:     int
    total_chunks:    int          # filled in after all chunks for a sermon are built
    chunk_type:      str          # "sermon" | "scripture" | "video"
    scripture_refs:  list[str]
    text:            str
    word_count:      int
    source_file:     str
    topics:          list[str]    # extracted keywords/topics
    section_type:    str          # "intro" | "body" | "conclusion" | "unknown"


# ── Topic Extraction ───────────────────────────────────────────────────────────

def extract_topics(text: str, max_keywords: int = 5) -> List[str]:
    """Extract key topics/keywords from text using YAKE algorithm."""
    if not text or len(text.split()) < 20 or yake is None:
        return []

    try:
        kw_extractor = yake.KeywordExtractor(
            lan="en",
            n=2,  # extract up to 2-grams
            dedupLim=0.7,
            top=max_keywords,
            features=None
        )
        keywords = kw_extractor.extract_keywords(text)
        # Return only the keyword strings, not the scores
        return [kw[0] for kw in keywords]
    except Exception as e:
        print(f"⚠️  Topic extraction failed: {e}")
        return []


def detect_section_type(chunk_index: int, total_chunks: int) -> str:
    """Heuristically determine if chunk is intro, body, or conclusion."""
    if total_chunks <= 2:
        return "body"

    ratio = chunk_index / total_chunks
    if ratio < 0.15:
        return "intro"
    elif ratio > 0.85:
        return "conclusion"
    else:
        return "body"


# ── Parsing ─────────────────────────────────────────────────────────────────────

SCRIPTURE_RE = re.compile(r'^\[([^\]]+\d+:\d+[^\]]*)\](.*)', re.DOTALL)
VIDEO_START_RE = re.compile(r'^<[Vv]ideo?\s*\d+\s*start>', re.IGNORECASE)
VIDEO_END_RE   = re.compile(r'^<[Vv]ideo?\s*\d+\s*end>',   re.IGNORECASE)


def parse_file(path: Path) -> dict:
    """Return {date, title, scripture_refs, paragraphs} from a sermon file."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    date, title = "", ""
    scripture_refs: list[str] = []
    paragraphs: list[dict] = []   # {"type": str, "text": str}

    # -- header --
    for i, line in enumerate(lines):
        if line.startswith("DATE:"):
            date = line.split(":", 1)[1].strip()
        elif line.startswith("TITLE:"):
            title = line.split(":", 1)[1].strip()
        else:
            body_start = i
            break

    # -- body: split on blank lines into raw paragraphs --
    raw_blocks: list[str] = []
    current: list[str] = []
    for line in lines[body_start:]:
        if line.strip() == "":
            if current:
                raw_blocks.append("\n".join(current).strip())
                current = []
        else:
            current.append(line)
    if current:
        raw_blocks.append("\n".join(current).strip())

    # ── Merge orphaned scripture tags with the following paragraph ──
    merged_blocks: list[str] = []
    i = 0
    while i < len(raw_blocks):
        block = raw_blocks[i]
        tag_only = re.match(r'^\[([^\]]+\d+:\d+[^\]]*)\]\s*$', block)
        if tag_only and i + 1 < len(raw_blocks):
            merged_blocks.append(block + " " + raw_blocks[i + 1])
            i += 2
        else:
            merged_blocks.append(block)
            i += 1
    raw_blocks = merged_blocks

    # -- classify each block --
    in_video = False
    video_lines: list[str] = []
    video_idx = 0

    for block in raw_blocks:
        if not block:
            continue

        # video start marker
        if VIDEO_START_RE.match(block):
            in_video = True
            video_lines = []
            continue

        # video end marker
        if VIDEO_END_RE.match(block):
            in_video = False
            video_idx += 1
            if video_lines:
                paragraphs.append({
                    "type": "video",
                    "video_index": video_idx,
                    "text": " ".join(video_lines),
                })
            video_lines = []
            continue

        if in_video:
            video_lines.append(block)
            continue

        # scripture block: [Book X:Y] ...text...
        m = SCRIPTURE_RE.match(block)
        if m:
            ref, body_text = m.group(1).strip(), m.group(2).strip()
            scripture_refs.append(ref)
            paragraphs.append({
                "type": "scripture",
                "ref": ref,
                "text": f"[{ref}] {body_text}",
            })
            continue

        # normal sermon paragraph
        paragraphs.append({"type": "sermon", "text": block})

    return {
        "date":           date,
        "title":          title,
        "scripture_refs": scripture_refs,
        "paragraphs":     paragraphs,
        "source_file":    path.name,
    }


# ── Chunking ────────────────────────────────────────────────────────────────────

def _word_count(text: str) -> int:
    return len(text.split())


def merge_short_paragraphs(paragraphs: list[dict]) -> list[dict]:
    """Merge very short sermon paragraphs upward into their predecessor."""
    merged: list[dict] = []
    for para in paragraphs:
        if (
            para["type"] == "sermon"
            and _word_count(para["text"]) < MIN_PARAGRAPH_WORDS
            and merged
            and merged[-1]["type"] == "sermon"
        ):
            merged[-1]["text"] += " " + para["text"]
        else:
            merged.append(dict(para))
    return merged


def build_chunks(sermon: dict, doc_type: str = "sermon") -> list[Chunk]:
    """
    Enhanced chunking strategy:
      - Scripture blocks  → include surrounding context for better retrieval
      - Video blocks      → one chunk each (type=video)
      - Sermon paragraphs → semantic-aware sliding window that breaks at paragraph boundaries
      - All chunks get topic extraction and section type detection
    """
    paragraphs = merge_short_paragraphs(sermon["paragraphs"])
    chunks: list[Chunk] = []
    chunk_index = 0

    scripture_refs = sermon["scripture_refs"]

    # Separate sermon paragraphs from special blocks
    sermon_paras: list[str] = []
    special_blocks: list[dict] = []

    for i, para in enumerate(paragraphs):
        if para["type"] in ("scripture", "video"):
            special_blocks.append({"at": i, "para": para})
        else:
            sermon_paras.append(para["text"])

    # -- Build contextual scripture chunks --
    for item in special_blocks:
        para = item["para"]
        chunk_type = para["type"]
        para_idx = item["at"]

        if chunk_type == "scripture":
            # Walk backwards/forwards through paragraphs by position to collect
            # surrounding sermon text — substring search was used before but always
            # failed because scripture text has a "[ref]" prefix that never appears
            # in sermon-only paragraphs.
            half = SCRIPTURE_CONTEXT_WORDS // 2
            before_words: list[str] = []
            after_words:  list[str] = []

            for j in range(para_idx - 1, -1, -1):
                if paragraphs[j]["type"] == "sermon":
                    before_words = paragraphs[j]["text"].split() + before_words
                    if len(before_words) >= half:
                        before_words = before_words[-half:]
                        break

            for j in range(para_idx + 1, len(paragraphs)):
                if paragraphs[j]["type"] == "sermon":
                    after_words.extend(paragraphs[j]["text"].split())
                    if len(after_words) >= half:
                        after_words = after_words[:half]
                        break

            contextual_text = " ".join(before_words + [para["text"]] + after_words)

            topics = extract_topics(contextual_text, max_keywords=3)

            chunks.append(Chunk(
                chunk_id       = hashlib.md5(f"{sermon['source_file']}_{chunk_index}".encode()).hexdigest(),
                date           = sermon["date"],
                title          = sermon["title"],
                chunk_index    = chunk_index,
                total_chunks   = 0,
                chunk_type     = doc_type if doc_type != "sermon" else chunk_type,
                scripture_refs = [para["ref"]],
                text           = contextual_text,
                word_count     = _word_count(contextual_text),
                source_file    = sermon["source_file"],
                topics         = topics,
                section_type   = "body",
            ))
            chunk_index += 1

        elif chunk_type == "video":
            topics = extract_topics(para["text"], max_keywords=3)
            chunks.append(Chunk(
                chunk_id       = hashlib.md5(f"{sermon['source_file']}_{chunk_index}".encode()).hexdigest(),
                date           = sermon["date"],
                title          = sermon["title"],
                chunk_index    = chunk_index,
                total_chunks   = 0,
                chunk_type     = chunk_type,
                scripture_refs = [],
                text           = para["text"],
                word_count     = _word_count(para["text"]),
                source_file    = sermon["source_file"],
                topics         = topics,
                section_type   = "body",
            ))
            chunk_index += 1

    # -- Build semantic-aware sermon chunks --
    # Instead of hard word count splits, try to break at paragraph boundaries
    current_paras: list[str] = []
    current_word_count = 0

    def flush_semantic_chunk(paras: list[str], overlap_text: str = ""):
        nonlocal chunk_index
        if overlap_text:
            text = overlap_text + " " + " ".join(paras)
        else:
            text = " ".join(paras)

        text = text.strip()
        if not text or len(text.split()) < 20:  # Skip very small chunks
            return

        topics = extract_topics(text, max_keywords=5)
        wc = _word_count(text)

        chunks.append(Chunk(
            chunk_id       = hashlib.md5(f"{sermon['source_file']}_{chunk_index}".encode()).hexdigest(),
            date           = sermon["date"],
            title          = sermon["title"],
            chunk_index    = chunk_index,
            total_chunks   = 0,
            chunk_type     = doc_type,
            scripture_refs = scripture_refs,
            text           = text,
            word_count     = wc,
            source_file    = sermon["source_file"],
            topics         = topics,
            section_type   = "body",  # Will be updated later
        ))
        chunk_index += 1

    overlap_text = ""

    for para_text in sermon_paras:
        para_wc = _word_count(para_text)

        # If adding this paragraph exceeds target, flush current chunk
        if current_word_count + para_wc >= TARGET_CHUNK_WORDS and current_paras:
            flush_semantic_chunk(current_paras, overlap_text)

            # Create overlap from last paragraph(s)
            last_para_words = current_paras[-1].split()
            overlap_text = " ".join(last_para_words[-OVERLAP_WORDS:]) if len(last_para_words) > OVERLAP_WORDS else current_paras[-1]

            current_paras = []
            current_word_count = 0

        current_paras.append(para_text)
        current_word_count += para_wc

    # Flush remaining paragraphs
    if current_paras:
        flush_semantic_chunk(current_paras, overlap_text)

    # Fill in total_chunks and section types
    total = len(chunks)
    for c in chunks:
        c.total_chunks = total
        if c.section_type == "body":  # Update section type for sermon chunks
            c.section_type = detect_section_type(c.chunk_index, total)

    chunks.sort(key=lambda c: c.chunk_index)

    return chunks


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

class NomicEmbeddingFunction:
    def name(self) -> str:
        return "nomic-embed-text-v1.5"

    def __call__(self, input: list[str]) -> list[list[float]]:
        return _get_embed_model().encode(
            ["search_document: " + t for t in input],
            normalize_embeddings=True,
            batch_size=8,
        ).tolist()


# ── ChromaDB ingestion ──────────────────────────────────────────────────────────

def ingest_to_chroma(chunks: list[Chunk], collection_name: str, db_path: str = "./chroma_db"):
    client = chromadb.PersistentClient(path=db_path)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=NomicEmbeddingFunction()
    )

    ids, documents, metadatas = [], [], []
    for chunk in chunks:
        ids.append(chunk.chunk_id)
        documents.append(chunk.text)
        metadatas.append({
            "date":           chunk.date,
            "title":          chunk.title,
            "chunk_index":    chunk.chunk_index,
            "total_chunks":   chunk.total_chunks,
            "chunk_type":     chunk.chunk_type,
            "scripture_refs": ", ".join(chunk.scripture_refs),
            "word_count":     chunk.word_count,
            "source_file":    chunk.source_file,
            "topics":         ", ".join(chunk.topics),
            "section_type":   chunk.section_type,
        })

    batch_size = 16
    total = len(ids)

    for i in range(0, total, batch_size):
        batch_end = min(i + batch_size, total)
        print(f"  Embedding batch {i+1}–{batch_end} of {total}...", end="\r")
        collection.upsert(
            ids       = ids[i:batch_end],
            documents = documents[i:batch_end],
            metadatas = metadatas[i:batch_end],
        )

    print()  # newline after progress
    print(f"✅  Ingested {len(chunks)} chunks into '{collection_name}' using nomic-embed-text-v1.5")
    return collection


# ── Query helper ────────────────────────────────────────────────────────────────

def query_collection(query: str, collection_name: str, db_path: str = "./chroma_db", n_results: int = 5):
    client = chromadb.PersistentClient(path=db_path)

    collection = client.get_collection(
        collection_name,
        embedding_function=NomicEmbeddingFunction()
    )

    results = collection.query(query_texts=["search_query: " + query], n_results=n_results)

    print(f"\n🔍  Query: {query}\n{'─'*60}")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        score = 1 - dist
        print(f"\n[{i+1}] Score: {score:.3f}  |  {meta['date']} — {meta['title']}")
        print(f"     Type: {meta['chunk_type']}  |  Section: {meta.get('section_type', 'unknown')}  |  Chunk {meta['chunk_index']+1}/{meta['total_chunks']}")
        if meta.get("scripture_refs"):
            print(f"     Scriptures: {meta['scripture_refs']}")
        if meta.get("topics"):
            print(f"     Topics: {meta['topics']}")
        print(f"     {doc[:300]}{'...' if len(doc) > 300 else ''}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chunk sermon transcripts → ChromaDB")
    parser.add_argument("--input",       default=".",           help="Folder containing .txt sermon files")
    parser.add_argument("--collection",  default="sermons",     help="ChromaDB collection name")
    parser.add_argument("--db-path",     default="./chroma_db", help="Path to persist ChromaDB")
    parser.add_argument("--doc-type",    default="sermon",      help="Chunk type label for body text: 'sermon' or 'lesson' (default: sermon)")
    parser.add_argument("--dry-run",     action="store_true",   help="Print chunks, skip ChromaDB write")
    parser.add_argument("--query",       default=None,          help="Run a query against the collection")
    parser.add_argument("--n-results",   type=int, default=5,   help="Number of results for --query")
    parser.add_argument("--export-json", default=None,          help="Also save chunks to a JSON file")
    args = parser.parse_args()

    # Query-only mode
    if args.query:
        query_collection(args.query, args.collection, args.db_path, args.n_results)

        return

    # Ingest mode
    input_path = Path(args.input)
    txt_files  = sorted(input_path.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {input_path}")
        return

    all_chunks: list[Chunk] = []

    for fpath in txt_files:
        print(f"\n📄  Parsing: {fpath.name}")
        sermon   = parse_file(fpath)
        chunks   = build_chunks(sermon, doc_type=args.doc_type)
        all_chunks.extend(chunks)

        type_counts = {}
        for c in chunks:
            type_counts[c.chunk_type] = type_counts.get(c.chunk_type, 0) + 1
        avg_words = sum(c.word_count for c in chunks) / len(chunks)
        print(f"    Date:    {sermon['date']}")
        print(f"    Title:   {sermon['title']}")
        print(f"    Chunks:  {len(chunks)} total — {type_counts}")
        print(f"    Avg words/chunk: {avg_words:.0f}")

        if args.dry_run:
            print(f"\n    {'─'*56}")
            for c in chunks:
                print(f"    [{c.chunk_type.upper()} {c.chunk_index+1}/{c.total_chunks}] ({c.word_count}w)")
                print(f"    {c.text[:200]}{'...' if len(c.text) > 200 else ''}")
                print()

    print(f"\n📦  Total chunks across all files: {len(all_chunks)}")

    if args.export_json:
        out = Path(args.export_json)
        out.write_text(
            json.dumps([asdict(c) for c in all_chunks], indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"💾  Chunks saved to {out}")

    if not args.dry_run:
        ingest_to_chroma(all_chunks, args.collection, args.db_path)


if __name__ == "__main__":
    main()