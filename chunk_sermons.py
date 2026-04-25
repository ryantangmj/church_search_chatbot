"""
chunk_sermons.py
----------------
Chunks cleaned sermon transcript .txt files into overlapping semantic windows
and loads them into a local ChromaDB collection.

Usage:
    # Install deps first:
    pip install chromadb

    # Chunk and ingest all .txt files in a folder:
    python chunk_sermons.py --input ./cleaned_sermons --collection sermons

    # Dry run — prints chunks without writing to ChromaDB:
    python chunk_sermons.py --input ./cleaned_sermons --dry-run

    # Query the collection after ingestion:
    python chunk_sermons.py --query "What did the pastor say about prayer?" --collection sermons
"""

import re
import json
import uuid
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional


# ── Chunking config ────────────────────────────────────────────────────────────

TARGET_CHUNK_WORDS   = 200   # aim for ~200 words per chunk
OVERLAP_WORDS        = 50    # carry over last ~50 words into next chunk
MIN_PARAGRAPH_WORDS  = 5     # paragraphs shorter than this are merged upward


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

        # scripture block: <Book X:Y> ...text...
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


def build_chunks(sermon: dict) -> list[Chunk]:
    """
    Strategy:
      - Scripture blocks  → one chunk each (type=scripture)
      - Video blocks      → one chunk each (type=video)
      - Sermon paragraphs → sliding window of ~TARGET_CHUNK_WORDS with
                            ~OVERLAP_WORDS carry-over between adjacent chunks
    """
    paragraphs = merge_short_paragraphs(sermon["paragraphs"])
    chunks: list[Chunk] = []
    chunk_index = 0

    scripture_refs = sermon["scripture_refs"]

    # Separate sermon paragraphs from special blocks
    sermon_paras: list[str] = []
    special_blocks: list[dict] = []   # scripture / video with insertion order

    for i, para in enumerate(paragraphs):
        if para["type"] in ("scripture", "video"):
            # flush any accumulated sermon text first
            special_blocks.append({"at": i, "para": para})
        else:
            sermon_paras.append(para["text"])

    # -- Build special chunks (scripture / video) --
    for item in special_blocks:
        para = item["para"]
        chunk_type = para["type"]
        extra = {}
        if chunk_type == "video":
            extra["video_index"] = para.get("video_index", "")

        chunks.append(Chunk(
            chunk_id       = str(uuid.uuid4()),
            date           = sermon["date"],
            title          = sermon["title"],
            chunk_index    = chunk_index,
            total_chunks   = 0,        # filled below
            chunk_type     = chunk_type,
            scripture_refs = [para["ref"]] if chunk_type == "scripture" else [],
            text           = para["text"],
            word_count     = _word_count(para["text"]),
            source_file    = sermon["source_file"],
        ))
        chunk_index += 1

    # -- Build sliding-window sermon chunks --
    current_words: list[str] = []
    overlap_carry: list[str] = []   # words to prepend to next chunk

    def flush_chunk(words: list[str]):
        nonlocal chunk_index
        text = " ".join(words).strip()
        if not text:
            return
        chunks.append(Chunk(
            chunk_id       = str(uuid.uuid4()),
            date           = sermon["date"],
            title          = sermon["title"],
            chunk_index    = chunk_index,
            total_chunks   = 0,
            chunk_type     = "sermon",
            scripture_refs = scripture_refs,
            text           = text,
            word_count     = _word_count(text),
            source_file    = sermon["source_file"],
        ))
        chunk_index += 1

    for para_text in sermon_paras:
        para_words = para_text.split()
        current_words.extend(para_words)

        if len(current_words) >= TARGET_CHUNK_WORDS:
            flush_chunk(current_words)
            # carry over overlap
            overlap_carry = current_words[-OVERLAP_WORDS:] if OVERLAP_WORDS else []
            current_words = overlap_carry.copy()

    # flush remainder
    if current_words:
        flush_chunk(current_words)

    # fill in total_chunks
    total = len(chunks)
    for c in chunks:
        c.total_chunks = total

    # sort by chunk_index for clean ordering
    chunks.sort(key=lambda c: c.chunk_index)

    return chunks


# ── ChromaDB ingestion ──────────────────────────────────────────────────────────

def ingest_to_chroma(chunks: list[Chunk], collection_name: str, db_path: str = "./chroma_db"):
    import chromadb

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},   # cosine similarity for semantic search
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
        })

    # Upsert in batches of 100
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids       = ids[i:i+batch_size],
            documents = documents[i:i+batch_size],
            metadatas = metadatas[i:i+batch_size],
        )

    print(f"✅  Ingested {len(chunks)} chunks into collection '{collection_name}' at {db_path}")
    return collection


# ── Query helper ────────────────────────────────────────────────────────────────

def query_collection(query: str, collection_name: str, db_path: str = "./chroma_db", n_results: int = 5):
    import chromadb

    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection_name)

    results = collection.query(query_texts=[query], n_results=n_results)

    print(f"\n🔍  Query: {query}\n{'─'*60}")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        score = 1 - dist   # cosine distance → similarity
        print(f"\n[{i+1}] Score: {score:.3f}  |  {meta['date']} — {meta['title']}")
        print(f"     Type: {meta['chunk_type']}  |  Chunk {meta['chunk_index']+1}/{meta['total_chunks']}")
        if meta.get("scripture_refs"):
            print(f"     Scriptures: {meta['scripture_refs']}")
        print(f"     {doc[:300]}{'...' if len(doc) > 300 else ''}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chunk sermon transcripts → ChromaDB")
    parser.add_argument("--input",      default=".",          help="Folder containing .txt sermon files")
    parser.add_argument("--collection", default="sermons",    help="ChromaDB collection name")
    parser.add_argument("--db-path",    default="./chroma_db",help="Path to persist ChromaDB")
    parser.add_argument("--dry-run",    action="store_true",  help="Print chunks, skip ChromaDB write")
    parser.add_argument("--query",      default=None,         help="Run a query against the collection")
    parser.add_argument("--n-results",  type=int, default=5,  help="Number of results for --query")
    parser.add_argument("--export-json",default=None,         help="Also save chunks to a JSON file")
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
        chunks   = build_chunks(sermon)
        all_chunks.extend(chunks)

        # Summary
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

    # Optional JSON export
    if args.export_json:
        out = Path(args.export_json)
        out.write_text(
            json.dumps([asdict(c) for c in all_chunks], indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        print(f"💾  Chunks saved to {out}")

    # ChromaDB ingestion
    if not args.dry_run:
        ingest_to_chroma(all_chunks, args.collection, args.db_path)


if __name__ == "__main__":
    main()