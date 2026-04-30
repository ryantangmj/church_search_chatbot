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

    # Chunk with Qwen3 embeddings via Ollama:
    python chunk_sermons.py --input ./cleaned_sermons --collection sermons_qwen3 --embed-model qwen3-embedding:4b

    # Dry run — prints chunks without writing to ChromaDB:
    python chunk_sermons.py --input ./cleaned_sermons --dry-run

    # Query the collection after ingestion:
    python chunk_sermons.py --query "What did the pastor say about prayer?" --collection sermons
"""

import hashlib
import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict


# ── Chunking config ────────────────────────────────────────────────────────────

TARGET_CHUNK_WORDS   = 512   # aim for ~512 words per chunk
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
    special_blocks: list[dict] = []

    for i, para in enumerate(paragraphs):
        if para["type"] in ("scripture", "video"):
            special_blocks.append({"at": i, "para": para})
        else:
            sermon_paras.append(para["text"])

    # -- Build special chunks (scripture / video) --
    for item in special_blocks:
        para = item["para"]
        chunk_type = para["type"]

        chunks.append(Chunk(
            chunk_id       = hashlib.md5(f"{sermon['source_file']}_{chunk_index}".encode()).hexdigest(),
            date           = sermon["date"],
            title          = sermon["title"],
            chunk_index    = chunk_index,
            total_chunks   = 0,
            chunk_type     = chunk_type,
            scripture_refs = [para["ref"]] if chunk_type == "scripture" else [],
            text           = para["text"],
            word_count     = _word_count(para["text"]),
            source_file    = sermon["source_file"],
        ))
        chunk_index += 1

    # -- Build sliding-window sermon chunks --
    current_words: list[str] = []
    overlap_carry: list[str] = []

    def flush_chunk(words: list[str]):
        nonlocal chunk_index
        text = " ".join(words).strip()
        if not text:
            return
        chunks.append(Chunk(
            chunk_id       = hashlib.md5(f"{sermon['source_file']}_{chunk_index}".encode()).hexdigest(),
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
            overlap_carry = current_words[-OVERLAP_WORDS:] if OVERLAP_WORDS else []
            current_words = overlap_carry.copy()

    # flush remainder
    if current_words:
        flush_chunk(current_words)

    # fill in total_chunks
    total = len(chunks)
    for c in chunks:
        c.total_chunks = total

    chunks.sort(key=lambda c: c.chunk_index)

    return chunks


# ── Embedding function ─────────────────────────────────────────────────────────

def get_embedding_function(model: str = None):
    """Returns embedding function. None = ChromaDB default (all-MiniLM-L6-v2)."""
    if model is None:
        return None

    from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
    print(f"🔗  Using Ollama embedding model: {model}")
    return OllamaEmbeddingFunction(
        url="http://localhost:11434/api/embeddings",
        model_name=model,
    )


# ── ChromaDB ingestion ──────────────────────────────────────────────────────────

def ingest_to_chroma(chunks: list[Chunk], collection_name: str, db_path: str = "./chroma_db", embed_model: str = None):
    import chromadb

    client = chromadb.PersistentClient(path=db_path)
    ef = get_embedding_function(embed_model)

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        **({"embedding_function": ef} if ef else {})
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

    # ── Use smaller batch size for large/slow embedding models ──
    batch_size = 10 if embed_model else 100
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
    model_label = embed_model if embed_model else "default (all-MiniLM-L6-v2)"
    print(f"✅  Ingested {len(chunks)} chunks into '{collection_name}' using {model_label}")
    return collection


# ── Query helper ────────────────────────────────────────────────────────────────

def query_collection(query: str, collection_name: str, db_path: str = "./chroma_db", n_results: int = 5, embed_model: str = None):
    import chromadb

    client = chromadb.PersistentClient(path=db_path)
    ef = get_embedding_function(embed_model)

    collection = client.get_collection(
        collection_name,
        **({"embedding_function": ef} if ef else {})
    )

    results = collection.query(query_texts=[query], n_results=n_results)

    print(f"\n🔍  Query: {query}\n{'─'*60}")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    )):
        score = 1 - dist
        print(f"\n[{i+1}] Score: {score:.3f}  |  {meta['date']} — {meta['title']}")
        print(f"     Type: {meta['chunk_type']}  |  Chunk {meta['chunk_index']+1}/{meta['total_chunks']}")
        if meta.get("scripture_refs"):
            print(f"     Scriptures: {meta['scripture_refs']}")
        print(f"     {doc[:300]}{'...' if len(doc) > 300 else ''}")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chunk sermon transcripts → ChromaDB")
    parser.add_argument("--input",       default=".",           help="Folder containing .txt sermon files")
    parser.add_argument("--collection",  default="sermons",     help="ChromaDB collection name")
    parser.add_argument("--db-path",     default="./chroma_db", help="Path to persist ChromaDB")
    parser.add_argument("--embed-model", default=None,          help="Ollama embedding model e.g. qwen3-embedding:4b")
    parser.add_argument("--dry-run",     action="store_true",   help="Print chunks, skip ChromaDB write")
    parser.add_argument("--query",       default=None,          help="Run a query against the collection")
    parser.add_argument("--n-results",   type=int, default=5,   help="Number of results for --query")
    parser.add_argument("--export-json", default=None,          help="Also save chunks to a JSON file")
    args = parser.parse_args()

    # Query-only mode
    if args.query:
        query_collection(args.query, args.collection, args.db_path, args.n_results, args.embed_model)
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
        ingest_to_chroma(all_chunks, args.collection, args.db_path, args.embed_model)


if __name__ == "__main__":
    main()