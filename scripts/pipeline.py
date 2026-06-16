"""
pipeline.py — incremental scrape → process → embed → upload

Run manually:   python scripts/pipeline.py
Run in CI:      same command, with SMF_COOKIE and HF_TOKEN env vars set
"""

import os
import sys
import time
import random
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

# Make scripts importable
sys.path.insert(0, str(Path(__file__).parent))

CHROMA_DB_PATH   = "./chroma_db"
COLLECTION_SERMONS = "sermons"
HF_DATASET_REPO  = "ryantangmj/church-chroma-db"
SERMON_DIR       = "./sermon_documents"
CLEANED_DIR      = "./cleaned_sermons"
BOARD_IDS        = [6, 7]


# ── Step 1: Download existing ChromaDB from HuggingFace ───────────────────────

def download_chroma():
    hf_token = os.environ.get("HF_TOKEN", "")
    print("⏳ Downloading ChromaDB from HuggingFace...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        local_dir=CHROMA_DB_PATH,
        token=hf_token or None,
    )
    print("✅ ChromaDB downloaded.")


# ── Step 2: Get already-embedded source filenames from ChromaDB ───────────────

def get_embedded_filenames() -> set[str]:
    import chromadb
    from chunk_sermons import NomicEmbeddingFunction

    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    try:
        collection = client.get_collection(
            name=COLLECTION_SERMONS,
            embedding_function=NomicEmbeddingFunction()
        )
        results = collection.get(include=["metadatas"])
        filenames = {m["source_file"] for m in results["metadatas"] if m.get("source_file")}
        print(f"✅ Found {len(filenames)} already-embedded source files.")
        return filenames
    except Exception as e:
        print(f"⚠️  Could not read existing collection: {e}")
        return set()


# ── Step 3: Scrape latest sermon from each board ──────────────────────────────

def scrape_new(already_embedded: set[str]) -> list[str]:
    from scrape import scrape_latest_from_board

    downloaded = []
    for board_id in BOARD_IDS:
        filename = scrape_latest_from_board(board_id, already_embedded)
        if filename:
            downloaded.append(filename)
        time.sleep(random.uniform(3.0, 6.0))

    print(f"\n✅ Scraped {len(downloaded)} new file(s): {downloaded}")
    return downloaded


# ── Step 4: Post-process docx → cleaned txt ───────────────────────────────────

def process_docs(filenames: list[str]) -> list[Path]:
    from post_process_doc import process_document

    os.makedirs(CLEANED_DIR, exist_ok=True)
    cleaned = []
    for fname in filenames:
        docx_path = Path(SERMON_DIR) / fname
        if not docx_path.exists():
            print(f"⚠️  File not found: {docx_path}")
            continue
        process_document(docx_path)
        stem = docx_path.stem
        txt_path = Path(CLEANED_DIR) / f"{stem}_pure.txt"
        if txt_path.exists():
            cleaned.append(txt_path)
        # Delete raw docx immediately after processing
        docx_path.unlink()
        print(f"   🗑  Deleted raw file: {fname}")

    return cleaned


# ── Step 5: Chunk and embed cleaned files into ChromaDB ───────────────────────

def embed_docs(txt_files: list[Path]):
    from chunk_sermons import parse_file, build_chunks, ingest_to_chroma, backfill_day_of_week

    all_chunks = []
    for fpath in txt_files:
        print(f"\n📄  Parsing: {fpath.name}")
        sermon = parse_file(fpath)
        chunks = build_chunks(sermon, doc_type="sermon")
        all_chunks.extend(chunks)
        print(f"    {sermon['date']} | {sermon['title']} | {len(chunks)} chunks")

    if all_chunks:
        ingest_to_chroma(all_chunks, COLLECTION_SERMONS, CHROMA_DB_PATH)

    # Delete cleaned txt files
    for fpath in txt_files:
        fpath.unlink()
        print(f"   🗑  Deleted cleaned file: {fpath.name}")

    return len(all_chunks)


# ── Step 6: Upload updated ChromaDB back to HuggingFace ──────────────────────

def upload_chroma():
    from huggingface_hub import HfApi

    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        print("⚠️  HF_TOKEN not set — skipping upload.")
        return

    print("⏳ Uploading ChromaDB to HuggingFace...")
    api = HfApi(token=hf_token)
    api.upload_folder(
        folder_path=CHROMA_DB_PATH,
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        delete_patterns="*",
    )
    print("✅ Upload complete.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    download_chroma()

    # One-time backfill: add day_of_week to chunks that predate the field
    from chunk_sermons import backfill_day_of_week
    backfill_day_of_week(COLLECTION_SERMONS, CHROMA_DB_PATH)

    already_embedded = get_embedded_filenames()
    new_files = scrape_new(already_embedded)

    if not new_files:
        print("\nNo new sermons found. Nothing to do.")
        sys.exit(0)

    cleaned = process_docs(new_files)
    if not cleaned:
        print("\nNo cleaned files produced. Check post-processing errors.")
        sys.exit(1)

    count = embed_docs(cleaned)
    print(f"\n✅ Embedded {count} new chunks.")

    upload_chroma()
    print("\nDone.")
