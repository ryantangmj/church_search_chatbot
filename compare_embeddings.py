"""
compare_embeddings.py
Run: venv/bin/python compare_embeddings.py
"""

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

DB_PATH = "./chroma_db"

COLLECTIONS = {
    "default (all-MiniLM-L6-v2)": ("sermons_default", None),
    "qwen3-embedding:4b":          ("sermons_qwen3",   "qwen3-embedding:4b"),
}

TEST_QUERIES = [
    "What does the pastor teach about prayer?",
    "How should we treat people who hurt us?",
    "Taking action in faith",
    "Ephesians 4:22",
    "What does it mean to be a new creation?",
]

def get_collection(client, name, model):
    ef = None
    if model:
        ef = OllamaEmbeddingFunction(
            url="http://localhost:11434/api/embeddings",
            model_name=model,
        )
    return client.get_collection(name, **({"embedding_function": ef} if ef else {}))


def run_comparison():
    client = chromadb.PersistentClient(path=DB_PATH)

    for query in TEST_QUERIES:
        print(f"\n{'═'*70}")
        print(f"QUERY: {query}")
        print(f"{'═'*70}")

        for label, (col_name, model) in COLLECTIONS.items():
            print(f"\n  ── {label} ──")
            try:
                col = get_collection(client, col_name, model)
                results = col.query(query_texts=[query], n_results=3)

                for i, (doc, meta, dist) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )):
                    score = 1 - dist
                    print(f"  [{i+1}] {score:.3f} | {meta['date']} — {meta['title'][:50]}")
                    print(f"       {doc[:120]}...")
            except Exception as e:
                print(f"  Error: {e}")

    print(f"\n{'═'*70}")
    print("Comparison complete. Look for:")
    print("  - Higher scores = better semantic match")
    print("  - More relevant excerpts for the query topic")
    print("  - Scripture queries returning scripture-type chunks")

if __name__ == "__main__":
    run_comparison()