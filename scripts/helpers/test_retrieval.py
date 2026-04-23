# debug_retrieval.py
# Purpose: Debug retrieval quality for RAG (FAISS + metadata)
# How to run: python debug_retrieval.py

import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import pprint

# -----------------------------
# 1. Load embedding model
# -----------------------------
print("Loading model...")
model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cpu")

# -----------------------------
# 2. Load FAISS index
# -----------------------------
print("Loading FAISS index...")
index = faiss.read_index("data/embeddings/faiss.index")

# -----------------------------
# 3. Load metadata
# -----------------------------
print("Loading metadata...")
with open("data/embeddings/chunk_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"Loaded {len(metadata)} chunks")

# -----------------------------
# 4. Inspect metadata structure (once)
# -----------------------------
print("\n--- METADATA SAMPLE ---")
pprint.pprint(metadata[0])
print("Keys:", metadata[0].keys())
print("-----------------------\n")

# -----------------------------
# 5. Query loop
# -----------------------------
while True:
    print("\n==============================")
    query = input("Enter query (or 'exit'): ")
    if query.lower() == "exit":
        break

    # 🔥 Important for BGE models
    query = "Represent this sentence for searching relevant passages: " + query

    print("\nEncoding query...")
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # -----------------------------
    # 6. Search FAISS
    # -----------------------------
    k = 10
    print("Searching...")
    distances, indices = index.search(query_embedding, k)

    # -----------------------------
    # 7. RAW RESULTS (PRIMARY DEBUG)
    # -----------------------------
    print("\n==============================")
    print(" RAW TOP-K RESULTS")
    print("==============================\n")

    for i, idx in enumerate(indices[0]):
        chunk = metadata[idx]
        text = chunk.get("text", "")
        meta = chunk.get("metadata", {})

        print(f"\nRank {i+1}")
        print("Score:", float(distances[0][i]))
        print("Company:", meta.get("company_name"))
        print("Section:", meta.get("section"))
        print("Chunk ID:", chunk.get("chunk_id"))
        print("Text Preview:\n", text[:300])
        print("-" * 60)

    # -----------------------------
    # 8. OPTIONAL FILTERED VIEW
    # -----------------------------
    print("\n==============================")
    print(" OPTIONAL FILTERED VIEW")
    print("==============================")

    target_company = input("Filter by company (press Enter to skip): ").strip()
    target_section = input("Filter by section (press Enter to skip): ").strip()

    shown = 0

    for i, idx in enumerate(indices[0]):
        chunk = metadata[idx]
        text = chunk.get("text", "")
        meta = chunk.get("metadata", {})

        if target_company and meta.get("company_name") != target_company:
            continue

        if target_section and meta.get("section") != target_section:
            continue

        print(f"\nFiltered Rank {shown+1}")
        print("Score:", float(distances[0][i]))
        print("Company:", meta.get("company_name"))
        print("Section:", meta.get("section"))
        print("Text Preview:\n", text[:300])
        print("-" * 60)

        shown += 1

    if shown == 0:
        print("No results found for given filters.")

    # -----------------------------
    # 9. Diagnostics
    # -----------------------------
    companies = [metadata[idx].get("metadata", {}).get("company_name") for idx in indices[0]]
    unique_companies = set(companies)

    print("\n==============================")
    print(" DIAGNOSTICS")
    print("==============================")
    print("Unique companies in top-k:", unique_companies)
    print("Count:", len(unique_companies))

    if len(unique_companies) > 3:
        print("⚠️ Retrieval is too scattered across companies (likely weak embeddings or chunking).")
    else:
        print("✅ Retrieval is relatively focused.")

    print("\nDone.\n")