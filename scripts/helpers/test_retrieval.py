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
# 3. Load metadata (FIXED UTF-8)
# -----------------------------
print("Loading metadata...")
with open("data/embeddings/chunk_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

print(f"Loaded {len(metadata)} chunks")

# 🔥 NEW: Inspect metadata structure (runs once)
print("\n--- METADATA SAMPLE ---")
pprint.pprint(metadata[0])
print("\nKeys:", metadata[0].keys())
print("-----------------------\n")

# -----------------------------
# 4. Query loop (interactive)
# -----------------------------
while True:
    query = input("\nEnter query (or 'exit'): ")
    if query.lower() == "exit":
        break

    # 🔥 Important for BGE models
    query = "Represent this sentence for searching relevant passages: " + query

    print("Encoding query...")
    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    # -----------------------------
    # 5. Search FAISS
    # -----------------------------
    k = 10  # 🔥 increased for better recall
    print("Searching...")
    distances, indices = index.search(query_embedding, k)

    # -----------------------------
    # 6. Display results (with filtering)
    # -----------------------------
    print("\nTop results:\n")

    shown = 0
    for i, idx in enumerate(indices[0]):
        chunk = metadata[idx]
        text = chunk.get("text", "")

        # 🔥 FILTER 1: Only Apple (edit as needed)
        if "Apple" not in text:
            continue

        # 🔥 FILTER 2: Skip useless boilerplate
        if "Item 1A" in text or "The following discussion" in text:
            continue

        print(f"Result {shown+1}")
        print("Score:", distances[0][i])
        print("Text:", text[:500])
        print("-" * 60)

        shown += 1

        if shown == 5:  # show top 5 AFTER filtering
            break

    if shown == 0:
        print("No relevant filtered results found.")