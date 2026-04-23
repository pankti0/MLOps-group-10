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
# 4. (Optional) Inspect metadata once
# -----------------------------
print("\n--- METADATA SAMPLE ---")
pprint.pprint(metadata[0])
print("Keys:", metadata[0].keys())
print("-----------------------\n")

# -----------------------------
# 5. Query loop
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
    # 6. Search FAISS
    # -----------------------------
    k = 10  # slightly higher for better recall
    print("Searching...")
    distances, indices = index.search(query_embedding, k)

    # -----------------------------
    # 7. Display filtered results
    # -----------------------------
    print("\nTop results (Filtered to Apple + Item 1A):\n")

    shown = 0

    for i, idx in enumerate(indices[0]):
        chunk = metadata[idx]
        text = chunk.get("text", "")
        meta = chunk.get("metadata", {})

        # ✅ Proper filtering using metadata
        if meta.get("company_name") != "Apple":
            continue

        if meta.get("section") != "item_1a":
            continue

        print(f"Result {shown+1}")
        print("Score:", distances[0][i])
        print("Chunk ID:", chunk.get("chunk_id"))
        print("Text:", text[:500])  # truncate for readability
        print("-" * 60)

        shown += 1

        if shown == 5:
            break

    if shown == 0:
        print("No relevant results found after filtering.")