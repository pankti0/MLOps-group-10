import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.models.rag_agent import RAGAgent
from src.embeddings.embedder import Embedder
from src.embeddings.faiss_store import FAISSStore

INDEX_PATH = os.path.join(_REPO_ROOT, "data", "embeddings", "faiss.index")
METADATA_PATH = os.path.join(_REPO_ROOT, "data", "embeddings", "chunk_metadata.json")

embedder = Embedder()
faiss_store = FAISSStore(dimension=embedder.get_dimension())
faiss_store.load(INDEX_PATH, METADATA_PATH)

agent = RAGAgent(None, None, faiss_store, embedder)

test_cases = [
    ("AAPL", "Apple"),
    ("T", "AT&T"),
    ("DAL", "Delta Air Lines"),
]

for ticker, name in test_cases:
    print(f"\n===== {name} ({ticker}) =====")

    query = f"{name} financial risks liquidity debt going concern risk factors"
    chunks = agent._retrieve_chunks_for_company(ticker, query)

    correct = 0

    for i, c in enumerate(chunks[:5]):
        meta = c.get("metadata", {})

        if meta.get("ticker") == ticker:
            correct += 1

        print(f"\nRank {i + 1}")
        print("Ticker:", meta.get("ticker"))
        print("Section:", meta.get("section"))
        print("Score:", round(c.get("score", 0), 3))
        print("Text:", c.get("text", "")[:200])

    precision = correct / len(chunks) if chunks else 0.0
    print(f"\nCompany precision: {precision:.2f}")