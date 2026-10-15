import chromadb
from src import config
from src.agent.llm import embed

_col = None


def _collection():
    global _col
    if _col is None:
        db = chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))
        _col = db.get_collection("kb")
    return _col


def retrieve(query, k=config.TOP_K, source_type=None):
    where = {"source_type": source_type} if source_type else None
    res = _collection().query(
        query_embeddings=embed([query]), n_results=k, where=where
    )
    return [
        {"text": d, "metadata": m, "distance": dist}
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


if __name__ == "__main__":
    for r in retrieve("How does LoRA reduce trainable parameters?"):
        print(r["metadata"]["title"], "|", r["metadata"]["section"], round(r["distance"], 3))