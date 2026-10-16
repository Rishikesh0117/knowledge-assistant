import json
import chromadb
from src import config
from src.agent.llm import embed
from src.ingestion.papers import load_papers
from src.ingestion.docs import load_docs

BATCH = 32
COLLECTION = "knowledge_base"
CACHE = config.ROOT / "data" / "chunks_cache.json"


def get_chunks():
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    chunks = load_papers() + load_docs()
    CACHE.write_text(json.dumps(chunks), encoding="utf-8")
    return chunks


def main():
    chunks = get_chunks()
    words = sum(len(c["text"].split()) for c in chunks)
    print(f"{len(chunks)} chunks, ~{int(words * 1.4)} tokens to embed")

    db = chromadb.PersistentClient(path=str(config.VECTORSTORE_DIR))
    try:
        db.delete_collection(COLLECTION)
    except Exception:
        pass
    col = db.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        col.add(
            ids=[f"c{i + j}" for j in range(len(batch))],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
            embeddings=embed([c["text"] for c in batch]),
        )
        print(f"embedded {min(i + BATCH, len(chunks))}/{len(chunks)}")


if __name__ == "__main__":
    main()