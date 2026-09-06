"""A thin wrapper around ChromaDB, a local vector database.

The vector database stores each chunk together with its embedding vector and,
given a question vector, quickly returns the chunks whose vectors are closest
in meaning (this is the "top-k semantic search" step of the pipeline).

We use cosine distance, which is the standard choice for text embeddings, and
persist everything to a .chroma/ folder so the index survives between runs.
"""

import chromadb

DB_PATH = ".chroma"
COLLECTION_NAME = "documents"
# cosine similarity is the recommended distance metric for text embeddings
COLLECTION_META = {"hnsw:space": "cosine"}


def _client():
    return chromadb.PersistentClient(path=DB_PATH)


def reset_collection():
    """Delete any existing index and return a fresh, empty collection."""
    client = _client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # nothing to delete on the first run
    return client.create_collection(COLLECTION_NAME, metadata=COLLECTION_META)


def get_collection():
    """Open the existing index (creating it if it isn't there yet)."""
    return _client().get_or_create_collection(COLLECTION_NAME, metadata=COLLECTION_META)


def add_chunks(collection, chunks, embeddings):
    """Store chunks + their vectors. `chunks` and `embeddings` line up 1:1."""
    collection.add(
        ids=[c["chunk_id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        embeddings=embeddings,
        metadatas=[{"source": c["source"]} for c in chunks],
    )


def search(collection, query_embedding, k=4):
    """Return the top-k chunks most similar to the question vector."""
    result = collection.query(query_embeddings=[query_embedding], n_results=k)
    hits = []
    # Chroma nests results one level per query; we only sent one query.
    for text, meta, distance in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        hits.append({
            "text": text,
            "source": meta["source"],
            "distance": distance,
        })
    return hits


if __name__ == "__main__":
    # Tiny self-test using fake 3-number vectors (no API call needed).
    col = reset_collection()
    demo = [
        {"text": "apples and oranges", "source": "fruit.txt", "chunk_id": "a"},
        {"text": "cars and trucks", "source": "vehicles.txt", "chunk_id": "b"},
    ]
    add_chunks(col, demo, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    hits = search(col, [0.9, 0.1, 0.0], k=1)
    print("closest chunk:", hits[0]["text"], "from", hits[0]["source"])
