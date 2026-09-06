"""Turn text into embedding vectors using the OpenAI API.

An *embedding* is a list of numbers that captures the meaning of a piece of
text, so that texts with similar meaning end up close together in vector space.
We embed every document chunk once (at ingest time) and embed each question at
query time, then compare them to find the most relevant chunks.
"""

import os

from openai import OpenAI

# Lazy client so importing this module never requires a key to be set yet.
_client = None
EMBED_MODEL = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY from the environment
    return _client


def embed(texts):
    """Turn a list of strings into a list of vectors (lists of floats)."""
    client = _get_client()
    vectors = []
    batch_size = 100  # the API accepts many inputs per call; batch to be safe
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(item.embedding for item in response.data)
    return vectors


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY to run this demo.")
        raise SystemExit

    samples = [
        "How many vacation days do I get?",
        "What is the paid time off policy?",
        "The cat sat on the mat.",
    ]
    vectors = embed(samples)
    print(f"Each sentence became a vector of {len(vectors[0])} numbers.\n")

    import numpy as np

    def similarity(a, b):
        a, b = np.array(a), np.array(b)
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    print("vacation vs. PTO :", round(similarity(vectors[0], vectors[1]), 3))
    print("vacation vs. cat :", round(similarity(vectors[0], vectors[2]), 3))
