from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed(texts):
    """Turn a list of strings into a list of vectors (lists of numbers)."""
    vectors = _model.encode(texts)
    return vectors.tolist()

if __name__ == "__main__":
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
