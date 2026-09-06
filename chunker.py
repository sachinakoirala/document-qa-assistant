"""Split long documents into small, overlapping chunks.

Why chunk? An embedding model turns a piece of text into ONE vector. If you
feed it a whole 20-page PDF, all that meaning gets squashed into a single
vector and search becomes vague. Splitting into ~200-word chunks means each
vector represents one focused idea, so retrieval can find the exact passage
that answers a question.

The overlap (a few shared words between neighbouring chunks) stops us from
cutting a sentence in half and losing the answer at a boundary.
"""


def chunk_text(text, chunk_size=200, overlap=40):
    """Split one string into a list of chunks, measured in words."""
    words = text.split()
    if not words:
        return []

    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_size]
        chunks.append(" ".join(chunk))
        if start + chunk_size >= len(words):
            break  # we've reached the end
    return chunks


def chunk_documents(documents, chunk_size=200, overlap=40):
    """Turn a list of documents into a flat list of chunk records.

    Each chunk keeps a reference back to the file it came from, so the
    assistant can cite its sources later.
    """
    chunks = []
    for doc in documents:
        pieces = chunk_text(doc["text"], chunk_size, overlap)
        for i, piece in enumerate(pieces):
            chunks.append({
                "text": piece,
                "source": doc["source"],
                "chunk_id": f"{doc['source']}#{i}",
            })
    return chunks


if __name__ == "__main__":
    sample = {
        "text": " ".join(f"word{i}" for i in range(500)),
        "source": "demo.txt",
    }
    pieces = chunk_documents([sample])
    print(f"500 words became {len(pieces)} chunks")
    for p in pieces:
        print(f"  {p['chunk_id']} - {len(p['text'].split())} words")
