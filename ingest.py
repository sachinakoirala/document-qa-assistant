"""Build the search index.

Run this once (and again whenever you add or change files in documents/):

    python ingest.py

It reads every file, splits them into chunks, turns each chunk into a vector,
and stores everything in the local Chroma database.
"""

from loader import load_documents
from chunker import chunk_documents
from embedder import embed
from vectorstore import reset_collection, add_chunks

DOCS_FOLDER = "documents"


def main():
    print(f"Reading files from {DOCS_FOLDER}/ ...")
    documents = load_documents(DOCS_FOLDER)
    if not documents:
        print("No .txt, .md or .pdf files found. Add some to documents/ first.")
        return
    print(f"  loaded {len(documents)} document(s)")

    chunks = chunk_documents(documents)
    print(f"  split into {len(chunks)} chunk(s)")

    print("Embedding chunks (first run downloads the model, please wait) ...")
    embeddings = embed([c["text"] for c in chunks])

    print("Storing in the vector database ...")
    collection = reset_collection()
    add_chunks(collection, chunks, embeddings)

    print(f"\nDone. Indexed {len(chunks)} chunks. You can now run:  python ask.py")


if __name__ == "__main__":
    main()
