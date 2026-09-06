"""Build the search index from the documents/ folder.

Run once, and again whenever you add or change files:

    python ingest.py

It reads every file, splits them into overlapping chunks, embeds each chunk
with the OpenAI API, and stores everything in the local ChromaDB database.
"""

from dotenv import load_dotenv

from loader import load_documents
from chunker import chunk_documents
from embedder import embed
from vectorstore import reset_collection, add_chunks

load_dotenv()  # read OPENAI_API_KEY etc. from a .env file if present

DOCS_FOLDER = "documents"


def build_index(folder=DOCS_FOLDER):
    """Read, chunk, embed and store every document. Returns the chunk count."""
    documents = load_documents(folder)
    if not documents:
        return 0

    chunks = chunk_documents(documents)
    embeddings = embed([c["text"] for c in chunks])

    collection = reset_collection()
    add_chunks(collection, chunks, embeddings)
    return len(chunks)


def main():
    print(f"Reading files from {DOCS_FOLDER}/ ...")
    documents = load_documents(DOCS_FOLDER)
    if not documents:
        print("No .txt, .md or .pdf files found. Add some to documents/ first.")
        return
    print(f"  loaded {len(documents)} document(s)")
    print("Chunking, embedding and indexing (this calls the OpenAI API) ...")
    count = build_index()
    print(f"\nDone. Indexed {count} chunks. Now run:  uvicorn api:app --reload")


if __name__ == "__main__":
    main()
