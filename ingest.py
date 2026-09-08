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
from loader import read_file  
import time
from vectorstore import get_collection 
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

def add_text(name, text):
    """Add pasted text to the EXISTING index (without wiping it)."""
    text = (text or "").strip()
    if not text:
        return 0
    name = (name or "pasted-text").strip() or "pasted-text"
    chunks = chunk_documents([{"text": text, "source": name}])
    stamp = str(int(time.time() * 1000))
    for c in chunks:
        c["chunk_id"] = f"{c['chunk_id']}@{stamp}"
    embeddings = embed([c["text"] for c in chunks])
    add_chunks(get_collection(), chunks, embeddings)
    return len(chunks)


def add_file(filename, data):   # <-- upload-only
    """Save an uploaded file into documents/ and add it to the index."""
    safe_name = Path(filename).name
    Path(DOCS_FOLDER).mkdir(exist_ok=True)
    path = Path(DOCS_FOLDER) / safe_name
    path.write_bytes(data)
    text = read_file(path)
    if not text:
        return 0, safe_name
    chunks = chunk_documents([{"text": text, "source": safe_name}])
    embeddings = embed([c["text"] for c in chunks])
    add_chunks(get_collection(), chunks, embeddings)
    return len(chunks), safe_name

if __name__ == "__main__":
    main()
