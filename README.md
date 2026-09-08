# Document Q&A Assistant

A question-answering app that reads my own documents and answers questions about
them in plain English, always pointing back to the file the answer came from.
Instead of relying on what a language model already knows, it looks things up in
the documents first and only answers from what it finds. This is a pattern
called RAG (Retrieval-Augmented Generation).

I built this to get hands-on with how retrieval and language models work together,
and to have a small but complete end-to-end project: from reading raw files all
the way to a working web app.

## What it can do

- Answer natural-language questions about a set of documents (.txt, .md, .pdf)
- Show which source file each answer came from
- Say "I don't know" when the answer isn't in the documents, instead of guessing
- Add documents three ways: drop files in a folder, paste text in the browser,
  or upload a file from the browser
- Works both as a REST API and through a simple chat web page

## How it works

The pipeline has a few steps:

1. **Load** – read the text out of each file (`loader.py`).
2. **Chunk** – split long text into smaller overlapping pieces so each piece is
   about one idea (`chunker.py`). The overlap avoids cutting a sentence in half.
3. **Embed** – turn each chunk into a vector (a list of numbers that represents
   its meaning) using the OpenAI embeddings API (`embedder.py`).
4. **Store** – save those vectors in ChromaDB, a local vector database
   (`vectorstore.py`).
5. **Ask** – when I ask a question, it gets embedded too, the database returns
   the most similar chunks (top-k search), those chunks are put into the prompt
   as context, and the model answers using only that context (`rag.py`).

Putting the retrieved text into the prompt and telling the model to stick to it
is what keeps the answers grounded and cuts down on made-up information.

```
files -> chunk -> embed -> ChromaDB
                                 |
question -> embed -> search top matches -> add to prompt -> OpenAI -> answer + source
```

## Tech used

- **Python**
- **FastAPI** – the REST API and web server
- **ChromaDB** – local vector database for semantic search
- **OpenAI API** – embeddings and the chat model that writes the answer

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenAI API key
# copy .env.example to .env and put your key inside
```

The `.env` file should contain:

```
OPENAI_API_KEY=your-key-here
```

## Running it

```bash
# Build the index from the documents/ folder
python ingest.py

# Start the app
uvicorn api:app --reload
```

Then open http://127.0.0.1:8000 in a browser and ask questions. There's also an
"Add a document" panel to paste text or upload a file directly in the browser.

## API endpoints

| Method | Route        | What it does                          |
|--------|--------------|---------------------------------------|
| POST   | `/ask`       | Ask a question, get an answer + sources |
| POST   | `/upload`    | Upload a PDF/TXT/MD file and index it |
| POST   | `/add-text`  | Add pasted text to the index          |
| POST   | `/ingest`    | Rebuild the index from the folder     |
| POST   | `/clear`     | Remove all documents from the index   |

## Author

Sachina Koirala
