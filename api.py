"""FastAPI REST API for the Document Q&A Assistant.

Endpoints
---------
GET  /            -> the chat web interface (static/index.html)
POST /ask         -> {"question": "..."}  ->  {"answer": "...", "sources": [...]}
POST /ingest      -> rebuild the index from the documents/ folder
GET  /health      -> simple liveness check

Run it with:   uvicorn api:app --reload
"""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import rag
from ingest import build_index

load_dotenv()  # load OPENAI_API_KEY etc. before anything calls OpenAI

app = FastAPI(title="Document Q&A Assistant", version="1.0.0")

STATIC_DIR = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Answer a question grounded in the indexed documents."""
    return rag.answer(request.question)


@app.post("/ingest")
def ingest():
    """(Re)build the vector index from the documents/ folder."""
    count = build_index()
    return {"indexed_chunks": count}


@app.get("/")
def home():
    """Serve the chat web interface."""
    return FileResponse(STATIC_DIR / "index.html")


# serve any other static assets (css/js) if you add them later
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
