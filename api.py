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
from fastapi import FastAPI, File, UploadFile      # add File, UploadFile
from ingest import build_index, add_text, add_file  # add add_text, add_file
from vectorstore import reset_collection            # new line
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

class AddTextRequest(BaseModel):
    name: str = "pasted-text"
    text: str
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

@app.post("/add-text")
def add_text_endpoint(request: AddTextRequest):
    count = add_text(request.name, request.text)
    return {"indexed_chunks": count, "source": (request.name or "pasted-text").strip()}


@app.post("/upload")                                # <-- upload-only
async def upload_file(file: UploadFile = File(...)):
    data = await file.read()
    count, source = add_file(file.filename, data)
    return {"indexed_chunks": count, "source": source}


@app.post("/clear")
def clear_index():
    reset_collection()
    return {"status": "cleared"}
# serve any other static assets (css/js) if you add them later
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
