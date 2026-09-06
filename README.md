# 📄 Document Q&A Assistant (RAG)

Ask questions about your own documents — contracts, research papers, notes — and
get answers grounded **only in those files**, with the source cited. Built with
[Chroma](https://www.trychroma.com/) for vector search, a local
[sentence-transformers](https://www.sbert.net/) embedding model, and
[Claude](https://www.anthropic.com/) for generating the answers.

This is a classic **RAG** (Retrieval-Augmented Generation) pipeline: instead of
hoping a language model already knows your data, you retrieve the relevant
passages first and hand them to the model as context.

---

## How it works

```
                    ┌─────────────┐
 documents/  ─────► │  loader.py  │  read .txt / .md / .pdf
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │ chunker.py  │  split into small overlapping pieces
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │ embedder.py │  each chunk → a vector (list of numbers)
                    └──────┬──────┘
                           ▼
                    ┌───────────────┐
                    │ vectorstore.py│  store vectors in Chroma  ◄── ingest.py
                    └──────┬────────┘
                           ▼
 your question ──► embed ──► find nearest chunks ──► Claude answers ── ask.py
```

1. **Retrieval** — your question is embedded and compared against every chunk;
   the closest few are pulled out.
2. **Augmented** — those chunks are pasted into the prompt as context.
3. **Generation** — Claude answers using that context and cites the source file.

---

## Setup

```bash
# 1. Clone and enter the project
git clone https://github.com/YOUR_USERNAME/rag-qa-assistant.git
cd rag-qa-assistant

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."   # get one at console.anthropic.com
```

## Usage

```bash
# 1. Put your files in the documents/ folder (.txt, .md, .pdf).
#    A sample handbook is already included so you can try it right away.

# 2. Build the search index (run again whenever documents change):
python ingest.py

# 3. Ask questions:
python ask.py
```

Example session:

```
You: How many vacation days do I get?
Claude: Full-time employees receive 20 paid vacation days per year, accrued
monthly, with up to 5 unused days carried into the next year.
Sources: employee_handbook.md
```

Each file also runs on its own for learning/debugging, e.g. `python loader.py`,
`python chunker.py`, or `python embedder.py`.

---

## Project structure

| File | Job |
|------|-----|
| `loader.py` | Reads `.txt`, `.md` and `.pdf` files into plain text |
| `chunker.py` | Splits long text into small overlapping chunks |
| `embedder.py` | Turns text into embedding vectors (`all-MiniLM-L6-v2`) |
| `vectorstore.py` | Stores and searches chunks with Chroma |
| `ingest.py` | One command to build the index from `documents/` |
| `ask.py` | Interactive question-answering loop |

---

## Customizing

- **Chunk size / overlap** — tweak the defaults in `chunker.py`. Smaller chunks =
  more precise retrieval; larger chunks = more context per hit.
- **How many chunks to retrieve** — change `TOP_K` in `ask.py`.
- **Model** — set `ANTHROPIC_MODEL` (see the current list of
  [models](https://docs.claude.com/en/docs/about-claude/models)).
- **Run the answer step fully local & free** — the retrieval half already runs
  locally; you can swap the Claude call in `ask.py` for a local model served by
  [Ollama](https://ollama.com/) if you'd rather not use an API.

## Notes & limits

- This is a learning project, kept deliberately small and readable.
- The embedding model (~90 MB) downloads automatically on first run.
- Answers are only as good as your documents and the retrieved chunks; if the
  answer isn't in your files, the assistant will say so.

## License

MIT — do whatever you like with it.
