"""The RAG core: Retrieval-Augmented Generation.

Given a question, this module:
  1. Retrieval  - embeds the question and pulls the top-k most similar chunks
                  out of the vector database.
  2. Augmented  - injects those chunks into the prompt as context.
  3. Generation - asks the OpenAI chat model to answer using ONLY that context,
                  which keeps the answer grounded and reduces hallucination.

The model is also told to cite the source filenames and to say when the answer
isn't in the documents.
"""

import os

from openai import OpenAI

from embedder import embed
from vectorstore import get_collection, search

# Model names change over time. If you get a "model not found" error, set a
# current one from https://platform.openai.com/docs/models
CHAT_MODEL = os.environ.get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.environ.get("TOP_K", "4"))

SYSTEM_PROMPT = (
    "You are a precise assistant that answers questions using ONLY the context "
    "passages provided by the user. Follow these rules:\n"
    "- If the answer is not contained in the context, reply that you don't know "
    "based on the provided documents.\n"
    "- Do not use outside knowledge or make anything up.\n"
    "- Keep the answer concise and cite the source filename(s) you relied on."
)

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY from the environment
    return _client


def build_context(hits):
    """Format retrieved chunks into a labelled block for the prompt."""
    blocks = []
    for hit in hits:
        blocks.append(f"[Source: {hit['source']}]\n{hit['text']}")
    return "\n\n---\n\n".join(blocks)


def answer(question, k=TOP_K):
    """Retrieve context and generate a grounded answer.

    Returns a dict: {"answer": str, "sources": [filenames]}.
    """
    query_vector = embed([question])[0]
    hits = search(get_collection(), query_vector, k=k)

    if not hits:
        return {
            "answer": "The index is empty. Run `python ingest.py` first.",
            "sources": [],
        }

    context = build_context(hits)
    user_message = f"Context passages:\n\n{context}\n\nQuestion: {question}"

    response = _get_client().chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,  # deterministic, factual answers
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    text = response.choices[0].message.content.strip()
    sources = sorted({hit["source"] for hit in hits})
    return {"answer": text, "sources": sources}


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY first.")
        raise SystemExit
    import json
    print(json.dumps(answer("How many vacation days do I get?"), indent=2))
