"""Ask questions about your documents.

    python ask.py

This is the "RAG" part: Retrieval-Augmented Generation.
  1. Retrieval  - find the chunks most relevant to your question.
  2. Augmented  - paste those chunks into the prompt as context.
  3. Generation - let Claude write an answer grounded in that context.

Because the model only sees your retrieved chunks, it answers from YOUR data
instead of from its general training, and it can tell you which file each fact
came from.
"""

import os

from anthropic import Anthropic

from embedder import embed
from vectorstore import get_collection, search

# The model name can change over time. If you get a "model not found" error,
# set a current one from https://docs.claude.com/en/docs/about-claude/models
# either here or via the ANTHROPIC_MODEL environment variable.
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")
TOP_K = 4  # how many chunks to retrieve per question

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using ONLY the context "
    "provided by the user. If the answer is not in the context, say you don't "
    "know based on the documents. Keep answers concise and cite the source "
    "filename(s) you used."
)


def build_context(hits):
    """Format retrieved chunks into a readable block for the prompt."""
    blocks = []
    for hit in hits:
        blocks.append(f"[Source: {hit['source']}]\n{hit['text']}")
    return "\n\n---\n\n".join(blocks)


def answer(client, collection, question):
    """Retrieve context and ask Claude to answer from it."""
    query_vector = embed([question])[0]
    hits = search(collection, query_vector, k=TOP_K)

    if not hits:
        return "The index is empty. Run `python ingest.py` first.", []

    context = build_context(hits)
    user_message = (
        f"Context from my documents:\n\n{context}\n\n"
        f"Question: {question}"
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    text = response.content[0].text
    sources = sorted({hit["source"] for hit in hits})
    return text, sources


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set your ANTHROPIC_API_KEY first, e.g.:")
        print('  export ANTHROPIC_API_KEY="sk-ant-..."')
        return

    client = Anthropic()
    collection = get_collection()

    print("Ask a question about your documents (empty line or Ctrl-C to quit).\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break

        text, sources = answer(client, collection, question)
        print(f"\nClaude: {text}")
        if sources:
            print(f"Sources: {', '.join(sources)}")
        print()


if __name__ == "__main__":
    main()
