from pathlib import Path

from pypdf import PdfReader


def _read_pdf(path):
    """Pull the text out of a PDF, one page at a time."""
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)

def read_file(path):
    """Read a single .txt, .md or .pdf file into plain text."""
    from pathlib import Path
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".txt", ".md"):
        text = path.read_text(encoding="utf-8-sig")
    elif suffix == ".pdf":
        text = _read_pdf(path)
    else:
        return ""
    return text.strip()

def load_documents(folder):
    """Read every .txt, .md and .pdf file in a folder into a list of documents.

    Each document is a dict: {"text": ..., "source": filename}.
    """
    documents = []
    for path in sorted(Path(folder).iterdir()):
        suffix = path.suffix.lower()
        if suffix in (".txt", ".md"):
            text = path.read_text(encoding="utf-8-sig")
        elif suffix == ".pdf":
            text = _read_pdf(path)
        else:
            continue  # skip anything that isn't text we can read

        text = text.strip()
        if text:  # ignore empty / unreadable files
            documents.append({"text": text, "source": path.name})
    return documents


if __name__ == "__main__":
    docs = load_documents("documents")
    print(f"Loaded {len(docs)} document(s)")
    for d in docs:
        print(f"  {d['source']} - {len(d['text'])} characters")
