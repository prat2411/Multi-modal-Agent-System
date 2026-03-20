"""Simple Day 5 style ingestion.

Reads files from knowledge_base and writes chunks to rag_store.json.
"""

import json
from pathlib import Path

from config import KNOWLEDGE_BASE_PATH, STORE_PATH, CHUNK_SIZE, CHUNK_OVERLAP


def read_documents() -> list[dict]:
    docs = []
    if not KNOWLEDGE_BASE_PATH.exists():
        return docs

    for path in KNOWLEDGE_BASE_PATH.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                if text.strip():
                    docs.append({"source": str(path), "text": text})
            except OSError:
                continue
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def build_store() -> dict:
    documents = read_documents()
    chunks = []
    idx = 0

    for doc in documents:
        for chunk in chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP):
            chunks.append({"id": idx, "source": doc["source"], "text": chunk})
            idx += 1

    return {"count": len(chunks), "chunks": chunks}


def main() -> None:
    store = build_store()
    STORE_PATH.write_text(json.dumps(store, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Built store: {STORE_PATH}")
    print(f"Chunks: {store['count']}")


if __name__ == "__main__":
    main()
