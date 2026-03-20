"""Simple Day 5 style Q&A with Groq and a local JSON store."""

import json
import math
import re
from difflib import get_close_matches

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL, STORE_PATH, TOP_K


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "how", "i", "in", "is", "it", "of", "on", "or", "that", "the", "to", "was",
    "were", "what", "when", "where", "who", "why", "with", "you", "your", "me", "my",
}


def normalize_token(token: str) -> str:
    t = token.lower()
    if len(t) <= 2:
        return ""

    if t in {"founder", "founders", "founded", "founding"}:
        t = "found"

    if t.endswith("ies") and len(t) > 4:
        t = t[:-3] + "y"
    elif t.endswith("ing") and len(t) > 5:
        t = t[:-3]
    elif t.endswith("ed") and len(t) > 4:
        t = t[:-2]
    elif t.endswith("s") and len(t) > 3:
        t = t[:-1]

    if t in STOPWORDS:
        return ""
    return t


def tokenize(text: str) -> set[str]:
    tokens = set()
    for raw in re.findall(r"[a-z0-9]+", text.lower()):
        norm = normalize_token(raw)
        if norm:
            tokens.add(norm)
    return tokens


def build_document_frequency(chunks: list[dict]) -> dict[str, int]:
    df = {}
    for chunk in chunks:
        seen = tokenize(chunk["text"])
        for tok in seen:
            df[tok] = df.get(tok, 0) + 1
    return df


def idf(token: str, df: dict[str, int], total_docs: int) -> float:
    return math.log((total_docs + 1) / (df.get(token, 0) + 1)) + 1.0


def score_chunk(query_tokens: set[str], doc_tokens: set[str], df: dict[str, int], total_docs: int) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0

    score = 0.0
    for q in query_tokens:
        w = idf(q, df, total_docs)
        if q in doc_tokens:
            score += w
            continue

        # Typo-tolerant fallback with guardrails to avoid random matches.
        candidates = [d for d in doc_tokens if d and d[0] == q[0] and abs(len(d) - len(q)) <= 2]
        close = get_close_matches(q, candidates, n=1, cutoff=0.75)
        if close:
            score += 0.7 * w

    return score / math.sqrt(len(query_tokens) * (len(doc_tokens) + 1))


def add_query_aware_boosts(question: str, source: str, base_score: float) -> float:
    q_tokens = tokenize(question)
    s = source.lower()
    boosted = base_score

    asks_founder = "found" in q_tokens
    asks_company = "insurellm" in q_tokens

    if asks_founder and "company" in s:
        boosted += 0.08
    if asks_founder and (s.endswith("about.md") or s.endswith("overview.md")):
        boosted += 0.12
    if asks_company and "company" in s:
        boosted += 0.05

    return boosted


def load_chunks() -> list[dict]:
    if not STORE_PATH.exists():
        return []
    data = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    return data.get("chunks", [])


def retrieve(question: str, chunks: list[dict], top_k: int, df: dict[str, int]) -> list[dict]:
    q_tokens = tokenize(question)
    if not q_tokens:
        return []

    total_docs = len(chunks)
    scored = []
    for c in chunks:
        doc_tokens = tokenize(c["text"])
        base_score = score_chunk(q_tokens, doc_tokens, df, total_docs)
        score = add_query_aware_boosts(question, c["source"], base_score)
        if score > 0.02:
            scored.append({"score": score, **c})
    scored.sort(key=lambda x: x["score"], reverse=True)

    unique = []
    seen_sources = set()
    for item in scored:
        src = item["source"]
        if src in seen_sources:
            continue
        seen_sources.add(src)
        unique.append(item)
        if len(unique) >= top_k:
            break
    return unique


def is_greeting(text: str) -> bool:
    cleaned = text.strip().lower()
    # Keep letters only so variants like "hi!!" or "hello..." are handled.
    letters_only = "".join(ch for ch in cleaned if ch.isalpha())
    if not letters_only:
        return False

    if letters_only in {"hi", "hey", "hello", "yo", "hiya"}:
        return True

    # Accept common stretched forms such as "hiii" or "hellooo".
    return bool(re.fullmatch(r"h+i+|he+y+|hello+|yo+", letters_only))


def ask_groq(client: Groq, question: str, contexts: list[dict]) -> str:
    context_text = "\n\n".join(
        [f"Source: {c['source']}\n{c['text'][:900]}" for c in contexts]
    )

    messages = [
        {
            "role": "system",
            "content": "You answer only from the provided context. If context is missing, say you do not know.",
        },
        {
            "role": "user",
            "content": f"Context:\n{context_text}\n\nQuestion: {question}",
        },
    ]

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content or "No response."


def main() -> None:
    if not GROQ_API_KEY:
        print("Missing GROQ_API_KEY. Add it to .env")
        return

    chunks = load_chunks()
    if not chunks:
        print("No local store found. Run: python ingest.py")
        return

    df = build_document_frequency(chunks)

    client = Groq(api_key=GROQ_API_KEY)
    print("Simple RAG chat (type 'exit' to quit)")

    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        if is_greeting(q):
            print("Assistant: Hey! Ask me anything about Insurellm, contracts, products, or employees.\n")
            continue

        top = retrieve(q, chunks, TOP_K, df)
        if not top:
            print("Assistant: I could not find relevant context.")
            continue

        answer = ask_groq(client, q, top)
        print(f"Assistant: {answer}\n")
        print("Sources:")
        for item in top:
            print(f"- {item['source']} (score={item['score']:.2f})")
        print()


if __name__ == "__main__":
    main()
