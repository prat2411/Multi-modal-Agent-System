#!/usr/bin/env python3
"""Minimal setup checker for simple Groq RAG."""

from pathlib import Path


def check_imports() -> bool:
    ok = True
    for mod in ["groq", "dotenv"]:
        try:
            __import__(mod)
            print(f"OK: {mod}")
        except ImportError:
            ok = False
            print(f"MISSING: {mod}")
    return ok


def check_files() -> bool:
    env_ok = Path(".env").exists()
    kb_ok = Path("knowledge_base").exists()
    print(f".env exists: {env_ok}")
    print(f"knowledge_base exists: {kb_ok}")
    return env_ok and kb_ok


def main() -> None:
    print("Simple Groq RAG setup check")
    imports_ok = check_imports()
    files_ok = check_files()
    print("ready:" if imports_ok and files_ok else "not ready")


if __name__ == "__main__":
    main()
