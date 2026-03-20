#!/usr/bin/env python3
"""
RAG Knowledge Worker - Unified Setup & Run Script
Works on Windows, Mac, and Linux
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report status"""
    print(f"\n[*] {description}...")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[!] ERROR: {description} failed")
        return False
    print(f"[+] OK: {description}")
    return True


def main():
    """Main setup and run workflow"""
    print("\n" + "=" * 60)
    print("  RAG Knowledge Worker - Setup & Run")
    print("=" * 60)

    project_dir = Path(__file__).parent

    # Step 1: Install dependencies
    print("\n[1/4] Installing dependencies...")
    deps = "groq python-dotenv pydantic tqdm openai"
    if not run_command(f"{sys.executable} -m pip install {deps} -q", "Install dependencies"):
        sys.exit(1)

    # Step 2: Setup .env file
    print("\n[2/4] Setting up .env configuration...")
    env_file = project_dir / ".env"
    env_example = project_dir / ".env.example"

    if not env_file.exists() and env_example.exists():
        print("[*] Creating .env from .env.example...")
        with open(env_example, "r") as src:
            with open(env_file, "w") as dst:
                dst.write(src.read())
        print("[!] Please edit .env and add your GROQ_API_KEY")
        print(f"[!] Location: {env_file}")
        input("[*] Press Enter after updating .env...")

    if not env_file.exists():
        print("[!] ERROR: .env file required with GROQ_API_KEY")
        sys.exit(1)

    # Step 3: Run ingestion
    print("\n[3/4] Running knowledge base ingestion...")
    ingest_script = project_dir / "ingest.py"
    if ingest_script.exists():
        if not run_command(f"{sys.executable} {ingest_script}", "Ingestion"):
            print("[!] Continuing anyway...")  # Don't fail on ingestion
    else:
        print(f"[!] WARNING: {ingest_script} not found")

    # Step 4: Start chats
    print("\n[4/4] Starting interactive chat...")
    answer_script = project_dir / "answer.py"
    if answer_script.exists():
        run_command(f"{sys.executable} {answer_script}", "Start chat")
    else:
        print(f"[!] ERROR: {answer_script} not found")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  ✨ Session complete!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[*] Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[!] Unexpected error: {e}")
        sys.exit(1)
