# Advanced RAG Knowledge Worker

A production-ready Retrieval Augmented Generation (RAG) system using **Groq API** for fast LLM inference and intelligent document chunking.

## Features

✨ **Advanced Chunking**: Uses Groq LLM to intelligently split documents semantically (not just by fixed size)
🚀 **Fast Inference**: Powered by Groq's ultra-low latency API
🔍 **Vector Search**: ChromaDB for efficient similarity search
📚 **Knowledge Base**: Loads markdown documents from organized folders
💬 **Interactive Chat**: Ask questions about your knowledge base with context-aware answers
🎯 **Source Citation**: Shows which documents were used for each answer

## Project Structure

```
.
├── knowledge_base/          # Your documents organized by type
│   ├── employees/
│   ├── contracts/
│   ├── company/
│   └── products/
├── config.py                # Configuration & paths
├── ingest.py                # Document ingestion & chunking pipeline
├── answer.py                # Query interface & chat
├── requirements.txt         # Python dependencies
├── .env.example              # Environment variables template
└── README.md                # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Environment

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

Get your API keys from:
- **Groq**: https://console.groq.com/
- **OpenAI**: https://platform.openai.com/api-keys

### 3. Prepare Knowledge Base

Organize your documents in the `knowledge_base/` folder:

```
knowledge_base/
├── employees/      # HR documents, resumes, profiles
├── contracts/      # Agreements, contracts
├── company/        # Company info, policies
└── products/       # Product documentation
```

Each folder should contain `.md` files.

### 4. Ingest Documents

Run the ingestion pipeline to process documents and create embeddings:

```bash
python ingest.py
```

This will:
- Load all documents from `knowledge_base/`
- Use Groq LLM to intelligently chunk each document
- Generate embeddings for all chunks
- Store in ChromaDB vector database

**Note**: First run may take several minutes depending on document count.

### 5. Start Interactive Chat

```bash
python answer.py
```

Ask questions about your knowledge base:

```
You: What are the key qualifications needed?
🔄 Processing...