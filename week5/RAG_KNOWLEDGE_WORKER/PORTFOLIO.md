# Advanced RAG Knowledge Worker - Portfolio Project

## Overview

This is a **production-grade RAG (Retrieval Augmented Generation) system** that demonstrates:

- 🤖 **Intelligent Document Chunking**: Using LLM (Groq) to split documents semantically
- 🚀 **High-Performance Inference**: Groq API for ultra-fast LLM responses
- 🔍 **Vector Search**: ChromaDB with OpenAI embeddings
- 💬 **Conversational AI**: Multi-turn chat with context awareness
- 📊 **Source Attribution**: Tracks which documents answered each question

## Architecture

```
User Question
    ↓
[Embedding] → [Vector Search] → Retrieve Relevant Chunks
    ↓
[Format Context]
    ↓
[Groq LLM] → Generate Answer with Context
    ↓
[Format & Return] → Answer + Sources
```

## Technologies Used

- **Groq API** - Fast LLM inference (used for intelligent chunking AND answering)
- **ChromaDB** - Vector database for semantic search
- **OpenAI** - Text embeddings (text-embedding-3-large)
- **Pydantic** - Data validation
- **Python 3.10+**

## Key Files

| File | Purpose |
|------|---------|
| `config.py` | Centralized configuration and constants |
| `ingest.py` | Document processing pipeline with intelligent chunking |
| `answer.py` | Query interface and chat functionality |
| `utils.py` | Helper functions for statistics and debugging |
| `test_rag.py` | Test queries to validate the system |

## Advanced Features

### 1. Intelligent Chunking (Day 5 Approach)

Instead of fixed-size chunks, the system uses Groq to:
- Understand document semantics
- Create headlines for better searchability
- Generate summaries for quick context
- Maintain overlapping chunks for connected retrieval

**vs Basic Approach (Day 4)**: Fixed 500-char chunks without semantic understanding

### 2. Multi-turn Conversation

The system maintains conversation history for:
- Context-aware follow-up questions
- Reference to previous answers
- Coherent dialogue flow

### 3. Source Tracking

Every answer includes:
- Which documents were used
- Number of context chunks retrieved
- Direct source file references

## Performance Metrics

- **Chunking Speed**: ~2-5 documents/minute (depends on Groq API)
- **Query Latency**: <2 seconds (low-latency Groq model)
- **Vector Search**: <100ms for k=10 retrieval
- **Scalability**: Tested with 300+ documents

## Usage Examples

### Basic Query
```python
answerer = RAGAnswerer()
result = answerer.answer_question("What are the requirements?")
print(result["answer"])
print(f"Used {result['context_docs']} documents")
```

### Conversation with History
```python
history = []
while True:
    question = input("You: ")
    result = answerer.answer_question(question, history)
    print(f"Assistant: {result['answer']}")
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": result['answer']})
```

### Get Statistics
```python
from utils import display_vector_store_stats
display_vector_store_stats()
```

## Portfolio Highlights

✅ **Production-Ready Code**: Error handling, logging, proper structure
✅ **Advanced AI Techniques**: LLM-based chunking, context retrieval
✅ **Clean Architecture**: Separation of concerns (config, ingest, query)
✅ **Scalability**: Works with large document collections
✅ **Documentation**: Comprehensive README and inline comments
✅ **Testing**: Included test suite with multiple query types

## Setup & Installation

See [README.md](README.md) for detailed setup instructions.

## Learning Resources

This project demonstrates:
- How to build RAG systems from scratch
- When to use LLMs vs traditional NLP
- Vector database operations
- API integration (Groq, OpenAI)
- Python best practices

## Future Enhancements

- [ ] Knowledge graph extraction from documents
- [ ] Multi-modal (text + images) support
- [ ] Web UI with FastAPI
- [ ] Evaluation metrics (RAGAS)
- [ ] Caching for repeated queries
- [ ] Fine-tuned embeddings for domain-specific searches
