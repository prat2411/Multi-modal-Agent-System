#!/usr/bin/env python3
"""Test script for RAG system"""

from answer import RAGAnswerer
import json


def test_rag():
    """Run test queries against the RAG system"""
    
    print("🧪 Testing Advanced RAG System\n")
    
    try:
        answerer = RAGAnswerer()
    except Exception as e:
        print(f"❌ Failed to initialize RAG: {e}")
        print("Run 'python ingest.py' first to create the knowledge base")
        return
    
    # Test queries
    test_queries = [
        "What are the main responsibilities?",
        "Tell me about the company structure",
        "What skills are most important?",
        "List key products and services",
    ]
    
    results = []
    
    for i, query in enumerate(test_queries, 1):
        print(f"Test {i}/{len(test_queries)}")
        print(f"Query: {query}")
        print("-" * 50)
        
        result = answerer.answer_question(query)
        
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Context Documents: {result['context_docs']}")
        if result['sources']:
            print(f"Sources: {', '.join(result['sources'][:2])}")
        print()
        
        results.append({
            "query": query,
            "answer_length": len(result['answer']),
            "context_docs": result['context_docs'],
            "sources": result['sources'],
        })
    
    # Save results
    with open("test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("=" * 50)
    print("✅ Test complete! Results saved to test_results.json")


if __name__ == "__main__":
    test_rag()
