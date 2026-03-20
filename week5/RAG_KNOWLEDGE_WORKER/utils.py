"""Utility functions for RAG system"""

import json
from pathlib import Path
from chromadb import PersistentClient
from config import DB_NAME, CHROMA_COLLECTION


def get_vector_store_stats() -> dict:
    """Get statistics about the vector store"""
    try:
        chroma = PersistentClient(path=str(DB_NAME))
        collection = chroma.get_collection(CHROMA_COLLECTION)
        
        # Get all documents
        all_docs = collection.get(include=["metadatas", "documents"])
        
        # Calculate stats
        doc_types = {}
        sources = {}
        
        for metadata in all_docs["metadatas"]:
            doc_type = metadata.get("doc_type", "unknown")
            source = metadata.get("source", "unknown")
            
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            sources[source] = sources.get(source, 0) + 1
        
        return {
            "total_chunks": collection.count(),
            "by_type": doc_types,
            "by_source": sources,
        }
    
    except Exception as e:
        return {"error": str(e)}


def display_vector_store_stats():
    """Display formatted statistics"""
    stats = get_vector_store_stats()
    
    if "error" in stats:
        print(f"❌ Error: {stats['error']}")
        return
    
    print("\n" + "=" * 60)
    print("📊 Vector Store Statistics")
    print("=" * 60)
    print(f"Total Chunks: {stats['total_chunks']}")
    
    print("\nChunks by Type:")
    for doc_type, count in stats["by_type"].items():
        print(f"  • {doc_type}: {count}")
    
    print(f"\nSources ({len(stats['by_source'])} files):")
    for source, count in sorted(stats["by_source"].items(), key=lambda x: x[1], reverse=True)[:5]:
        filename = Path(source).name
        print(f"  • {filename}: {count}")
    
    print("=" * 60 + "\n")


def reset_vector_store():
    """Delete vector store to start fresh"""
    try:
        chroma = PersistentClient(path=str(DB_NAME))
        chroma.delete_collection(CHROMA_COLLECTION)
        print("✅ Vector store reset successfully")
    except Exception as e:
        print(f"❌ Error resetting vector store: {e}")


def export_chunks_to_json(output_file: str = "chunks_export.json"):
    """Export all chunks to JSON file"""
    try:
        chroma = PersistentClient(path=str(DB_NAME))
        collection = chroma.get_collection(CHROMA_COLLECTION)
        
        all_docs = collection.get(include=["metadatas", "documents"])
        
        export_data = {
            "metadata": {
                "total_chunks": collection.count(),
                "collection": CHROMA_COLLECTION,
            },
            "chunks": [
                {
                    "content": doc,
                    "metadata": meta,
                }
                for doc, meta in zip(all_docs["documents"], all_docs["metadatas"])
            ],
        }
        
        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✅ Exported {len(export_data['chunks'])} chunks to {output_file}")
    
    except Exception as e:
        print(f"❌ Error exporting chunks: {e}")


if __name__ == "__main__":
    display_vector_store_stats()
