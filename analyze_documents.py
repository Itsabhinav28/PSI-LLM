"""
Analyze existing documents to create realistic test data
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.vector_store.chroma_store import ChromaStore
from src.retrieval.retriever import DocumentRetriever
import json

def analyze_documents():
    """Analyze existing documents to understand their content"""
    print("🔍 Analyzing existing documents...")
    
    # Initialize components
    store = ChromaStore()
    retriever = DocumentRetriever(store)
    
    # Test queries to see what content is available
    test_queries = [
        "machine learning",
        "artificial intelligence", 
        "data science",
        "resume",
        "experience",
        "skills",
        "education",
        "work",
        "project",
        "technology"
    ]
    
    print("\n📊 Document Analysis Results:")
    print("=" * 50)
    
    all_documents = []
    
    for query in test_queries:
        print(f"\n🔍 Searching for: '{query}'")
        try:
            results = retriever.retrieve_documents(query, n_results=3)
            print(f"   Found {len(results)} documents")
            
            for i, result in enumerate(results):
                content_preview = result.content[:150] + "..." if len(result.content) > 150 else result.content
                print(f"   {i+1}. Score: {result.similarity_score:.3f}")
                print(f"      Content: {content_preview}")
                print(f"      Metadata: {result.metadata}")
                print()
                
                # Collect unique documents
                doc_id = result.metadata.get('document_id', f'unknown_{i}')
                if doc_id not in [d['id'] for d in all_documents]:
                    all_documents.append({
                        'id': doc_id,
                        'content': result.content,
                        'metadata': result.metadata,
                        'similarity_score': result.similarity_score
                    })
                    
        except Exception as e:
            print(f"   Error: {e}")
    
    print(f"\n📈 Summary:")
    print(f"Total unique documents found: {len(all_documents)}")
    
    # Save analysis results
    with open("document_analysis.json", "w") as f:
        json.dump(all_documents, f, indent=2)
    
    print("💾 Analysis saved to document_analysis.json")
    
    return all_documents

if __name__ == "__main__":
    analyze_documents()

