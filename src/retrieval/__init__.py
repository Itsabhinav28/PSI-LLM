"""
Retrieval Module

Implements document retrieval strategies including similarity search,
reranking, and query expansion for the RAG pipeline.
"""

# Import classes directly to avoid relative import issues
try:
    from .retriever import DocumentRetriever
    from .query_processor import QueryProcessor
    __all__ = ["DocumentRetriever", "QueryProcessor"]
except ImportError:
    # Fallback for direct imports
    __all__ = []
