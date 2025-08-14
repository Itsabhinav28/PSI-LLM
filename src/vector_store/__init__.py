"""
Vector Store Module

Handles vector database operations including ChromaDB and FAISS integration
for storing and retrieving document embeddings.
"""

from .chroma_store import ChromaStore

__all__ = ["ChromaStore"]
