"""
Data Ingestion Module

Handles document loading, preprocessing, and chunking for the RAG pipeline.
Supports multiple document formats and implements advanced text segmentation.
"""

from .document_loader import DocumentLoader
from .text_chunker import TextChunker
from .preprocessor import DocumentPreprocessor

__all__ = ["DocumentLoader", "TextChunker", "DocumentPreprocessor"]
