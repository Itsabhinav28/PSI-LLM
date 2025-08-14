"""
Generation Module

Handles LLM integration and response generation using retrieved
context for the RAG pipeline.
"""

from .gemini_client import GeminiClient

__all__ = ["GeminiClient"]
