"""
Query Processor for RAG Pipeline

Handles query preprocessing, enhancement, and optimization
to improve document retrieval accuracy.
"""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class QueryProcessor:
    """Processes and enhances user queries for better retrieval."""
    
    def __init__(self):
        """Initialize query processor."""
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
        }
    
    def process_query(self, query: str) -> str:
        """Process and enhance a user query."""
        if not query or not query.strip():
            return query
        
        # Clean and normalize
        processed = query.strip().lower()
        
        # Remove extra whitespace
        processed = re.sub(r'\s+', ' ', processed)
        
        # Remove punctuation (keep important ones)
        processed = re.sub(r'[^\w\s\-]', ' ', processed)
        
        # Remove stop words for better vector matching
        words = processed.split()
        filtered_words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        # Reconstruct query
        processed = ' '.join(filtered_words)
        
        logger.info(f"Query processed: '{query}' -> '{processed}'")
        return processed
    
    def expand_query(self, query: str, expansion_terms: List[str] = None) -> str:
        """Expand query with related terms."""
        if not expansion_terms:
            return query
        
        expanded = f"{query} {' '.join(expansion_terms)}"
        logger.info(f"Query expanded: '{query}' -> '{expanded}'")
        return expanded
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract key terms from query."""
        processed = self.process_query(query)
        keywords = [word for word in processed.split() if len(word) > 2]
        return keywords
