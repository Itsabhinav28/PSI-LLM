"""
Retrieval System for RAG Pipeline

Handles query processing, document retrieval, and result ranking
to provide relevant context for the generation system.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .query_processor import QueryProcessor
from ..vector_store.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a retrieval result with metadata."""
    content: str
    metadata: Dict[str, Any]
    similarity_score: float
    rank: int
    source: str


class DocumentRetriever:
    """Main document retriever for the RAG pipeline."""
    
    def __init__(
        self,
        vector_store: ChromaStore,
        query_processor: Optional[QueryProcessor] = None,
        max_results: int = 5,
        similarity_threshold: float = 0.5
    ):
        """Initialize document retriever."""
        self.vector_store = vector_store
        self.query_processor = query_processor or QueryProcessor()
        self.max_results = max_results
        self.similarity_threshold = similarity_threshold
        
        logger.info("Document retriever initialized")
    
    def retrieve_documents(
        self,
        query: str,
        n_results: Optional[int] = None,
        similarity_threshold: Optional[float] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant documents for a query."""
        try:
            # Process the query
            processed_query = self.query_processor.process_query(query)
            logger.info(f"Processed query: {query} -> {processed_query}")
            
            # Set parameters
            n_results = n_results or self.max_results
            threshold = similarity_threshold or self.similarity_threshold
            
            # Search vector store
            raw_results = self.vector_store.search_documents(
                query=processed_query,
                n_results=n_results,
                similarity_threshold=threshold
            )
            
            # Convert to RetrievalResult objects
            results = []
            for result in raw_results:
                retrieval_result = RetrievalResult(
                    content=result['content'],
                    metadata=result['metadata'],
                    similarity_score=result['similarity_score'],
                    rank=result['rank'],
                    source=result['metadata'].get('file_name', 'unknown')
                )
                results.append(retrieval_result)
            
            logger.info(f"Retrieved {len(results)} documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    def retrieve_with_reranking(
        self,
        query: str,
        n_results: int = 10,
        final_results: int = 5
    ) -> List[RetrievalResult]:
        """Retrieve documents with reranking for better relevance."""
        try:
            # Get more results initially for reranking
            initial_results = self.retrieve_documents(
                query=query,
                n_results=n_results,
                similarity_threshold=0.3  # Lower threshold for more candidates
            )
            
            if not initial_results:
                return []
            
            # Apply reranking based on multiple factors
            reranked_results = self._rerank_results(query, initial_results)
            
            # Return top results
            final_results = min(final_results, len(reranked_results))
            return reranked_results[:final_results]
            
        except Exception as e:
            logger.error(f"Error in reranked retrieval: {e}")
            return []
    
    def _rerank_results(
        self,
        query: str,
        results: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """Rerank results based on multiple relevance factors."""
        try:
            # Calculate additional relevance scores
            for result in results:
                # Content length score (prefer medium-length documents)
                length_score = self._calculate_length_score(result.content)
                
                # Content freshness score (if metadata has timestamp)
                freshness_score = self._calculate_freshness_score(result.metadata)
                
                # Query term density score
                density_score = self._calculate_term_density_score(query, result.content)
                
                # Combined score (weighted average)
                combined_score = (
                    result.similarity_score * 0.5 +
                    length_score * 0.2 +
                    freshness_score * 0.1 +
                    density_score * 0.2
                )
                
                # Store the combined score
                result.metadata['combined_score'] = round(combined_score, 4)
            
            # Sort by combined score
            reranked_results = sorted(
                results,
                key=lambda x: x.metadata['combined_score'],
                reverse=True
            )
            
            # Update ranks
            for i, result in enumerate(reranked_results):
                result.rank = i + 1
            
            logger.info("Results reranked successfully")
            return reranked_results
            
        except Exception as e:
            logger.error(f"Error reranking results: {e}")
            return results
    
    def _calculate_length_score(self, content: str) -> float:
        """Calculate score based on content length."""
        length = len(content)
        
        # Prefer documents between 100-1000 characters
        if 100 <= length <= 1000:
            return 1.0
        elif length < 100:
            return 0.3
        elif length > 2000:
            return 0.6
        else:
            return 0.8
    
    def _calculate_freshness_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate score based on document freshness."""
        # Default score if no timestamp available
        return 0.5
    
    def _calculate_term_density_score(self, query: str, content: str) -> float:
        """Calculate score based on query term density in content."""
        try:
            # Extract key terms from query
            query_terms = re.findall(r'\b\w+\b', query.lower())
            
            if not query_terms:
                return 0.5
            
            # Count term occurrences
            content_lower = content.lower()
            term_counts = sum(content_lower.count(term) for term in query_terms)
            
            # Calculate density score
            if len(content) > 0:
                density = term_counts / len(content) * 1000  # Normalize
                return min(1.0, density / 10)  # Cap at 1.0
            
            return 0.5
            
        except Exception:
            return 0.5
    
    def get_retrieval_stats(self, query: str) -> Dict[str, Any]:
        """Get statistics about the retrieval process."""
        try:
            results = self.retrieve_documents(query, n_results=10)
            
            if not results:
                return {"error": "No results found"}
            
            # Calculate statistics
            scores = [r.similarity_score for r in results]
            lengths = [len(r.content) for r in results]
            
            stats = {
                "query": query,
                "total_results": len(results),
                "average_similarity": round(sum(scores) / len(scores), 4),
                "min_similarity": min(scores),
                "max_similarity": max(scores),
                "average_length": round(sum(lengths) / len(lengths), 2),
                "min_length": min(lengths),
                "max_length": max(lengths),
                "sources": list(set(r.source for r in results))
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting retrieval stats: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the retriever."""
        try:
            # Test vector store connection
            vector_store_health = self.vector_store.health_check()
            
            # Test query processing
            test_query = "test query"
            processed_query = self.query_processor.process_query(test_query)
            
            health_status = {
                "status": "healthy" if vector_store_health["status"] == "healthy" else "unhealthy",
                "vector_store": vector_store_health,
                "query_processor": "working",
                "retriever": "working"
            }
            
            return health_status
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "vector_store": "failed",
                "query_processor": "failed",
                "retriever": "failed"
            }
