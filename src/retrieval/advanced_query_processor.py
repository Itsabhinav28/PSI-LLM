"""
Advanced Query Processor for Phase 4

Provides advanced query processing features including:
- Query expansion and reformulation
- Semantic search optimization
- Document filtering and metadata search
- Query analytics and performance tracking
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from collections import defaultdict

logger = logging.getLogger(__name__)

class AdvancedQueryProcessor:
    """Advanced query processing with expansion, filtering, and optimization."""
    
    def __init__(self):
        self.query_history = []
        self.expansion_patterns = {
            "technical": [
                "implementation", "architecture", "design", "development",
                "deployment", "testing", "optimization", "scalability"
            ],
            "business": [
                "strategy", "market", "competition", "revenue", "growth",
                "partnership", "investment", "customer", "product"
            ],
            "research": [
                "methodology", "analysis", "findings", "conclusions",
                "recommendations", "future_work", "limitations"
            ]
        }
        
        # Common query reformulations
        self.reformulation_templates = {
            "what_is": ["define", "explain", "describe", "elaborate on"],
            "how_to": ["steps to", "process for", "method to", "approach for"],
            "compare": ["differences between", "similarities of", "vs", "versus"],
            "examples": ["examples of", "instances of", "cases of", "samples of"]
        }
    
    def process_query(
        self, 
        query: str, 
        use_expansion: bool = False,
        filters: Optional[Dict[str, Any]] = None,
        semantic_search: bool = True
    ) -> Dict[str, Any]:
        """
        Process query with advanced features.
        
        Args:
            query: Original user query
            use_expansion: Whether to use query expansion
            filters: Document filters to apply
            semantic_search: Whether to use semantic search
            
        Returns:
            Processed query information
        """
        try:
            start_time = datetime.now()
            
            # Basic query cleaning
            cleaned_query = self._clean_query(query)
            
            # Query classification
            query_type = self._classify_query(cleaned_query)
            
            # Query expansion if requested
            expanded_queries = []
            if use_expansion:
                expanded_queries = self._expand_query(cleaned_query, query_type)
            
            # Apply filters
            applied_filters = self._process_filters(filters) if filters else {}
            
            # Generate search strategies
            search_strategies = self._generate_search_strategies(
                cleaned_query, query_type, semantic_search
            )
            
            # Track query processing
            processing_time = (datetime.now() - start_time).total_seconds()
            self._track_query(query, cleaned_query, query_type, processing_time)
            
            return {
                "original_query": query,
                "cleaned_query": cleaned_query,
                "query_type": query_type,
                "expanded_queries": expanded_queries,
                "filters": applied_filters,
                "search_strategies": search_strategies,
                "processing_time": processing_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in advanced query processing: {e}")
            return {
                "original_query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _clean_query(self, query: str) -> str:
        """Clean and normalize the query."""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', query.strip())
        
        # Remove special characters but keep important ones
        cleaned = re.sub(r'[^\w\s\-\.\?\!]', ' ', cleaned)
        
        # Normalize to lowercase
        cleaned = cleaned.lower()
        
        return cleaned
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of query."""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["what", "define", "explain"]):
            return "definition"
        elif any(word in query_lower for word in ["how", "steps", "process"]):
            return "how_to"
        elif any(word in query_lower for word in ["compare", "difference", "vs"]):
            return "comparison"
        elif any(word in query_lower for word in ["example", "instance", "case"]):
            return "examples"
        elif any(word in query_lower for word in ["why", "reason", "cause"]):
            return "explanation"
        elif any(word in query_lower for word in ["when", "date", "time"]):
            return "temporal"
        elif any(word in query_lower for word in ["where", "location", "place"]):
            return "location"
        else:
            return "general"
    
    def _expand_query(self, query: str, query_type: str) -> List[str]:
        """Expand query with related terms and reformulations."""
        expanded = [query]
        
        # Add domain-specific expansions
        for domain, terms in self.expansion_patterns.items():
            if any(term in query for term in terms):
                for term in terms[:3]:  # Limit to 3 terms per domain
                    if term not in query:
                        expanded.append(f"{query} {term}")
        
        # Add reformulations based on query type
        if query_type in self.reformulation_templates:
            for template in self.reformulation_templates[query_type][:2]:
                if template not in query:
                    expanded.append(f"{template} {query}")
        
        # Add synonyms for common terms
        synonyms = {
            "implement": ["build", "create", "develop", "construct"],
            "optimize": ["improve", "enhance", "boost", "maximize"],
            "analyze": ["examine", "study", "investigate", "review"],
            "design": ["plan", "architect", "structure", "model"]
        }
        
        for original, syns in synonyms.items():
            if original in query:
                for syn in syns[:2]:
                    expanded.append(query.replace(original, syn))
        
        return list(set(expanded))  # Remove duplicates
    
    def _process_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate document filters."""
        processed_filters = {}
        
        # File type filter
        if "file_type" in filters:
            allowed_types = ["pdf", "docx", "txt", "html"]
            file_type = filters["file_type"].lower()
            if file_type in allowed_types:
                processed_filters["file_type"] = file_type
        
        # Date range filter
        if "date_from" in filters or "date_to" in filters:
            try:
                if "date_from" in filters:
                    date_from = datetime.fromisoformat(filters["date_from"])
                    processed_filters["date_from"] = date_from
                
                if "date_to" in filters:
                    date_to = datetime.fromisoformat(filters["date_to"])
                    processed_filters["date_to"] = date_to
                    
                # Validate date range
                if "date_from" in processed_filters and "date_to" in processed_filters:
                    if processed_filters["date_from"] > processed_filters["date_to"]:
                        logger.warning("Invalid date range, ignoring date filters")
                        processed_filters.pop("date_from")
                        processed_filters.pop("date_to")
                        
            except ValueError:
                logger.warning("Invalid date format, ignoring date filters")
        
        # Content length filter
        if "min_length" in filters:
            try:
                min_length = int(filters["min_length"])
                if min_length > 0:
                    processed_filters["min_length"] = min_length
            except ValueError:
                logger.warning("Invalid min_length, ignoring")
        
        if "max_length" in filters:
            try:
                max_length = int(filters["max_length"])
                if max_length > 0:
                    processed_filters["max_length"] = max_length
            except ValueError:
                logger.warning("Invalid max_length, ignoring")
        
        # Source filter
        if "source" in filters:
            processed_filters["source"] = str(filters["source"])
        
        return processed_filters
    
    def _generate_search_strategies(
        self, 
        query: str, 
        query_type: str, 
        semantic_search: bool
    ) -> List[Dict[str, Any]]:
        """Generate different search strategies for the query."""
        strategies = []
        
        # Strategy 1: Exact match
        strategies.append({
            "name": "exact_match",
            "description": "Find exact text matches",
            "priority": 1 if query_type == "definition" else 3,
            "parameters": {"exact_match": True}
        })
        
        # Strategy 2: Semantic search
        if semantic_search:
            strategies.append({
                "name": "semantic_search",
                "description": "Find semantically similar content",
                "priority": 1 if query_type in ["general", "explanation"] else 2,
                "parameters": {"semantic_search": True, "similarity_threshold": 0.7}
            })
        
        # Strategy 3: Keyword-based search
        keywords = self._extract_keywords(query)
        if keywords:
            strategies.append({
                "name": "keyword_search",
                "description": "Search by extracted keywords",
                "priority": 2,
                "parameters": {"keywords": keywords, "keyword_boost": 1.5}
            })
        
        # Strategy 4: Contextual search
        if query_type in ["how_to", "comparison"]:
            strategies.append({
                "name": "contextual_search",
                "description": "Search for contextual information",
                "priority": 1,
                "parameters": {"context_window": 100, "context_boost": 1.2}
            })
        
        # Sort strategies by priority
        strategies.sort(key=lambda x: x["priority"])
        
        return strategies
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from the query."""
        # Remove common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "what", "how", "why", "when", "where"
        }
        
        words = query.split()
        keywords = [word for word in words if word.lower() not in stop_words and len(word) > 2]
        
        return keywords[:5]  # Limit to 5 keywords
    
    def _track_query(self, original: str, cleaned: str, query_type: str, processing_time: float):
        """Track query processing for analytics."""
        self.query_history.append({
            "original": original,
            "cleaned": cleaned,
            "type": query_type,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 100 queries
        if len(self.query_history) > 100:
            self.query_history = self.query_history[-100:]
    
    def get_query_analytics(self) -> Dict[str, Any]:
        """Get analytics about query processing."""
        if not self.query_history:
            return {"message": "No query history available"}
        
        # Calculate average processing time
        avg_time = sum(q["processing_time"] for q in self.query_history) / len(self.query_history)
        
        # Count query types
        type_counts = defaultdict(int)
        for query in self.query_history:
            type_counts[query["type"]] += 1
        
        # Find slowest queries
        slowest_queries = sorted(
            self.query_history, 
            key=lambda x: x["processing_time"], 
            reverse=True
        )[:5]
        
        return {
            "total_queries": len(self.query_history),
            "avg_processing_time": round(avg_time, 3),
            "query_type_distribution": dict(type_counts),
            "slowest_queries": [
                {
                    "query": q["original"][:50] + "..." if len(q["original"]) > 50 else q["original"],
                    "processing_time": q["processing_time"],
                    "type": q["type"]
                }
                for q in slowest_queries
            ],
            "recent_queries": len([q for q in self.query_history 
                                 if datetime.now() - datetime.fromisoformat(q["timestamp"]) < timedelta(hours=1)])
        }
    
    def optimize_query(self, query: str, performance_data: Dict[str, Any]) -> str:
        """Optimize query based on performance data."""
        # This is a placeholder for query optimization logic
        # In a real implementation, you would use ML models or heuristics
        # to improve query performance based on historical data
        
        return query
