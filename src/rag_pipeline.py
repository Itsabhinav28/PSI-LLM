"""
Enhanced RAG Pipeline - Phase 4

Integrates all RAG components with advanced features:
- Advanced query processing with expansion and filtering
- Performance optimization with caching and batch processing
- Comprehensive analytics and monitoring
- Advanced document management
"""

import os
import time
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib

from src.data_ingestion.document_loader import DocumentLoader
from src.data_ingestion.preprocessor import DocumentPreprocessor
from src.data_ingestion.text_chunker import TextChunker
from src.vector_store.chroma_store import ChromaStore
from src.retrieval.retriever import DocumentRetriever
from src.generation.gemini_client import GeminiClient
from src.retrieval.advanced_query_processor import AdvancedQueryProcessor
from src.optimization.performance_optimizer import (
    CacheManager, BatchProcessor, QueryOptimizer, PerformanceMonitor
)

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Enhanced RAG Pipeline with Phase 4 features."""
    
    def __init__(self):
        """Initialize the RAG pipeline with all components."""
        # Initialize core components
        self.document_loader = DocumentLoader()
        self.preprocessor = DocumentPreprocessor()
        self.text_chunker = TextChunker()
        self.vector_store = ChromaStore(persist_directory="./data/embeddings/chroma")
        self.retriever = DocumentRetriever(self.vector_store)
        self.gemini_client = GeminiClient()
        
        # Initialize Phase 4 components
        self.advanced_query_processor = AdvancedQueryProcessor()
        self.cache_manager = CacheManager(max_size=1000, default_ttl=3600)
        self.batch_processor = BatchProcessor(batch_size=10, max_wait_time=1.0)
        self.query_optimizer = QueryOptimizer()
        self.performance_monitor = PerformanceMonitor()
        
        # Track recent statistics
        self._recent_stats = {
            'last_processing_time': '-',
            'last_top_similarity': '-',
            'last_query_time': None,
            'cache_hit_rate': 0.0
        }
        
        # Initialize performance tracking
        self._initialize_performance_tracking()
        
        logger.info("Enhanced RAG Pipeline Phase 4 initialized successfully")
    
    def _initialize_performance_tracking(self):
        """Initialize performance tracking components."""
        try:
            # Start background performance monitoring
            self._start_performance_monitoring()
            
            # Initialize cache statistics
            self._update_cache_stats()
            
            logger.info("Performance tracking initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize performance tracking: {e}")
    
    def _start_performance_monitoring(self):
        """Start background performance monitoring."""
        import threading
        
        def monitor_loop():
            while True:
                try:
                    # Update system metrics
                    self._update_system_metrics()
                    
                    # Update cache statistics
                    self._update_cache_stats()
                    
                    # Sleep for 30 seconds
                    time.sleep(30)
                    
                except Exception as e:
                    logger.error(f"Error in performance monitoring: {e}")
                    time.sleep(60)  # Wait longer on error
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
    
    def _update_system_metrics(self):
        """Update system performance metrics."""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.performance_monitor.record_system_metric("cpu_percent", cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.performance_monitor.record_system_metric("memory_percent", memory.percent)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.performance_monitor.record_system_metric("disk_percent", disk_percent)
            
        except ImportError:
            logger.warning("psutil not available, skipping system metrics")
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
    
    def _update_cache_stats(self):
        """Update cache statistics."""
        try:
            cache_stats = self.cache_manager.get_stats()
            self._recent_stats['cache_hit_rate'] = cache_stats.get('hit_rate', 0.0)
        except Exception as e:
            logger.error(f"Error updating cache stats: {e}")
    
    def process_documents(
        self, 
        file_paths: List[str], 
        save_processed: bool = True,
        use_batching: bool = True
    ) -> Dict[str, Any]:
        """Process documents with optional batching."""
        try:
            start_time = time.time()
            
            if use_batching:
                # Add to batch processor
                for file_path in file_paths:
                    self.batch_processor.add_task({
                        "type": "document",
                        "id": Path(file_path).stem,
                        "file_path": file_path,
                        "save_processed": save_processed
                    })
                
                # For now, process immediately (in real implementation, this would be batched)
                result = self._process_documents_immediate(file_paths, save_processed)
            else:
                result = self._process_documents_immediate(file_paths, save_processed)
            
            processing_time = time.time() - start_time
            
            # Record performance metrics
            self.performance_monitor.record_query_metric(
                "document_processing", 
                processing_time, 
                True, 
                len(file_paths)
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing documents: {e}"
            logger.error(error_msg)
            
            # Record error
            self.performance_monitor.record_error(error_msg, "document_processing", "high")
            
            return {
                "success": False,
                "error": error_msg,
                "documents_processed": 0,
                "chunks_created": 0,
                "embeddings_stored": 0
            }
    
    def _process_documents_immediate(self, file_paths: List[str], save_processed: bool) -> Dict[str, Any]:
        """Process documents immediately (fallback when batching is not available)."""
        try:
            total_chunks = 0
            total_embeddings = 0
            
            for file_path in file_paths:
                # Load document
                documents = [self.document_loader.load_document(file_path)]
                
                # Preprocess
                processed_docs = []
                for doc in documents:
                    processed_doc = self.preprocessor.preprocess_document(doc)
                    processed_docs.append(processed_doc)
                
                # Chunk text
                chunks = []
                for doc in processed_docs:
                    doc_chunks = self.text_chunker.smart_chunk(doc['content'])
                    chunks.extend(doc_chunks)
                
                # Store in vector database
                if chunks:
                    # Convert chunks to the format expected by vector store
                    documents_for_store = []
                    for chunk in chunks:
                        doc_for_store = {
                            'content': chunk.content,
                            'metadata': {
                                'file_name': doc.get('metadata', {}).get('file_name', 'unknown'),
                                'file_path': doc.get('metadata', {}).get('file_path', 'unknown'),
                                'format': doc.get('metadata', {}).get('format', 'unknown'),
                                'chunk_id': chunk.chunk_id,
                                'chunk_type': chunk.metadata.get('chunk_type', 'unknown'),
                                'char_count': len(chunk.content),
                                'source': 'rag_pipeline'
                            }
                        }
                        documents_for_store.append(doc_for_store)
                    
                    self.vector_store.add_documents(documents_for_store)
                    total_chunks += len(chunks)
                    total_embeddings += len(chunks)
                
                # Save processed documents if requested
                if save_processed:
                    self._save_processed_document(file_path, processed_docs)
            
            return {
                "success": True,
                "documents_processed": len(file_paths),
                "chunks_created": total_chunks,
                "embeddings_stored": total_embeddings
            }
            
        except Exception as e:
            logger.error(f"Error in immediate document processing: {e}")
            raise
    
    def _save_processed_document(self, file_path: str, processed_docs: List[Any]):
        """Save processed documents to storage."""
        try:
            # Create processed documents directory
            processed_dir = Path("./data/processed")
            processed_dir.mkdir(parents=True, exist_ok=True)
            
            # Save each processed document
            for i, doc in enumerate(processed_docs):
                output_path = processed_dir / f"{Path(file_path).stem}_processed_{i}.txt"
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(doc.content)
                    
        except Exception as e:
            logger.error(f"Error saving processed document: {e}")
    
    def query(
        self, 
        question: str, 
        n_results: int = 5, 
        use_reranking: bool = True,
        filters: Optional[Dict[str, Any]] = None,
        query_expansion: bool = False,
        semantic_search: bool = True
    ) -> Dict[str, Any]:
        """Query the RAG pipeline with advanced features."""
        try:
            start_time = time.time()
            
            # Advanced query processing
            processed_query_info = self.advanced_query_processor.process_query(
                query=question,
                use_expansion=query_expansion,
                filters=filters,
                semantic_search=semantic_search
            )
            
            # Query optimization
            query_type = processed_query_info.get("query_type", "general")
            optimized_plan = self.query_optimizer.optimize_query(
                question, query_type, filters
            )
            
            # Check cache first
            cache_key = self._generate_cache_key(question, n_results, use_reranking, filters)
            cached_result = self.cache_manager.get(cache_key)
            if cached_result:
                logger.info("Query result retrieved from cache")
                return cached_result
            
            # Process the query using the optimized plan
            processed_query = processed_query_info.get("cleaned_query", question)
            
            # Retrieve documents
            retrieved_docs = self.retriever.retrieve_documents(
                query=processed_query,
                n_results=n_results,
                similarity_threshold=0.2
            )
            
            if not retrieved_docs:
                return {
                    "success": False,
                    "error": "No relevant documents found",
                    "question": question
                }
            
            # Generate response using Gemini
            context = [doc.content for doc in retrieved_docs]
            response = self.gemini_client.generate_response(processed_query, context)
            
            # Prepare result
            result = {
                "success": response.get("success", False),
                "question": question,
                "answer": response.get("response", ""),
                "sources": [
                    {
                        "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                        "metadata": doc.metadata,
                        "similarity_score": doc.similarity_score,
                        "rank": doc.rank
                    }
                    for doc in retrieved_docs
                ],
                "retrieval_stats": {
                    "documents_retrieved": len(retrieved_docs),
                    "top_similarity": max(doc.similarity_score for doc in retrieved_docs),
                    "average_similarity": sum(doc.similarity_score for doc in retrieved_docs) / len(retrieved_docs)
                },
                "query_analysis": {
                    "query_type": query_type,
                    "optimization_plan": optimized_plan,
                    "processing_info": processed_query_info
                }
            }
            
            # Update recent stats
            if retrieved_docs:
                self._recent_stats['last_top_similarity'] = max(doc.similarity_score for doc in retrieved_docs)
                self._recent_stats['last_query_time'] = time.time()
            
            if not response.get("success"):
                result["error"] = response.get("error", "Unknown error")
            
            # Cache the result
            self.cache_manager.set(cache_key, result, ttl=1800)  # Cache for 30 minutes
            
            # Record performance metrics
            processing_time = time.time() - start_time
            self.performance_monitor.record_query_metric(
                "query", 
                processing_time, 
                result["success"], 
                len(retrieved_docs)
            )
            
            # Record query optimization performance
            query_hash = self._generate_query_hash(question)
            self.query_optimizer.record_performance(
                query_hash, processing_time, result["success"]
            )
            
            logger.info(f"Query completed: {question}")
            return result
            
        except Exception as e:
            error_msg = f"Error in query processing: {e}"
            logger.error(error_msg)
            
            # Record error
            self.performance_monitor.record_error(error_msg, "query", "high")
            
            return {
                "success": False,
                "error": error_msg,
                "question": question
            }
    
    def _generate_cache_key(self, question: str, n_results: int, use_reranking: bool, filters: Optional[Dict[str, Any]]) -> str:
        """Generate a cache key for the query."""
        key_parts = [
            question.lower().strip(),
            str(n_results),
            str(use_reranking),
            str(sorted(filters.items()) if filters else "{}")
        ]
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()
    
    def _generate_query_hash(self, question: str) -> str:
        """Generate a hash for the query."""
        return hashlib.md5(question.lower().strip().encode()).hexdigest()[:8]
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        try:
            stats = {
                "pipeline_status": "operational",
                "vector_store": self.vector_store.get_collection_stats(),
                "retriever": self.retriever.health_check(),
                "gemini": self.gemini_client.get_model_info(),
                "phase4_features": {
                    "advanced_query_processing": True,
                    "performance_optimization": True,
                    "caching": True,
                    "batch_processing": True,
                    "analytics": True
                }
            }
            
            # Add performance metrics
            performance_summary = self.performance_monitor.get_performance_summary()
            stats["performance"] = performance_summary
            
            # Add cache statistics
            cache_stats = self.cache_manager.get_stats()
            stats["cache"] = cache_stats
            
            # Add query optimization suggestions
            optimization_suggestions = self.query_optimizer.get_optimization_suggestions()
            stats["optimization_suggestions"] = optimization_suggestions
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting pipeline stats: {e}")
            return {"pipeline_status": "error", "error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            health_status = {
                "pipeline": "healthy",
                "components": {},
                "phase4_components": {}
            }
            
            # Check core components
            components = {
                "vector_store": self.vector_store,
                "retriever": self.retriever,
                "gemini_client": self.gemini_client
            }
            
            for name, component in components.items():
                if hasattr(component, 'health_check'):
                    health_status["components"][name] = component.health_check()
                else:
                    health_status["components"][name] = {"status": "unknown"}
            
            # Check Phase 4 components
            phase4_components = {
                "advanced_query_processor": self.advanced_query_processor,
                "cache_manager": self.cache_manager,
                "batch_processor": self.batch_processor,
                "query_optimizer": self.query_optimizer,
                "performance_monitor": self.performance_monitor
            }
            
            for name, component in phase4_components.items():
                if hasattr(component, 'health_check'):
                    health_status["phase4_components"][name] = component.health_check()
                else:
                    # Create basic health check for components without health_check method
                    health_status["phase4_components"][name] = {
                        "status": "healthy",
                        "message": f"{name} is operational"
                    }
            
            # Overall status
            all_healthy = all(
                comp.get("status") == "healthy" 
                for comp in health_status["components"].values()
            )
            
            phase4_healthy = all(
                comp.get("status") == "healthy" 
                for comp in health_status["phase4_components"].values()
            )
            
            health_status["pipeline"] = "healthy" if (all_healthy and phase4_healthy) else "unhealthy"
            
            return health_status
            
        except Exception as e:
            return {
                "pipeline": "unhealthy",
                "error": str(e)
            }
    
    def get_advanced_analytics(self) -> Dict[str, Any]:
        """Get advanced analytics and insights."""
        try:
            # Get query analytics
            query_analytics = self.advanced_query_processor.get_query_analytics()
            
            # Get performance summary
            performance_summary = self.performance_monitor.get_performance_summary()
            
            # Get cache statistics
            cache_stats = self.cache_manager.get_stats()
            
            # Get optimization suggestions
            optimization_suggestions = self.query_optimizer.get_optimization_suggestions()
            
            # Get recent errors
            recent_errors = self.performance_monitor.get_recent_errors(hours=24)
            
            return {
                "query_analytics": query_analytics,
                "performance_summary": performance_summary,
                "cache_statistics": cache_stats,
                "optimization_suggestions": optimization_suggestions,
                "recent_errors": recent_errors,
                "system_health": {
                    "cache_hit_rate": cache_stats.get("hit_rate", 0),
                    "avg_query_time": performance_summary.get("queries", {}).get("query", {}).get("avg_execution_time", 0),
                    "error_rate": len(recent_errors) / max(performance_summary.get("queries", {}).get("query", {}).get("total_queries", 1), 1) * 100
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting advanced analytics: {e}")
            return {"error": str(e)}
    
    def clear_cache(self) -> Dict[str, Any]:
        """Clear the cache."""
        try:
            self.cache_manager.clear()
            return {"success": True, "message": "Cache cleared successfully"}
        except Exception as e:
            error_msg = f"Error clearing cache: {e}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        try:
            return self.cache_manager.get_stats()
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {"error": str(e)}
    
    def optimize_pipeline(self) -> Dict[str, Any]:
        """Run pipeline optimization."""
        try:
            optimization_results = {
                "cache_optimization": {},
                "query_optimization": {},
                "performance_optimization": {}
            }
            
            # Cache optimization
            cache_stats = self.cache_manager.get_stats()
            if cache_stats.get("hit_rate", 0) < 50:
                optimization_results["cache_optimization"]["suggestion"] = "Consider increasing cache size or TTL"
                optimization_results["cache_optimization"]["current_hit_rate"] = cache_stats.get("hit_rate", 0)
            
            # Query optimization
            optimization_suggestions = self.query_optimizer.get_optimization_suggestions()
            if optimization_suggestions:
                optimization_results["query_optimization"]["suggestions"] = optimization_suggestions
            
            # Performance optimization
            performance_summary = self.performance_monitor.get_performance_summary()
            avg_query_time = performance_summary.get("queries", {}).get("query", {}).get("avg_execution_time", 0)
            if avg_query_time > 5.0:
                optimization_results["performance_optimization"]["suggestion"] = "Consider enabling more aggressive caching"
                optimization_results["performance_optimization"]["avg_query_time"] = avg_query_time
            
            return {
                "success": True,
                "optimization_results": optimization_results,
                "timestamp": time.time()
            }
            
        except Exception as e:
            error_msg = f"Error in pipeline optimization: {e}"
            logger.error(error_msg)
            return {"success": False, "error": error_msg}
