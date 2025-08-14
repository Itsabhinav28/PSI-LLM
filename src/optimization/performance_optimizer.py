"""
Performance Optimizer for Phase 4
"""

import logging
import time
import hashlib
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
import threading
import queue

logger = logging.getLogger(__name__)

class CacheManager:
    """Advanced caching with LRU, TTL, and intelligent eviction."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = OrderedDict()
        self.timestamps = {}
        self.access_counts = {}
        self.lock = threading.Lock()
        
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with TTL check."""
        with self.lock:
            if key in self.cache:
                if self._is_expired(key):
                    self._remove(key)
                    self.stats["misses"] += 1
                    return None
                
                self.access_counts[key] += 1
                self.cache.move_to_end(key)
                self.stats["hits"] += 1
                return self.cache[key]
            
            self.stats["misses"] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL."""
        with self.lock:
            if key in self.cache:
                self._remove(key)
            
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = value
            self.timestamps[key] = time.time()
            self.access_counts[key] = 0
            self.stats["size"] = len(self.cache)
    
    def _is_expired(self, key: str) -> bool:
        """Check if a cache entry has expired."""
        if key not in self.timestamps:
            return True
        return time.time() - self.timestamps[key] > self.default_ttl
    
    def _remove(self, key: str) -> None:
        """Remove a key from cache."""
        if key in self.cache:
            del self.cache[key]
            del self.timestamps[key]
            del self.access_counts[key]
            self.stats["size"] = len(self.cache)
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if self.cache:
            key = next(iter(self.cache))
            self._remove(key)
            self.stats["evictions"] += 1
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.timestamps.clear()
            self.access_counts.clear()
            self.stats["size"] = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "size": self.stats["size"],
                "max_size": self.max_size,
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "evictions": self.stats["evictions"],
                "hit_rate": round(hit_rate, 2),
                "utilization": round((self.stats["size"] / self.max_size) * 100, 2)
            }

class BatchProcessor:
    """Batch processing for multiple operations."""
    
    def __init__(self, batch_size: int = 10, max_wait_time: float = 1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.queue = queue.Queue()
        self.processing = False
        self.lock = threading.Lock()
        
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
    
    def add_task(self, task: Dict[str, Any]) -> None:
        """Add a task to the batch queue."""
        self.queue.put(task)
    
    def _worker(self) -> None:
        """Background worker that processes batches."""
        while True:
            try:
                batch = []
                start_time = time.time()
                
                while len(batch) < self.batch_size and (time.time() - start_time) < self.max_wait_time:
                    try:
                        task = self.queue.get(timeout=0.1)
                        batch.append(task)
                    except queue.Empty:
                        break
                
                if batch:
                    self._process_batch(batch)
                    
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
    
    def _process_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Process a batch of tasks."""
        try:
            tasks_by_type = {}
            for task in batch:
                task_type = task.get("type", "unknown")
                if task_type not in tasks_by_type:
                    tasks_by_type[task_type] = []
                tasks_by_type[task_type].append(task)
            
            for task_type, tasks in tasks_by_type.items():
                if task_type == "embedding":
                    self._process_embedding_batch(tasks)
                elif task_type == "query":
                    self._process_query_batch(tasks)
                elif task_type == "document":
                    self._process_document_batch(tasks)
                else:
                    logger.warning(f"Unknown task type: {task_type}")
                    
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
    
    def _process_embedding_batch(self, tasks: List[Dict[str, Any]]) -> None:
        logger.info(f"Processing {len(tasks)} embedding tasks in batch")
        for task in tasks:
            try:
                time.sleep(0.01)
                logger.debug(f"Processed embedding task: {task.get('id', 'unknown')}")
            except Exception as e:
                logger.error(f"Error processing embedding task: {e}")
    
    def _process_query_batch(self, tasks: List[Dict[str, Any]]) -> None:
        logger.info(f"Processing {len(tasks)} query tasks in batch")
        for task in tasks:
            try:
                time.sleep(0.01)
                logger.debug(f"Processed query task: {task.get('id', 'unknown')}")
            except Exception as e:
                logger.error(f"Error processing query task: {e}")
    
    def _process_document_batch(self, tasks: List[Dict[str, Any]]) -> None:
        logger.info(f"Processing {len(tasks)} document tasks in batch")
        for task in tasks:
            try:
                time.sleep(0.01)
                logger.debug(f"Processed document task: {task.get('id', 'unknown')}")
            except Exception as e:
                logger.error(f"Error processing document task: {e}")

class QueryOptimizer:
    """Query optimization and execution planning."""
    
    def __init__(self):
        self.query_plans = {}
        self.performance_history = {}
        self.optimization_rules = self._load_optimization_rules()
    
    def _load_optimization_rules(self) -> Dict[str, Any]:
        """Load query optimization rules."""
        return {
            "vector_search": {
                "max_results": 50,
                "similarity_threshold": 0.3,
                "use_reranking": True,
                "batch_size": 10
            },
            "keyword_search": {
                "max_results": 100,
                "use_fuzzy": True,
                "boost_exact_matches": True
            },
            "hybrid_search": {
                "vector_weight": 0.7,
                "keyword_weight": 0.3,
                "fusion_method": "reciprocal_rank"
            }
        }
    
    def optimize_query(self, query: str, query_type: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize a query based on type and filters."""
        try:
            plan = self._generate_query_plan(query, query_type, filters)
            optimized_plan = self._apply_optimization_rules(plan)
            
            plan_hash = hashlib.md5(f"{query}_{query_type}".encode()).hexdigest()[:8]
            self.query_plans[plan_hash] = optimized_plan
            
            return optimized_plan
            
        except Exception as e:
            logger.error(f"Error optimizing query: {e}")
            return {"error": str(e)}
    
    def _generate_query_plan(self, query: str, query_type: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a query execution plan."""
        plan = {
            "query": query,
            "type": query_type,
            "filters": filters or {},
            "execution_steps": [],
            "estimated_cost": 0,
            "optimization_hints": []
        }
        
        if query_type == "vector_search":
            plan["execution_steps"] = [
                {"step": "preprocess", "description": "Clean and normalize query"},
                {"step": "embedding", "description": "Generate query embedding"},
                {"step": "vector_search", "description": "Search vector database"},
                {"step": "reranking", "description": "Rerank results if enabled"}
            ]
            plan["estimated_cost"] = 10
        
        elif query_type == "keyword_search":
            plan["execution_steps"] = [
                {"step": "tokenization", "description": "Tokenize query"},
                {"step": "keyword_extraction", "description": "Extract keywords"},
                {"step": "index_search", "description": "Search inverted index"},
                {"step": "result_merging", "description": "Merge and sort results"}
            ]
            plan["estimated_cost"] = 5
        
        elif query_type == "hybrid_search":
            plan["execution_steps"] = [
                {"step": "parallel_search", "description": "Execute vector and keyword search in parallel"},
                {"step": "result_fusion", "description": "Fuse results using hybrid method"},
                {"step": "reranking", "description": "Final reranking of fused results"}
            ]
            plan["estimated_cost"] = 15
        
        if filters and len(filters) > 2:
            plan["optimization_hints"].append("Consider reducing filter complexity for better performance")
        
        if len(query.split()) > 10:
            plan["optimization_hints"].append("Long queries may benefit from query expansion")
        
        return plan
    
    def _apply_optimization_rules(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimization rules to the query plan."""
        optimized_plan = plan.copy()
        
        query_type = plan["type"]
        if query_type in self.optimization_rules:
            rules = self.optimization_rules[query_type]
            optimized_plan["parameters"] = rules
        
        optimized_plan["performance_optimizations"] = []
        
        if plan["estimated_cost"] > 10:
            optimized_plan["performance_optimizations"].append("Enable result caching")
        
        if "batch_size" in optimized_plan.get("parameters", {}):
            optimized_plan["performance_optimizations"].append("Use batch processing for multiple queries")
        
        if len(plan["execution_steps"]) > 3:
            optimized_plan["performance_optimizations"].append("Consider parallel execution of independent steps")
        
        return optimized_plan
    
    def record_performance(self, query_hash: str, execution_time: float, success: bool) -> None:
        """Record query performance for optimization."""
        if query_hash not in self.performance_history:
            self.performance_history[query_hash] = []
        
        self.performance_history[query_hash].append({
            "execution_time": execution_time,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.performance_history[query_hash]) > 10:
            self.performance_history[query_hash] = self.performance_history[query_hash][-10:]
    
    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Get optimization suggestions based on performance data."""
        suggestions = []
        
        for query_hash, performance_records in self.performance_history.items():
            if len(performance_records) < 3:
                continue
            
            avg_time = sum(record["execution_time"] for record in performance_records) / len(performance_records)
            
            if avg_time > 5.0:
                suggestions.append({
                    "query_hash": query_hash,
                    "issue": "Slow execution time",
                    "suggestion": "Consider enabling caching or reducing query complexity",
                    "avg_time": round(avg_time, 2),
                    "priority": "high"
                })
            
            failed_count = sum(1 for record in performance_records if not record["success"])
            if failed_count > len(performance_records) * 0.3:
                suggestions.append({
                    "query_hash": query_hash,
                    "issue": "High failure rate",
                    "suggestion": "Review query parameters and error handling",
                    "failure_rate": round(failed_count / len(performance_records) * 100, 1),
                    "priority": "high"
                })
        
        return suggestions

class PerformanceMonitor:
    """Performance monitoring and metrics collection."""
    
    def __init__(self):
        self.metrics = {
            "endpoints": defaultdict(list),
            "queries": defaultdict(list),
            "system": defaultdict(list),
            "errors": []
        }
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def record_endpoint_metric(self, endpoint: str, method: str, response_time: float, status_code: int) -> None:
        """Record endpoint performance metrics."""
        with self.lock:
            metric = {
                "timestamp": datetime.now().isoformat(),
                "method": method,
                "response_time": response_time,
                "status_code": status_code
            }
            self.metrics["endpoints"][endpoint].append(metric)
            
            if len(self.metrics["endpoints"][endpoint]) > 100:
                self.metrics["endpoints"][endpoint] = self.metrics["endpoints"][endpoint][-100:]
    
    def record_query_metric(self, query_type: str, execution_time: float, success: bool, result_count: int) -> None:
        """Record query performance metrics."""
        with self.lock:
            metric = {
                "timestamp": datetime.now().isoformat(),
                "execution_time": execution_time,
                "success": success,
                "result_count": result_count
            }
            self.metrics["queries"][query_type].append(metric)
            
            if len(self.metrics["queries"][query_type]) > 100:
                self.metrics["queries"][query_type] = self.metrics["queries"][query_type][-100:]
    
    def record_system_metric(self, metric_name: str, value: float) -> None:
        """Record system performance metrics."""
        with self.lock:
            metric = {
                "timestamp": datetime.now().isoformat(),
                "value": value
            }
            self.metrics["system"][metric_name].append(metric)
            
            if len(self.metrics["system"][metric_name]) > 100:
                self.metrics["system"][metric_name] = self.metrics["system"][metric_name][-100:]
    
    def record_error(self, error: str, context: str, severity: str = "medium") -> None:
        """Record error information."""
        with self.lock:
            error_record = {
                "timestamp": datetime.now().isoformat(),
                "error": error,
                "context": context,
                "severity": severity
            }
            self.metrics["errors"].append(error_record)
            
            if len(self.metrics["errors"]) > 100:
                self.metrics["errors"] = self.metrics["errors"][-100:]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        with self.lock:
            summary = {
                "uptime": time.time() - self.start_time,
                "endpoints": {},
                "queries": {},
                "system": {},
                "errors": len(self.metrics["errors"])
            }
            
            for endpoint, metrics in self.metrics["endpoints"].items():
                if metrics:
                    response_times = [m["response_time"] for m in metrics]
                    status_codes = [m["status_code"] for m in metrics]
                    
                    summary["endpoints"][endpoint] = {
                        "total_requests": len(metrics),
                        "avg_response_time": round(sum(response_times) / len(response_times), 3),
                        "min_response_time": min(response_times),
                        "max_response_time": max(response_times),
                        "success_rate": round(sum(1 for s in status_codes if s < 400) / len(status_codes) * 100, 1)
                    }
            
            for query_type, metrics in self.metrics["queries"].items():
                if metrics:
                    execution_times = [m["execution_time"] for m in metrics]
                    success_count = sum(1 for m in metrics if m["success"])
                    
                    summary["queries"][query_type] = {
                        "total_queries": len(metrics),
                        "avg_execution_time": round(sum(execution_times) / len(execution_times), 3),
                        "success_rate": round(success_count / len(metrics) * 100, 1)
                    }
            
            for metric_name, metrics in self.metrics["system"].items():
                if metrics:
                    values = [m["value"] for m in metrics]
                    summary["system"][metric_name] = {
                        "current": values[-1] if values else 0,
                        "avg": round(sum(values) / len(values), 3),
                        "min": min(values) if values else 0,
                        "max": max(values) if values else 0
                    }
            
            return summary
    
    def get_recent_errors(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent errors within specified hours."""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_errors = [
                error for error in self.metrics["errors"]
                if datetime.fromisoformat(error["timestamp"]) > cutoff_time
            ]
            return recent_errors
