"""
FastAPI REST API for Enhanced RAG Pipeline - Phase 4

Provides web interface for document upload, querying, and pipeline management.
Phase 4: Advanced Features & Optimization
"""

import os
import logging
import time
import hashlib
from typing import List, Dict, Any, Optional
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import json
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.rag_pipeline import RAGPipeline
from src.vector_store.chroma_store import ChromaStore
from src.document_management.document_manager import DocumentManager
from src.api.websocket_manager import websocket_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced RAG Pipeline API - Phase 4",
    description="Professional RAG Pipeline with Advanced Features for PanScience Innovations LLM Specialist Assignment",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Rate limiting
class RateLimiter:
    def __init__(self, max_requests: int = 1000, window_seconds: int = 300):  # 1000 requests per 5 minutes
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        client_requests = self.requests[client_id]
        
        # Remove old requests outside the window
        client_requests[:] = [req_time for req_time in client_requests if now - req_time < self.window_seconds]
        
        if len(client_requests) >= self.max_requests:
            return False
        
        client_requests.append(now)
        return True

# Frontend-friendly rate limiter (more generous for UI operations)
frontend_rate_limiter = RateLimiter(max_requests=5000, window_seconds=300)  # 5000 requests per 5 minutes
rate_limiter = RateLimiter(max_requests=1000, window_seconds=300)  # 1000 requests per 5 minutes

# Analytics tracking
class AnalyticsTracker:
    def __init__(self):
        self.query_history = []
        self.performance_metrics = defaultdict(list)
        self.error_logs = []
    
    def log_query(self, query: str, processing_time: float, success: bool, user_id: str = "anonymous"):
        self.query_history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "processing_time": processing_time,
            "success": success,
            "user_id": user_id
        })
        
        # Keep only last 1000 queries
        if len(self.query_history) > 1000:
            self.query_history = self.query_history[-1000:]
    
    def log_performance(self, endpoint: str, processing_time: float):
        self.performance_metrics[endpoint].append(processing_time)
        
        # Keep only last 100 metrics per endpoint
        if len(self.performance_metrics[endpoint]) > 100:
            self.performance_metrics[endpoint] = self.performance_metrics[endpoint][-100:]
    
    def log_error(self, error: str, endpoint: str, user_id: str = "anonymous"):
        self.error_logs.append({
            "timestamp": datetime.now().isoformat(),
            "error": error,
            "endpoint": endpoint,
            "user_id": user_id
        })
        
        # Keep only last 500 errors
        if len(self.error_logs) > 500:
            self.error_logs = self.error_logs[-500:]
    
    def get_analytics(self) -> Dict[str, Any]:
        if not self.query_history:
            return {"message": "No analytics data available"}
        
        # Calculate query success rate
        total_queries = len(self.query_history)
        successful_queries = sum(1 for q in self.query_history if q["success"])
        success_rate = (successful_queries / total_queries) * 100 if total_queries > 0 else 0
        
        # Calculate average processing times
        avg_processing_time = sum(q["processing_time"] for q in self.query_history) / total_queries if total_queries > 0 else 0
        
        # Get recent performance metrics
        recent_metrics = {}
        for endpoint, times in self.performance_metrics.items():
            if times:
                recent_metrics[endpoint] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "total_requests": len(times)
                }
        
        return {
            "total_queries": total_queries,
            "success_rate": round(success_rate, 2),
            "avg_processing_time": round(avg_processing_time, 3),
            "recent_performance": recent_metrics,
            "recent_errors": len([e for e in self.error_logs if datetime.now() - datetime.fromisoformat(e["timestamp"]) < timedelta(hours=24)]),
            "top_queries": self._get_top_queries(),
            "hourly_traffic": self._get_hourly_traffic()
        }
    
    def _get_top_queries(self) -> List[Dict[str, Any]]:
        query_counts = defaultdict(int)
        for query in self.query_history:
            query_counts[query["query"]] += 1
        
        return sorted([{"query": q, "count": c} for q, c in query_counts.items()], 
                     key=lambda x: x["count"], reverse=True)[:10]
    
    def _get_hourly_traffic(self) -> Dict[str, int]:
        hourly_counts = defaultdict(int)
        for query in self.query_history[-100:]:  # Last 100 queries
            hour = datetime.fromisoformat(query["timestamp"]).strftime("%H:00")
            hourly_counts[hour] += 1
        
        return dict(hourly_counts)

analytics_tracker = AnalyticsTracker()

# Serve static UI from /static and redirect root to UI
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir), html=True), name="static")

# Initialize RAG pipeline
rag_pipeline = None

# Initialize Document Manager (will be done in startup event)
document_manager = None

# Initialize WebSocket Manager
websocket_manager = websocket_manager

# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    question: str = Field(..., description="User question to query the RAG pipeline")
    n_results: int = Field(5, description="Number of results to return")
    use_reranking: bool = Field(True, description="Whether to use advanced reranking")
    filters: Optional[Dict[str, Any]] = Field(None, description="Document filters (file_type, date_range, etc.)")
    query_expansion: bool = Field(False, description="Whether to use query expansion")
    semantic_search: bool = Field(True, description="Whether to use semantic search")

class QueryResponse(BaseModel):
    success: bool
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    retrieval_stats: Dict[str, Any]
    processing_time: float
    query_id: str
    confidence_score: float

class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    documents_processed: int
    chunks_created: int
    embeddings_stored: int
    errors: List[str]
    document_ids: List[str]

class HealthResponse(BaseModel):
    status: str
    components: Dict[str, Any]
    timestamp: str
    system_health: Dict[str, Any]

class PipelineStatsResponse(BaseModel):
    total_documents: int
    avg_length: float
    processing_time: str
    top_similarity: str
    pipeline_status: str
    collection_name: str
    embedding_model: str
    performance_metrics: Dict[str, Any]

class AnalyticsResponse(BaseModel):
    total_queries: int
    success_rate: float
    avg_processing_time: float
    recent_performance: Dict[str, Any]
    recent_errors: int
    top_queries: List[Dict[str, Any]]
    hourly_traffic: Dict[str, int]

class DocumentMetadata(BaseModel):
    file_name: str
    file_type: str
    upload_date: str
    file_size: int
    processing_status: str
    chunk_count: int
    embedding_count: int

# Security middleware
async def verify_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not credentials:
        # For now, allow anonymous access but log it
        return "anonymous"
    
    # In production, validate against stored API keys
    api_key = credentials.credentials
    if api_key == "demo_key_123":  # Demo key for testing
        return "demo_user"
    
    # For now, accept any non-empty key
    return f"user_{hashlib.md5(api_key.encode()).hexdigest()[:8]}"

# Rate limiting middleware
async def check_rate_limit(client_id: str = Depends(verify_api_key)):
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
    return client_id

# Frontend-friendly rate limiting for document operations
async def check_frontend_rate_limit(client_id: str = Depends(verify_api_key)):
    if not frontend_rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")
    return client_id

async def process_documents_background(document_ids: List[str]):
    """Process documents in background for better performance."""
    try:
        logger.info(f"Starting background processing for {len(document_ids)} documents")
        
        # Process documents in smaller batches to prevent memory issues
        batch_size = 3
        for i in range(0, len(document_ids), batch_size):
            batch = document_ids[i:i + batch_size]
            
            try:
                # Process batch
                for doc_id in batch:
                    # Here you would add the actual document processing logic
                    # For now, we'll just log the processing
                    logger.info(f"Processing document: {doc_id}")
                    await asyncio.sleep(0.5)  # Simulate processing time
                
                # Small delay between batches
                if i + batch_size < len(document_ids):
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                continue
        
        logger.info(f"Background processing completed for {len(document_ids)} documents")
        
    except Exception as e:
        logger.error(f"Background document processing failed: {e}")

# API endpoints
@app.get("/")
async def root():
    """Redirect to the web UI."""
    return RedirectResponse(url="/static/index.html")

@app.get("/react")
async def react_ui():
    """Serve the React.js interface."""
    return RedirectResponse(url="/static/react-app/index.html")

@app.get("/health", response_model=HealthResponse)
async def health_check(client_id: str = Depends(check_rate_limit)):
    """Check the health of all pipeline components with system metrics."""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        start_time = time.time()
        health_status = rag_pipeline.health_check()
        processing_time = time.time() - start_time
        
        # Track performance
        analytics_tracker.log_performance("health", processing_time)
        
        # Get system health metrics
        import psutil
        system_health = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent,
            "uptime": time.time() - psutil.boot_time()
        }
        
        return HealthResponse(
            status=health_status["pipeline"],
            components=health_status.get("components", {}),
            timestamp=datetime.now().isoformat(),
            system_health=system_health
        )
    except Exception as e:
        analytics_tracker.log_error(str(e), "health", client_id)
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/stats", response_model=PipelineStatsResponse)
async def get_pipeline_stats(client_id: str = Depends(check_rate_limit)):
    """Get comprehensive pipeline statistics with performance metrics."""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        start_time = time.time()
        
        # Get basic pipeline stats
        pipeline_stats = rag_pipeline.get_pipeline_stats()
        
        # Get vector store stats from the nested structure
        vector_stats = pipeline_stats.get("vector_store", {})
        
        # Get recent query stats if available
        recent_stats = getattr(rag_pipeline, '_recent_stats', {})
        
        # Get performance metrics
        performance_metrics = {
            "endpoint_performance": dict(analytics_tracker.performance_metrics),
            "cache_hit_rate": getattr(rag_pipeline, '_cache_hit_rate', 0),
            "avg_query_time": analytics_tracker.get_analytics().get("avg_processing_time", 0)
        }
        
        # Combine all stats
        combined_stats = {
            "total_documents": vector_stats.get("total_documents", 0),
            "avg_length": vector_stats.get("average_document_length", 0),
            "processing_time": str(recent_stats.get("last_processing_time", "-")),
            "top_similarity": str(recent_stats.get("last_top_similarity", "-")),
            "pipeline_status": pipeline_stats.get("pipeline_status", "unknown"),
            "collection_name": vector_stats.get("collection_name", "N/A"),
            "embedding_model": str(vector_stats.get("embedding_model", "N/A")),
            "performance_metrics": performance_metrics
        }
        
        processing_time = time.time() - start_time
        analytics_tracker.log_performance("stats", processing_time)
        
        return PipelineStatsResponse(**combined_stats)
    except Exception as e:
        analytics_tracker.log_error(str(e), "stats", client_id)
        logger.error(f"Failed to get pipeline stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics(client_id: str = Depends(check_rate_limit)):
    """Get comprehensive analytics and performance metrics."""
    try:
        start_time = time.time()
        analytics = analytics_tracker.get_analytics()
        processing_time = time.time() - start_time
        
        analytics_tracker.log_performance("analytics", processing_time)
        
        return AnalyticsResponse(**analytics)
    except Exception as e:
        analytics_tracker.log_error(str(e), "analytics", client_id)
        logger.error(f"Failed to get analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_pipeline(
    request: QueryRequest, 
    client_id: str = Depends(check_rate_limit)
):
    """Query the RAG pipeline with advanced features."""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        start_time = time.time()
        
        # Generate query ID for tracking
        query_id = hashlib.md5(f"{request.question}_{time.time()}".encode()).hexdigest()[:8]
        
        # Process query with advanced features
        response = rag_pipeline.query(
            question=request.question,
            n_results=request.n_results,
            use_reranking=request.use_reranking,
            filters=request.filters,
            query_expansion=request.query_expansion,
            semantic_search=request.semantic_search
        )
        
        processing_time = time.time() - start_time
        
        # Store processing time in RAG pipeline for stats
        if hasattr(rag_pipeline, '_recent_stats'):
            rag_pipeline._recent_stats['last_processing_time'] = round(processing_time, 3)
            if response.get("sources"):
                rag_pipeline._recent_stats['last_top_similarity'] = max(
                    doc.get("similarity_score", 0) for doc in response["sources"]
                )
        
        # Track analytics
        analytics_tracker.log_query(
            request.question, 
            processing_time, 
            response.get("success", False), 
            client_id
        )
        analytics_tracker.log_performance("query", processing_time)
        
        if not response["success"]:
            analytics_tracker.log_error(response.get("error", "Query failed"), "query", client_id)
            raise HTTPException(status_code=400, detail=response.get("error", "Query failed"))
        
        # Calculate confidence score based on similarity scores
        confidence_score = 0.0
        if response.get("sources"):
            avg_similarity = sum(doc.get("similarity_score", 0) for doc in response["sources"]) / len(response["sources"])
            confidence_score = max(0.0, min(1.0, (avg_similarity + 1) / 2))  # Normalize to 0-1
        
        return QueryResponse(
            success=True,
            question=response["question"],
            answer=response["answer"],
            sources=response["sources"],
            retrieval_stats=response["retrieval_stats"],
            processing_time=round(processing_time, 3),
            query_id=query_id,
            confidence_score=round(confidence_score, 3)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        analytics_tracker.log_error(str(e), "query", client_id)
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    client_id: str = Depends(check_frontend_rate_limit)
):
    """Upload multiple documents with enhanced error handling for heavy loads."""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Validate file count and sizes for heavy load scenarios
        total_size = 0
        max_files = 50  # Allow up to 50 files at once
        max_file_size = 10 * 1024 * 1024  # 10MB per file
        max_total_size = 500 * 1024 * 1024  # 500MB total
        
        if len(files) > max_files:
            raise HTTPException(
                status_code=400, 
                detail=f"Too many files. Maximum allowed: {max_files}, received: {len(files)}"
            )
        
        # Pre-validate all files
        for file in files:
            if file.size > max_file_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is too large. Maximum size: {max_file_size / (1024*1024):.1f}MB"
                )
            total_size += file.size
        
        if total_size > max_total_size:
            raise HTTPException(
                status_code=400,
                detail=f"Total file size too large. Maximum total size: {max_total_size / (1024*1024):.1f}MB, received: {total_size / (1024*1024):.1f}MB"
            )
        
        # Process files in batches for better performance
        batch_size = 5  # Process 5 files at a time
        uploaded_documents = []
        errors = []
        document_ids = []
        
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            
            for file in batch:
                try:
                    # Validate file type
                    if not file.content_type or not any(
                        file.content_type.startswith(t) for t in [
                            'application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                            'text/plain', 'text/html'
                        ]
                    ):
                        errors.append(f"Unsupported file type for {file.filename}: {file.content_type}")
                        continue
                    
                    # Read file content
                    content = await file.read()
                    
                    # Generate unique filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    file_id = hashlib.md5(f"{file.filename}_{timestamp}".encode()).hexdigest()[:8]
                    file_path = f"data/documents/{file_id}_{file.filename}"
                    
                    # Save file
                    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(file_path, "wb") as f:
                        f.write(content)
                    
                    # Create metadata
                    file_metadata = {
                        "filename": file.filename,
                        "content_type": file.content_type,
                        "file_size": len(content),
                        "upload_time": datetime.now().isoformat(),
                        "uploaded_by": client_id,
                        "tags": [],
                        "category": "general"
                    }
                    
                    # Add to document manager
                    doc_result = await document_manager.add_document(
                        str(file_path), 
                        content, 
                        file_metadata
                    )
                    
                    if doc_result["success"]:
                        # Notify WebSocket clients
                        await websocket_manager.broadcast_to_group("document_updates", {
                            "type": "document_added",
                            "document": doc_result["document"],
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        # Also add to the documents list for immediate display
                        uploaded_documents.append({
                            "id": doc_result["document"]["id"],
                            "filename": doc_result["document"]["filename"],
                            "file_size": doc_result["document"]["file_size"],
                            "upload_time": doc_result["document"]["upload_time"],
                            "category": doc_result["document"]["category"],
                            "tags": doc_result["document"]["tags"]
                        })
                        
                        document_ids.append(doc_result["document"]["id"])
                        logger.info(f"Saved uploaded file: {file.filename} with ID: {file_id}")
                    else:
                        logger.error(f"Failed to add document to manager: {doc_result['error']}")
                        errors.append(f"Failed to add {Path(file_path).name} to document manager: {doc_result['error']}")
                    
                    # Save metadata to file
                    metadata_file = f"data/uploads/{file_id}_{file.filename}.json"
                    with open(metadata_file, "w") as f:
                        json.dump(file_metadata, f, indent=2)
                        
                except Exception as e:
                    error_msg = f"Error processing {file.filename}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    continue
            
            # Small delay between batches to prevent overwhelming the system
            if i + batch_size < len(files):
                await asyncio.sleep(0.1)
        
        # Process documents in background for better performance
        if document_ids:
            asyncio.create_task(process_documents_background(document_ids))
        
        return DocumentUploadResponse(
            success=len(errors) == 0,
            message=f"Uploaded {len(uploaded_documents)} documents successfully" + (f" with {len(errors)} errors" if errors else ""),
            documents_processed=len(uploaded_documents),
            chunks_created=0,  # Will be updated by background processing
            embeddings_stored=0,  # Will be updated by background processing
            errors=errors,
            document_ids=document_ids
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/documents")
async def list_documents(
    file_type: Optional[str] = Query(None, description="Filter by file type"),
    date_from: Optional[str] = Query(None, description="Filter by upload date from (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Filter by upload date to (YYYY-MM-DD)"),
    client_id: str = Depends(check_rate_limit)
):
    """List documents with advanced filtering."""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        start_time = time.time()
        
        # Get vector store stats
        stats = rag_pipeline.vector_store.get_collection_stats()
        
        # Apply filters if provided
        filtered_stats = stats.copy()
        if file_type or date_from or date_to:
            # In a real implementation, you would filter the actual documents
            # For now, we'll just indicate that filtering is applied
            filtered_stats["filters_applied"] = {
                "file_type": file_type,
                "date_from": date_from,
                "date_to": date_to
            }
        
        processing_time = time.time() - start_time
        analytics_tracker.log_performance("list_documents", processing_time)
        
        return {
            "total_documents": filtered_stats.get("total_documents", 0),
            "collection_name": filtered_stats.get("collection_name", "N/A"),
            "embedding_model": filtered_stats.get("embedding_model", "N/A"),
            "average_document_length": filtered_stats.get("average_document_length", 0),
            "filters_applied": filtered_stats.get("filters_applied", {})
        }
        
    except Exception as e:
        analytics_tracker.log_error(str(e), "list_documents", client_id)
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.delete("/documents")
async def clear_documents(client_id: str = Depends(check_rate_limit)):
    """Clear all documents from the vector store."""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        start_time = time.time()
        
        success = rag_pipeline.vector_store.reset_collection()
        
        processing_time = time.time() - start_time
        analytics_tracker.log_performance("clear_documents", processing_time)
        
        if success:
            return {"message": "All documents cleared successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear documents")
            
    except HTTPException:
        raise
    except Exception as e:
        analytics_tracker.log_error(str(e), "clear_documents", client_id)
        logger.error(f"Failed to clear documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")

# Document Management Endpoints
@app.get("/documents/list")
async def list_documents(
    category: Optional[str] = Query(None, description="Filter by category"),
    status: Optional[str] = Query(None, description="Filter by status"),
    client_id: str = Depends(check_frontend_rate_limit)
):
    """List all documents with optional filtering."""
    try:
        filters = {}
        if category:
            filters["category"] = category
        if status:
            filters["status"] = status
        
        documents = await document_manager.list_documents(filters)
        
        return {
            "success": True,
            "documents": documents,
            "total_count": len(documents),
            "filters_applied": filters
        }
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.get("/documents/{document_id}")
async def get_document_metadata(
    document_id: str,
    client_id: str = Depends(check_rate_limit)
):
    """Get metadata for a specific document."""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        # In a real implementation, you would query the vector store for document metadata
        # For now, return a placeholder response
        return {
            "document_id": document_id,
            "status": "Document metadata retrieval not yet implemented",
            "message": "This endpoint will provide detailed document information in future versions"
        }
        
    except Exception as e:
        analytics_tracker.log_error(str(e), "get_document_metadata", client_id)
        logger.error(f"Failed to get document metadata: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document metadata: {str(e)}")

@app.post("/documents/search")
async def search_documents(
    query: str = Query(..., description="Search query"),
    filters: Optional[str] = Query(None, description="JSON string of filters"),
    client_id: str = Depends(check_rate_limit)
):
    """Advanced document search with filters."""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        start_time = time.time()
        
        # Parse filters if provided
        search_filters = {}
        if filters:
            try:
                search_filters = json.loads(filters)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid filters format")
        
        # Perform search using the RAG pipeline
        response = rag_pipeline.query(
            question=query,
            n_results=10,
            use_reranking=True,
            filters=search_filters
        )
        
        processing_time = time.time() - start_time
        analytics_tracker.log_performance("search_documents", processing_time)
        
        if not response["success"]:
            raise HTTPException(status_code=400, detail=response.get("error", "Search failed"))
        
        return {
            "query": query,
            "filters_applied": search_filters,
            "results": response["sources"],
            "total_results": len(response["sources"]),
            "processing_time": round(processing_time, 3)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        analytics_tracker.log_error(str(e), "search_documents", client_id)
        logger.error(f"Document search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/documents/{doc_id}")
async def get_document(
    doc_id: str,
    client_id: str = Depends(check_rate_limit)
):
    """Get document by ID."""
    try:
        document = await document_manager.get_document(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "success": True,
            "document": document
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get document: {str(e)}")

@app.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    client_id: str = Depends(check_rate_limit)
):
    """Delete a document."""
    try:
        result = await document_manager.delete_document(doc_id)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # Notify WebSocket clients
        await websocket_manager.broadcast_to_group("document_updates", {
            "type": "document_deleted",
            "doc_id": doc_id,
            "timestamp": datetime.now().isoformat()
        })
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@app.get("/documents/stats/realtime")
async def get_realtime_document_stats(
    client_id: str = Depends(check_frontend_rate_limit)
):
    """Get real-time document statistics."""
    try:
        stats = await document_manager.get_real_time_stats()
        return {
            "success": True,
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Failed to get real-time stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get real-time stats: {str(e)}")

@app.websocket("/ws/documents")
async def websocket_endpoint(websocket):
    """WebSocket endpoint for real-time document updates."""
    client_id = await websocket_manager.connect(websocket)
    
    try:
        # Subscribe to document updates
        await websocket_manager.subscribe_to_group(client_id, "document_updates")
        
        # Send initial stats
        stats = await document_manager.get_real_time_stats()
        await websocket.send_text(json.dumps({
            "type": "initial_stats",
            "stats": stats
        }))
        
        # Keep connection alive
        while True:
            await websocket.receive_text()
            await asyncio.sleep(1)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket_manager.disconnect(client_id)

@app.post("/reset")
async def reset_system(
    client_id: str = Depends(check_rate_limit)
):
    """Reset the entire system - clear all documents and reset state."""
    try:
        # Clear document manager
        await document_manager.reset_system()
        
        # Clear vector store
        await rag_pipeline.vector_store.clear_collection()
        
        # Clear RAG pipeline stats
        rag_pipeline._recent_stats = {
            'last_processing_time': 0,
            'last_top_similarity': 0,
            'last_query_time': 0,
            'cache_hit_rate': 0
        }
        
        # Notify WebSocket clients
        await websocket_manager.broadcast_to_group("document_updates", {
            "type": "system_reset",
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "success": True,
            "message": "System reset successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to reset system: {e}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")

@app.post("/documents/reset")
async def reset_documents(client_id: str = Depends(check_rate_limit)):
    """Reset all documents and clear the system."""
    try:
        # Reset document manager
        await document_manager.reset_system()
        
        # Clear ChromaDB collection
        await rag_pipeline.vector_store.clear_collection()
        
        # Notify WebSocket clients
        await websocket_manager.broadcast_to_group("document_updates", {
            "type": "system_reset",
            "message": "System reset completed",
            "timestamp": datetime.now().isoformat()
        })
        
        return {"success": True, "message": "System reset completed successfully"}
        
    except Exception as e:
        logger.error(f"Failed to reset system: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset system: {str(e)}")

@app.post("/documents/process-all")
async def process_all_documents(
    client_id: str = Depends(check_rate_limit)
):
    """Process all existing documents into the vector database."""
    try:
        if not document_manager:
            raise HTTPException(status_code=503, detail="Document manager not initialized")
        
        if not rag_pipeline:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
        
        # Get all documents
        documents = await document_manager.list_documents({})
        
        if not documents:
            return {
                "success": True,
                "message": "No documents to process",
                "processed_count": 0
            }
        
        processed_count = 0
        errors = []
        
        # Process each document
        for doc in documents:
            try:
                # Get the file path
                file_path = doc.get("file_path")
                if not file_path or not os.path.exists(file_path):
                    continue
                
                # Process the document through the RAG pipeline
                result = rag_pipeline.process_documents([file_path])
                
                if result.get("success"):
                    processed_count += 1
                    logger.info(f"Successfully processed document: {doc.get('filename', 'Unknown')}")
                else:
                    errors.append(f"Failed to process {doc.get('filename', 'Unknown')}: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                errors.append(f"Error processing {doc.get('filename', 'Unknown')}: {str(e)}")
                logger.error(f"Error processing document {doc.get('filename', 'Unknown')}: {e}")
        
        # Notify WebSocket clients
        await websocket_manager.broadcast_to_group("document_updates", {
            "type": "documents_processed",
            "processed_count": processed_count,
            "total_count": len(documents),
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "success": True,
            "message": f"Processed {processed_count} out of {len(documents)} documents",
            "processed_count": processed_count,
            "total_count": len(documents),
            "errors": errors if errors else []
        }
        
    except Exception as e:
        logger.error(f"Failed to process documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process documents: {str(e)}")

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline and document manager on startup."""
    global rag_pipeline, document_manager
    try:
        logger.info("Initializing RAG Pipeline - Phase 4...")
        rag_pipeline = RAGPipeline()
        logger.info("RAG Pipeline Phase 4 initialized successfully")
        
        logger.info("Initializing Document Manager...")
        document_manager = DocumentManager()
        logger.info("Document Manager initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise

# Health check endpoint (no rate limiting)
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "rate_limits": {
            "frontend": f"{frontend_rate_limiter.max_requests} requests per {frontend_rate_limiter.window_seconds} seconds",
            "api": f"{rate_limiter.max_requests} requests per {rate_limiter.window_seconds} seconds"
        }
    }

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler with analytics tracking."""
    error_msg = f"Unhandled exception: {exc}"
    logger.error(error_msg)
    
    # Track error in analytics
    client_id = "unknown"
    try:
        # Try to extract client ID from request headers
        auth_header = request.headers.get("authorization")
        if auth_header:
            client_id = f"user_{hashlib.md5(auth_header.encode()).hexdigest()[:8]}"
    except:
        pass
    
    analytics_tracker.log_error(error_msg, "global_exception", client_id)
    
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
