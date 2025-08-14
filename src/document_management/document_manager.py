"""
Real-Time Document Management System
Provides live document tracking, CRUD operations, and real-time statistics
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import threading
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)

class DocumentManager:
    """Real-time document management with live updates and CRUD operations."""
    
    def __init__(self, data_dir: str = "data/uploads"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Document storage
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.document_hashes: Dict[str, str] = {}
        self.document_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Real-time tracking
        self._subscribers: Set[str] = set()
        self._update_queue = asyncio.Queue()
        self._stats_cache = {}
        self._last_update = time.time()
        
        # Performance tracking
        self.upload_history: List[Dict[str, Any]] = []
        self.delete_history: List[Dict[str, Any]] = []
        self.edit_history: List[Dict[str, Any]] = []
        
        # Load existing documents from data directory
        self._load_existing_documents()
        
        # Start background tasks
        self._running = True
        self._background_task = asyncio.create_task(self._background_updater())
        
        logger.info(f"Document Manager initialized with real-time capabilities. Found {len(self.documents)} existing documents.")
    
    def _load_existing_documents(self):
        """Load existing documents from the data directory."""
        try:
            for file_path in self.data_dir.glob("*"):
                if file_path.is_file() and not file_path.name.endswith('.json'):
                    # Try to extract document info from filename
                    # Format: {doc_id}_{filename}
                    if '_' in file_path.name:
                        parts = file_path.name.split('_', 1)
                        if len(parts) == 2:
                            doc_id = parts[0]
                            filename = parts[1]
                            
                            # Create basic document record
                            document = {
                                "id": doc_id,
                                "filename": filename,
                                "file_path": str(file_path),
                                "file_size": file_path.stat().st_size,
                                "upload_time": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                                "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                                "status": "active",
                                "metadata": {},
                                "tags": [],
                                "category": "general",
                                "version": 1
                            }
                            
                            self.documents[doc_id] = document
                            logger.info(f"Loaded existing document: {filename}")
        except Exception as e:
            logger.error(f"Error loading existing documents: {e}")
    
    async def add_document(self, file_path: str, file_content: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Add a new document with real-time tracking."""
        try:
            # Generate document ID and hash
            doc_id = self._generate_doc_id(file_path, file_content)
            file_hash = hashlib.md5(file_content).hexdigest()
            
            # Check if document already exists
            if doc_id in self.documents:
                return await self.update_document(doc_id, file_content, metadata)
            
            # Create document record
            document = {
                "id": doc_id,
                "filename": Path(file_path).name,
                "file_path": str(self.data_dir / f"{doc_id}_{Path(file_path).name}"),
                "file_size": len(file_content),
                "file_hash": file_hash,
                "upload_time": datetime.now().isoformat(),
                "last_modified": datetime.now().isoformat(),
                "status": "active",
                "metadata": metadata,
                "tags": metadata.get("tags", []),
                "category": metadata.get("category", "general"),
                "version": 1
            }
            
            # Save file
            file_save_path = self.data_dir / f"{doc_id}_{Path(file_path).name}"
            file_save_path.write_bytes(file_content)
            
            # Store document info
            self.documents[doc_id] = document
            self.document_hashes[doc_id] = file_hash
            self.document_metadata[doc_id] = metadata
            
            # Record upload
            upload_record = {
                "timestamp": datetime.now().isoformat(),
                "doc_id": doc_id,
                "filename": document["filename"],
                "file_size": document["file_size"],
                "action": "upload"
            }
            self.upload_history.append(upload_record)
            
            # Trigger real-time update
            await self._notify_subscribers("document_added", document)
            
            logger.info(f"Document added: {doc_id} - {document['filename']}")
            return {"success": True, "document": document, "message": "Document added successfully"}
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return {"success": False, "error": str(e)}
    
    async def update_document(self, doc_id: str, file_content: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing document."""
        if doc_id not in self.documents:
            return {"success": False, "error": "Document not found"}
        
        try:
            old_document = self.documents[doc_id].copy()
            
            # Update document
            self.documents[doc_id].update({
                "file_size": len(file_content),
                "file_hash": hashlib.md5(file_content).hexdigest(),
                "last_modified": datetime.now().isoformat(),
                "version": self.documents[doc_id].get("version", 1) + 1,
                "metadata": metadata
            })
            
            # Save updated file
            file_save_path = Path(self.documents[doc_id]["file_path"])
            file_save_path.write_bytes(file_content)
            
            # Record edit
            edit_record = {
                "timestamp": datetime.now().isoformat(),
                "doc_id": doc_id,
                "filename": self.documents[doc_id]["filename"],
                "old_size": old_document["file_size"],
                "new_size": len(file_content),
                "action": "edit"
            }
            self.edit_history.append(edit_record)
            
            # Trigger real-time update
            await self._notify_subscribers("document_updated", self.documents[doc_id])
            
            logger.info(f"Document updated: {doc_id} - {self.documents[doc_id]['filename']}")
            return {"success": True, "document": self.documents[doc_id], "message": "Document updated successfully"}
            
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return {"success": False, "error": str(e)}
    
    async def delete_document(self, doc_id: str) -> Dict[str, Any]:
        """Delete a document with real-time tracking."""
        if doc_id not in self.documents:
            return {"success": False, "error": "Document not found"}
        
        try:
            document = self.documents[doc_id]
            
            # Delete file
            file_path = Path(document["file_path"])
            if file_path.exists():
                file_path.unlink()
            
            # Record deletion
            delete_record = {
                "timestamp": datetime.now().isoformat(),
                "doc_id": doc_id,
                "filename": document["filename"],
                "file_size": document["file_size"],
                "action": "delete"
            }
            self.delete_history.append(delete_record)
            
            # Remove from storage
            del self.documents[doc_id]
            if doc_id in self.document_hashes:
                del self.document_hashes[doc_id]
            if doc_id in self.document_metadata:
                del self.document_metadata[doc_id]
            
            # Trigger real-time update
            await self._notify_subscribers("document_deleted", {"id": doc_id, "filename": document["filename"]})
            
            logger.info(f"Document deleted: {doc_id} - {document['filename']}")
            return {"success": True, "message": "Document deleted successfully"}
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        return self.documents.get(doc_id)
    
    async def list_documents(self, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List all documents with optional filtering."""
        documents = list(self.documents.values())
        
        if filters:
            documents = self._apply_filters(documents, filters)
        
        return documents
    
    async def search_documents(self, query: str) -> List[Dict[str, Any]]:
        """Search documents by filename, tags, or metadata."""
        query_lower = query.lower()
        results = []
        
        for doc in self.documents.values():
            # Search in filename
            if query_lower in doc["filename"].lower():
                results.append(doc)
                continue
            
            # Search in tags
            if any(query_lower in tag.lower() for tag in doc.get("tags", [])):
                results.append(doc)
                continue
            
            # Search in metadata
            if any(query_lower in str(value).lower() for value in doc.get("metadata", {}).values()):
                results.append(doc)
                continue
        
        return results
    
    async def get_real_time_stats(self) -> Dict[str, Any]:
        """Get real-time document statistics."""
        now = time.time()
        
        # Cache stats for 1 second to avoid excessive calculations
        if now - self._last_update < 1.0 and self._stats_cache:
            return self._stats_cache
        
        stats = {
            "total_documents": len(self.documents),
            "total_size": sum(doc["file_size"] for doc in self.documents.values()),
            "active_documents": len([doc for doc in self.documents.values() if doc["status"] == "active"]),
            "categories": defaultdict(int),
            "recent_uploads": len([u for u in self.upload_history if now - datetime.fromisoformat(u["timestamp"]).timestamp() < 3600]),
            "recent_deletes": len([d for d in self.delete_history if now - datetime.fromisoformat(d["timestamp"]).timestamp() < 3600]),
            "recent_edits": len([e for e in self.edit_history if now - datetime.fromisoformat(e["timestamp"]).timestamp() < 3600]),
            "last_update": datetime.now().isoformat(),
            "upload_history": self.upload_history[-10:],  # Last 10 uploads
            "delete_history": self.delete_history[-10:],  # Last 10 deletes
            "edit_history": self.edit_history[-10:]      # Last 10 edits
        }
        
        # Count categories
        for doc in self.documents.values():
            category = doc.get("category", "general")
            stats["categories"][category] += 1
        
        self._stats_cache = stats
        self._last_update = now
        
        return stats
    
    async def subscribe_to_updates(self, subscriber_id: str) -> str:
        """Subscribe to real-time document updates."""
        self._subscribers.add(subscriber_id)
        logger.info(f"Subscriber added: {subscriber_id}")
        return f"Subscribed to document updates. Total subscribers: {len(self._subscribers)}"
    
    async def unsubscribe_from_updates(self, subscriber_id: str) -> str:
        """Unsubscribe from real-time document updates."""
        self._subscribers.discard(subscriber_id)
        logger.info(f"Subscriber removed: {subscriber_id}")
        return f"Unsubscribed from document updates. Total subscribers: {len(self._subscribers)}"
    
    async def _notify_subscribers(self, event_type: str, data: Any):
        """Notify all subscribers of document changes."""
        if not self._subscribers:
            return
        
        message = {
            "event_type": event_type,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        await self._update_queue.put(message)
        logger.debug(f"Queued update for {len(self._subscribers)} subscribers: {event_type}")
    
    async def _background_updater(self):
        """Background task for processing real-time updates."""
        while self._running:
            try:
                # Process update queue
                while not self._update_queue.empty():
                    message = await self._update_queue.get()
                    # In a real implementation, this would send to WebSocket clients
                    logger.debug(f"Processing update: {message['event_type']}")
                
                # Update stats periodically
                if time.time() - self._last_update > 5.0:  # Update every 5 seconds
                    await self.get_real_time_stats()
                
                await asyncio.sleep(0.1)  # Small delay to prevent CPU spinning
                
            except Exception as e:
                logger.error(f"Error in background updater: {e}")
                await asyncio.sleep(1.0)
    
    def _generate_doc_id(self, file_path: str, file_content: bytes) -> str:
        """Generate unique document ID."""
        filename = Path(file_path).name
        content_hash = hashlib.md5(file_content).hexdigest()[:8]
        timestamp = str(int(time.time()))[-6:]
        return f"{content_hash}_{timestamp}"
    
    def _apply_filters(self, documents: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply filters to document list."""
        filtered = documents
        
        if "category" in filters:
            filtered = [doc for doc in filtered if doc.get("category") == filters["category"]]
        
        if "status" in filters:
            filtered = [doc for doc in filtered if doc.get("status") == filters["status"]]
        
        if "min_size" in filters:
            filtered = [doc for doc in filtered if doc.get("file_size", 0) >= filters["min_size"]]
        
        if "max_size" in filters:
            filtered = [doc for doc in filtered if doc.get("file_size", 0) <= filters["max_size"]]
        
        if "tags" in filters:
            required_tags = set(filters["tags"])
            filtered = [doc for doc in filtered if required_tags.issubset(set(doc.get("tags", [])))]
        
        return filtered
    
    async def reset_system(self):
        """Reset the entire system - clear all documents and reset state."""
        try:
            # Clear all documents
            self.documents.clear()
            self.document_hashes.clear()
            self.document_metadata.clear()
            
            # Clear history
            self.upload_history.clear()
            self.delete_history.clear()
            self.edit_history.clear()
            
            # Clear stats cache
            self._stats_cache.clear()
            self._last_update = time.time()
            
            # Clear all files from data directory
            for file_path in self.data_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    import shutil
                    shutil.rmtree(file_path)
            
            # Recreate data directory
            self.data_dir.mkdir(parents=True, exist_ok=True)
            
            # Notify subscribers
            await self._notify_subscribers("system_reset", {
                "message": "System reset completed",
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info("Document Manager system reset completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset Document Manager system: {e}")
            return False
    
    def shutdown(self):
        """Shutdown the document manager."""
        self._running = False
        if hasattr(self, '_background_task'):
            self._background_task.cancel()
        logger.info("Document Manager shutdown complete")
