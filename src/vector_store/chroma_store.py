"""
ChromaDB Vector Store for RAG Pipeline

Handles document embedding storage, indexing, and similarity search
using ChromaDB for efficient vector operations.
"""

import os
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class ChromaStore:
    """ChromaDB-based vector store for document embeddings."""
    
    def __init__(
        self,
        persist_directory: str = "./data/embeddings/chroma",
        embedding_model: str = "all-MiniLM-L6-v2",
        collection_name: str = "rag_documents"
    ):
        """Initialize ChromaDB vector store."""
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        
        # Create directory if it doesn't exist
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            logger.info(f"Loaded embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        logger.info(f"ChromaDB store initialized at {self.persist_directory}")
    
    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get existing collection or create new one."""
        try:
            collection = self.client.get_collection(self.collection_name)
            logger.info(f"Using existing collection: {self.collection_name}")
        except Exception:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "RAG Pipeline Document Collection"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        return collection
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False)
            return embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def add_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """Add documents to the vector store."""
        if not documents:
            return {"added": 0, "errors": 0}
        
        total_added = 0
        total_errors = 0
        
        # Process documents in batches
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            try:
                # Extract text content and metadata
                texts = []
                metadatas = []
                ids = []
                
                for doc in batch:
                    content = doc.get('content', '')
                    if not content or len(content.strip()) < 10:
                        continue
                    
                    # Generate unique ID
                    doc_id = self._generate_document_id(doc)
                    
                    # Prepare metadata
                    metadata = {
                        'file_name': doc.get('metadata', {}).get('file_name', 'unknown'),
                        'file_path': doc.get('metadata', {}).get('file_path', 'unknown'),
                        'format': doc.get('metadata', {}).get('format', 'unknown'),
                        'chunk_id': doc.get('chunk_id', 'unknown'),
                        'chunk_type': doc.get('metadata', {}).get('chunk_type', 'unknown'),
                        'char_count': len(content),
                        'source': 'rag_pipeline'
                    }
                    
                    texts.append(content)
                    metadatas.append(metadata)
                    ids.append(doc_id)
                
                if not texts:
                    continue
                
                # Generate embeddings
                embeddings = self.generate_embeddings(texts)
                
                # Add to collection
                self.collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
                total_added += len(texts)
                logger.info(f"Added batch of {len(texts)} documents")
                
            except Exception as e:
                logger.error(f"Error adding batch {i//batch_size + 1}: {e}")
                total_errors += len(batch)
        
        logger.info(f"Document addition complete: {total_added} added, {total_errors} errors")
        return {
            "added": total_added,
            "errors": total_errors,
            "total_processed": len(documents)
        }
    
    def search_documents(
        self,
        query: str,
        n_results: int = 5,
        similarity_threshold: float = 0.2
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using vector similarity."""
        try:
            # Generate query embedding
            query_embedding = self.generate_embeddings([query])[0]
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Process results
            documents = []
            if results.get('documents') and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(
                    zip(results['documents'][0], results['metadatas'][0], results['distances'][0])
                ):
                    # Convert distance to similarity score (Chroma uses cosine distance by default)
                    try:
                        similarity_score = 1 - float(distance)
                    except Exception:
                        similarity_score = 0.0
                    
                    if similarity_score >= similarity_threshold:
                        documents.append({
                            'content': doc,
                            'metadata': metadata,
                            'similarity_score': round(similarity_score, 4),
                            'rank': i + 1
                        })
            
            # If nothing met the threshold, return the top results without filtering
            if not documents and results.get('documents') and results['documents'][0]:
                fallback = []
                for i, (doc, metadata, distance) in enumerate(
                    zip(results['documents'][0], results['metadatas'][0], results['distances'][0])
                ):
                    try:
                        sim = 1 - float(distance)
                    except Exception:
                        sim = 0.0
                    fallback.append({
                        'content': doc,
                        'metadata': metadata,
                        'similarity_score': round(sim, 4),
                        'rank': i + 1
                    })
                documents = fallback

            logger.info(f"Search completed: {len(documents)} relevant documents found")
            return documents
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            
            # Get sample documents for analysis
            sample_results = self.collection.peek(limit=min(100, count))
            
            # Analyze document lengths
            doc_lengths = [len(doc) for doc in sample_results['documents']]
            
            stats = {
                'total_documents': count,
                'collection_name': self.collection_name,
                'embedding_model': str(self.embedding_model),
                'persist_directory': str(self.persist_directory),
                'average_document_length': round(np.mean(doc_lengths), 2) if doc_lengths else 0,
                'min_document_length': min(doc_lengths) if doc_lengths else 0,
                'max_document_length': max(doc_lengths) if doc_lengths else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {}
    
    def delete_collection(self) -> bool:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} deleted successfully")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
    
    def reset_collection(self) -> bool:
        """Reset the collection (delete and recreate)."""
        try:
            self.delete_collection()
            self.collection = self._get_or_create_collection()
            logger.info(f"Collection {self.collection_name} reset successfully")
            return True
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False
    
    def _generate_document_id(self, document: Dict[str, Any]) -> str:
        """Generate a unique document ID."""
        content = document.get('content', '')
        metadata = document.get('metadata', {})
        
        # Create a hash from content and metadata
        hash_input = f"{content[:100]}_{metadata.get('file_name', '')}_{metadata.get('chunk_id', '')}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the vector store."""
        try:
            # Test embedding generation
            test_embedding = self.generate_embeddings(["test"])[0]
            
            # Test collection access
            stats = self.get_collection_stats()
            
            return {
                'status': 'healthy',
                'embedding_model': 'working',
                'chroma_connection': 'working',
                'collection_accessible': True,
                'stats': stats
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'embedding_model': 'failed',
                'chroma_connection': 'failed',
                'collection_accessible': False
            }

    async def clear_collection(self):
        """Clear all documents from the collection."""
        try:
            if self.collection:
                # Get all document IDs first
                results = self.collection.get()
                if results and results['ids']:
                    # Delete all documents by their IDs
                    self.collection.delete(ids=results['ids'])
                    logger.info(f"ChromaDB collection cleared: {len(results['ids'])} documents removed")
                else:
                    logger.info("ChromaDB collection is already empty")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to clear ChromaDB collection: {e}")
            return False
