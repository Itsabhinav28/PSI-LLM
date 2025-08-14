"""
Gemini API Client for RAG Pipeline

Handles integration with Google's Gemini API for text generation
and response creation using retrieved context.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import google.generativeai as genai
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class GeminiClient:
    """Client for Google Gemini API integration."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        """Initialize Gemini client with configuration."""
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model_name = model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
        
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        try:
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"Successfully initialized Gemini model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            raise
    
    def generate_response(self, prompt: str, context: Optional[List[str]] = None) -> Dict[str, Any]:
        """Generate response using Gemini API."""
        try:
            # Build full prompt with context
            full_prompt = self._build_prompt(prompt, context)
            
            # Configure generation parameters
            generation_config = {
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 40,
                'max_output_tokens': 2048,
            }
            
            # Generate response
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            return {
                'success': True,
                'response': response.text,
                'prompt_tokens': len(full_prompt.split()),
                'response_tokens': len(response.text.split()),
                'model_used': self.model_name
            }
            
        except Exception as e:
            error_msg = str(e)
            
            # Handle rate limit errors specifically
            if "429" in error_msg or "quota" in error_msg.lower():
                return {
                    'success': False,
                    'error': 'API rate limit exceeded. Please wait a moment and try again.',
                    'model_used': self.model_name,
                    'rate_limited': True
                }
            
            logger.error(f"Error generating response with Gemini: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_used': self.model_name
            }
    
    def _build_prompt(self, prompt: str, context: Optional[List[str]] = None) -> str:
        """Build a comprehensive prompt with context for RAG."""
        if not context:
            return prompt
        
        context_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(context)])
        
        full_prompt = f"""You are a helpful AI assistant. Use the following context to answer the user's question.

{context_text}

User Question: {prompt}

Please provide a comprehensive answer based on the context provided. If the context doesn't contain enough information to answer the question, please say so clearly."""
        
        return full_prompt
    
    def generate_rag_response(
        self,
        query: str,
        retrieved_documents: List[Dict[str, Any]],
        max_context_length: int = 4000
    ) -> Dict[str, Any]:
        """Generate RAG response using retrieved documents as context."""
        try:
            # Extract and prepare context from retrieved documents
            context_chunks = []
            total_length = 0
            
            for doc in retrieved_documents:
                content = doc.get('content', '')
                if total_length + len(content) <= max_context_length:
                    context_chunks.append(content)
                    total_length += len(content)
                else:
                    break
            
            # Generate response with context
            result = self.generate_response(query, context_chunks)
            
            # Add RAG-specific metadata
            if result['success']:
                result['rag_metadata'] = {
                    'documents_used': len(context_chunks),
                    'total_context_length': total_length,
                    'context_chunks': [len(chunk) for chunk in context_chunks]
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_used': self.model_name
            }
    
    def test_connection(self) -> bool:
        """Test the connection to Gemini API."""
        try:
            test_prompt = "Hello, this is a test message. Please respond with 'Connection successful'."
            response = self.generate_response(test_prompt)
            return response['success']
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current Gemini model."""
        return {
            'model_name': self.model_name,
            'api_key_configured': bool(self.api_key),
            'connection_status': self.test_connection()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the Gemini client."""
        try:
            # Test basic connection
            connection_test = self.test_connection()
            
            if connection_test:
                return {
                    'status': 'healthy',
                    'model_name': self.model_name,
                    'api_key_configured': bool(self.api_key),
                    'connection': 'working',
                    'last_test': 'successful'
                }
            else:
                return {
                    'status': 'unhealthy',
                    'model_name': self.model_name,
                    'api_key_configured': bool(self.api_key),
                    'connection': 'failed',
                    'error': 'Connection test failed'
                }
                
        except Exception as e:
            error_msg = str(e)
            
            # Check if it's a rate limit error
            if "429" in error_msg or "quota" in error_msg.lower():
                return {
                    'status': 'rate_limited',
                    'model_name': self.model_name,
                    'api_key_configured': bool(self.api_key),
                    'connection': 'rate_limited',
                    'error': 'API rate limit exceeded - will resume shortly'
                }
            
            return {
                'status': 'unhealthy',
                'model_name': self.model_name,
                'api_key_configured': bool(self.api_key),
                'connection': 'failed',
                'error': str(e)
            }
