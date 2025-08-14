"""
Document Preprocessor for RAG Pipeline

Handles text cleaning, normalization, and preprocessing to improve
document quality before chunking and embedding.
"""

import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import unicodedata

logger = logging.getLogger(__name__)


class DocumentPreprocessor:
    """Advanced document preprocessing with multiple cleaning strategies."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize preprocessor with configuration."""
        self.config = config or {}
        
        # Default preprocessing options
        self.remove_extra_whitespace = self.config.get('remove_extra_whitespace', True)
        self.normalize_unicode = self.config.get('normalize_unicode', True)
        self.remove_special_chars = self.config.get('remove_special_chars', False)
        self.convert_to_lowercase = self.config.get('convert_to_lowercase', False)
        self.remove_numbers = self.config.get('remove_numbers', False)
        self.remove_urls = self.config.get('remove_urls', True)
        self.remove_emails = self.config.get('remove_emails', True)
        self.min_text_length = self.config.get('min_text_length', 10)
    
    def clean_text(self, text: str) -> str:
        """Apply comprehensive text cleaning."""
        if not text or not isinstance(text, str):
            return ""
        
        # Store original length for logging
        original_length = len(text)
        
        # Apply cleaning steps
        text = self._normalize_unicode(text)
        text = self._remove_urls(text)
        text = self._remove_emails(text)
        text = self._remove_special_characters(text)
        text = self._clean_whitespace(text)
        text = self._remove_numbers(text)
        text = self._convert_case(text)
        text = self._final_cleanup(text)
        
        # Log cleaning results
        cleaned_length = len(text)
        if original_length != cleaned_length:
            logger.info(f"Text cleaned: {original_length} -> {cleaned_length} characters")
        
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        if not self.normalize_unicode:
            return text
        
        try:
            # Normalize unicode to NFKC (compatibility composition)
            text = unicodedata.normalize('NFKC', text)
            
            # Replace common unicode characters with ASCII equivalents
            unicode_replacements = {
                '–': '-',  # en dash
                '—': '-',  # em dash
                '"': '"',  # left double quotation mark
                '"': '"',  # right double quotation mark
                ''': "'",  # left single quotation mark
                ''': "'",  # right single quotation mark
                '…': '...',  # horizontal ellipsis
                '°': ' degrees',  # degree sign
                '±': '+/-',  # plus-minus sign
                '≤': '<=',  # less-than or equal to
                '≥': '>=',  # greater-than or equal to
            }
            
            for unicode_char, ascii_char in unicode_replacements.items():
                text = text.replace(unicode_char, ascii_char)
            
            return text
        except Exception as e:
            logger.warning(f"Unicode normalization failed: {e}")
            return text
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text."""
        if not self.remove_urls:
            return text
        
        # Pattern to match various URL formats
        url_patterns = [
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            r'www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            r'[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/[^\s]*)?'
        ]
        
        for pattern in url_patterns:
            text = re.sub(pattern, '[URL]', text)
        
        return text
    
    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text."""
        if not self.remove_emails:
            return text
        
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        text = re.sub(email_pattern, '[EMAIL]', text)
        
        return text
    
    def _remove_special_characters(self, text: str) -> str:
        """Remove or replace special characters."""
        if not self.remove_special_chars:
            return text
        
        # Replace special characters with spaces
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}]', ' ', text)
        
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean and normalize whitespace."""
        if not self.remove_extra_whitespace:
            return text
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Clean up line breaks and paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+\n', '\n', text)
        
        return text
    
    def _remove_numbers(self, text: str) -> str:
        """Remove numbers from text."""
        if not self.remove_numbers:
            return text
        
        # Remove standalone numbers but preserve numbers in words
        text = re.sub(r'\b\d+\b', '', text)
        
        return text
    
    def _convert_case(self, text: str) -> str:
        """Convert text case if specified."""
        if self.convert_to_lowercase:
            text = text.lower()
        
        return text
    
    def _final_cleanup(self, text: str) -> str:
        """Final cleanup and validation."""
        # Remove any remaining excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Ensure minimum text length
        if len(text) < self.min_text_length:
            return ""
        
        return text
    
    def preprocess_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess a complete document."""
        if not document or 'content' not in document:
            return document
        
        try:
            # Clean the text content
            cleaned_content = self.clean_text(document['content'])
            
            # Update document with cleaned content
            processed_document = document.copy()
            processed_document['content'] = cleaned_content
            processed_document['preprocessing_info'] = {
                'original_length': len(document['content']),
                'cleaned_length': len(cleaned_content),
                'preprocessing_applied': True
            }
            
            # Remove document if content is too short after cleaning
            if len(cleaned_content) < self.min_text_length:
                logger.warning(f"Document {document.get('metadata', {}).get('file_name', 'unknown')} too short after cleaning")
                processed_document['content'] = ""
                processed_document['preprocessing_info']['removed'] = True
            
            return processed_document
            
        except Exception as e:
            logger.error(f"Error preprocessing document: {e}")
            return document
    
    def preprocess_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess multiple documents."""
        processed_documents = []
        
        for doc in documents:
            try:
                processed_doc = self.preprocess_document(doc)
                if processed_doc.get('content'):  # Only keep documents with content
                    processed_documents.append(processed_doc)
            except Exception as e:
                logger.error(f"Error preprocessing document: {e}")
                continue
        
        logger.info(f"Preprocessed {len(documents)} documents, kept {len(processed_documents)}")
        return processed_documents
    
    def get_preprocessing_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about preprocessing results."""
        if not documents:
            return {}
        
        total_docs = len(documents)
        total_original_chars = sum(
            doc.get('preprocessing_info', {}).get('original_length', 0) 
            for doc in documents
        )
        total_cleaned_chars = sum(
            doc.get('preprocessing_info', {}).get('cleaned_length', 0) 
            for doc in documents
        )
        
        removed_docs = sum(
            1 for doc in documents 
            if doc.get('preprocessing_info', {}).get('removed', False)
        )
        
        return {
            'total_documents': total_docs,
            'documents_removed': removed_docs,
            'documents_kept': total_docs - removed_docs,
            'total_original_characters': total_original_chars,
            'total_cleaned_characters': total_cleaned_chars,
            'character_reduction_percent': round(
                ((total_original_chars - total_cleaned_chars) / total_original_chars * 100), 2
            ) if total_original_chars > 0 else 0
        }
    
    def validate_text_quality(self, text: str) -> Dict[str, Any]:
        """Validate text quality after preprocessing."""
        if not text:
            return {'quality_score': 0, 'issues': ['empty_text']}
        
        issues = []
        score = 100
        
        # Check text length
        if len(text) < 50:
            issues.append('very_short_text')
            score -= 30
        elif len(text) < 100:
            issues.append('short_text')
            score -= 15
        
        # Check for excessive whitespace
        if text.count('  ') > len(text) * 0.1:
            issues.append('excessive_whitespace')
            score -= 10
        
        # Check for repeated characters
        if re.search(r'(.)\1{4,}', text):
            issues.append('repeated_characters')
            score -= 10
        
        # Check for proper sentence structure
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 1:
            avg_sentence_length = sum(len(s.strip()) for s in sentences if s.strip()) / len([s for s in sentences if s.strip()])
            if avg_sentence_length > 200:
                issues.append('very_long_sentences')
                score -= 15
            elif avg_sentence_length < 10:
                issues.append('very_short_sentences')
                score -= 10
        
        return {
            'quality_score': max(0, score),
            'issues': issues,
            'text_length': len(text),
            'sentence_count': len([s for s in sentences if s.strip()])
        }
