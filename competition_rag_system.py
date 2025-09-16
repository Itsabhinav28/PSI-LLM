# Competition-Grade RAG System for AI Grand Challenge Stage-1
# Target: >0.90 Recall@k, >0.70 Precision@k, >0.90 NDCG@k

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Plus
import faiss
from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModel
import re
from collections import defaultdict

class CompetitionRAGRetriever:
    """
    Competition-optimized RAG system for AI Grand Challenge
    Targets Stage-1 metrics: Recall@k (50%), Precision@k (20%), NDCG@k (30%)
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Multi-model ensemble for maximum coverage
        self.dense_models = {
            'mpnet': SentenceTransformer('all-mpnet-base-v2'),
            'minilm': SentenceTransformer('all-MiniLM-L12-v2'),
            'e5': SentenceTransformer('intfloat/e5-large-v2'),
            'bge': SentenceTransformer('BAAI/bge-large-en-v1.5')
        }
        
        # Domain-specific models
        self.domain_models = {
            'scientific': SentenceTransformer('allenai/specter'),
            'legal': SentenceTransformer('nlpaueb/legal-bert-base-uncased'),
            'medical': SentenceTransformer('emilyalsentzer/Bio_ClinicalBERT')
        }
        
        # Cross-encoder for reranking
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        
        # BM25 for sparse retrieval
        self.bm25 = None
        self.documents = []
        self.document_embeddings = {}
        
        # Query processor for expansion
        self.query_processor = QueryProcessor()
        
        # Score fusion weights (optimized for competition metrics)
        self.fusion_weights = {
            'bm25': 0.3,
            'mpnet': 0.25,
            'minilm': 0.2,
            'e5': 0.15,
            'bge': 0.1
        }

    def index_documents(self, documents: List[Dict], domain_type: str = 'general'):
        """
        Index documents with multi-strategy approach for maximum recall
        """
        self.documents = documents
        texts = [doc['text'] for doc in documents]
        
        # 1. Build BM25 index with enhanced preprocessing
        processed_texts = self._preprocess_for_bm25(texts)
        self.bm25 = BM25Plus(processed_texts)
        
        # 2. Build dense embeddings for all models
        print("Building dense embeddings...")
        for model_name, model in self.dense_models.items():
            print(f"Encoding with {model_name}...")
            embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
            
            # Build FAISS index for fast similarity search
            dimension = embeddings.shape[1]
            index = faiss.IndexHNSWFlat(dimension, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 100
            
            faiss.normalize_L2(embeddings)
            index.add(embeddings.astype(np.float32))
            
            self.document_embeddings[model_name] = {
                'index': index,
                'embeddings': embeddings
            }
        
        # 3. Domain-specific indexing if applicable
        if domain_type in self.domain_models:
            model = self.domain_models[domain_type]
            embeddings = model.encode(texts, batch_size=32)
            
            dimension = embeddings.shape[1]
            index = faiss.IndexHNSWFlat(dimension, 32)
            faiss.normalize_L2(embeddings)
            index.add(embeddings.astype(np.float32))
            
            self.document_embeddings[f'domain_{domain_type}'] = {
                'index': index,
                'embeddings': embeddings
            }

    def retrieve(self, query: str, k: int = 100, final_k: int = 10) -> List[Dict]:
        """
        Multi-stage retrieval optimized for competition metrics
        Stage 1: Cast wide net with multiple retrievers (maximize recall)
        Stage 2: Precision filtering and reranking
        """
        
        # Stage 1: Multi-retriever candidate generation (k=5*final_k for wide coverage)
        all_candidates = {}
        
        # 1.1: Query expansion for better coverage
        expanded_queries = self.query_processor.expand_query(query)
        
        # 1.2: BM25 sparse retrieval
        for exp_query in expanded_queries:
            bm25_scores = self.bm25.get_scores(self._preprocess_for_bm25([exp_query])[0])
            top_indices = np.argsort(bm25_scores)[::-1][:k]
            
            for idx in top_indices:
                doc_id = idx
                if doc_id not in all_candidates:
                    all_candidates[doc_id] = {'scores': {}, 'document': self.documents[idx]}
                all_candidates[doc_id]['scores']['bm25'] = float(bm25_scores[idx])
        
        # 1.3: Dense retrieval with all models
        for model_name, model_data in self.document_embeddings.items():
            model = self.dense_models.get(model_name, None)
            if model is None:
                continue
                
            for exp_query in expanded_queries:
                query_embedding = model.encode([exp_query])
                faiss.normalize_L2(query_embedding)
                
                # Search with high k for maximum recall
                scores, indices = model_data['index'].search(query_embedding.astype(np.float32), k)
                
                for score, idx in zip(scores[0], indices[0]):
                    if idx != -1:  # Valid index
                        doc_id = int(idx)
                        if doc_id not in all_candidates:
                            all_candidates[doc_id] = {'scores': {}, 'document': self.documents[doc_id]}
                        all_candidates[doc_id]['scores'][model_name] = float(score)
        
        # Stage 2: Score fusion and candidate ranking
        ranked_candidates = self._fuse_scores(all_candidates)
        
        # Stage 3: Cross-encoder reranking for precision optimization
        if len(ranked_candidates) > 50:  # Only rerank if we have enough candidates
            reranked = self._cross_encoder_rerank(query, ranked_candidates[:100], final_k)
            return reranked
        else:
            return ranked_candidates[:final_k]

    def _preprocess_for_bm25(self, texts: List[str]) -> List[List[str]]:
        """Enhanced BM25 preprocessing for better keyword matching"""
        processed = []
        for text in texts:
            # Lowercasing and basic cleaning
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Tokenization with stopword removal
            tokens = text.split()
            # Remove common English stopwords but keep domain-specific terms
            stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            tokens = [token for token in tokens if token not in stopwords and len(token) > 2]
            
            processed.append(tokens)
        return processed

    def _fuse_scores(self, candidates: Dict) -> List[Dict]:
        """Advanced score fusion optimized for competition metrics"""
        fused_candidates = []
        
        for doc_id, data in candidates.items():
            scores = data['scores']
            fused_score = 0.0
            
            # Weighted combination of all available scores
            for score_type, weight in self.fusion_weights.items():
                if score_type in scores:
                    # Normalize scores to [0,1] range
                    normalized_score = self._normalize_score(scores[score_type], score_type)
                    fused_score += weight * normalized_score
            
            # Boost score if document appears in multiple retrievers (ensemble confidence)
            ensemble_boost = len(scores) / len(self.fusion_weights)
            fused_score *= ensemble_boost
            
            fused_candidates.append({
                'document': data['document'],
                'fused_score': fused_score,
                'individual_scores': scores,
                'doc_id': doc_id
            })
        
        # Sort by fused score (descending)
        return sorted(fused_candidates, key=lambda x: x['fused_score'], reverse=True)

    def _normalize_score(self, score: float, score_type: str) -> float:
        """Normalize different types of scores to [0,1] range"""
        if score_type == 'bm25':
            # BM25 scores are typically in [0, inf), use sigmoid
            return 1 / (1 + np.exp(-score))
        else:
            # Dense retrieval scores are typically cosine similarities in [-1,1]
            return (score + 1) / 2

    def _cross_encoder_rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """Cross-encoder reranking for precision optimization"""
        if len(candidates) == 0:
            return []
        
        # Prepare query-document pairs
        pairs = []
        for candidate in candidates:
            doc_text = candidate['document']['text'][:512]  # Truncate for efficiency
            pairs.append([query, doc_text])
        
        # Get cross-encoder scores
        ce_scores = self.cross_encoder.predict(pairs)
        
        # Update candidates with cross-encoder scores
        for i, candidate in enumerate(candidates):
            candidate['ce_score'] = float(ce_scores[i])
            # Combine cross-encoder with fusion score (weighted)
            candidate['final_score'] = 0.7 * candidate['ce_score'] + 0.3 * candidate['fused_score']
        
        # Final ranking
        reranked = sorted(candidates, key=lambda x: x['final_score'], reverse=True)
        return reranked[:top_k]


class QueryProcessor:
    """Advanced query processing for better retrieval coverage"""
    
    def __init__(self):
        self.entity_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Names
            r'\b\d{4}\b',  # Years
            r'\b[A-Z]{2,}\b',  # Acronyms
        ]
    
    def expand_query(self, query: str, max_expansions: int = 5) -> List[str]:
        """Generate query variations for better coverage"""
        expansions = [query]  # Original query first
        
        # 1. Synonym expansion
        synonyms = self._generate_synonyms(query)
        if synonyms:
            expansions.append(synonyms)
        
        # 2. Entity extraction and expansion
        entities = self._extract_entities(query)
        for entity in entities[:2]:  # Limit to 2 entities
            expansions.append(f"{query} {entity}")
        
        # 3. Question type variations
        if "what is" in query.lower():
            expansions.append(query.replace("what is", "definition of"))
            expansions.append(query.replace("what is", "meaning of"))
        elif "how does" in query.lower():
            expansions.append(query.replace("how does", "mechanism of"))
            expansions.append(query.replace("how does", "process of"))
        elif "why does" in query.lower():
            expansions.append(query.replace("why does", "reason for"))
            expansions.append(query.replace("why does", "cause of"))
        
        # 4. Remove duplicates and limit
        unique_expansions = list(dict.fromkeys(expansions))
        return unique_expansions[:max_expansions]
    
    def _generate_synonyms(self, query: str) -> str:
        """Basic synonym generation (can be enhanced with WordNet/ConceptNet)"""
        synonyms_map = {
            'disease': 'illness condition disorder',
            'treatment': 'therapy cure remedy',
            'research': 'study investigation analysis',
            'development': 'creation formation growth',
            'analysis': 'examination evaluation study'
        }
        
        words = query.lower().split()
        expanded_words = []
        
        for word in words:
            if word in synonyms_map:
                expanded_words.extend([word] + synonyms_map[word].split())
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract potential entities from query"""
        entities = []
        for pattern in self.entity_patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        return entities


class CompetitionEvaluator:
    """Evaluation utilities optimized for competition metrics"""
    
    @staticmethod
    def calculate_recall_at_k(retrieved_docs: List, relevant_docs: List, k: int) -> float:
        """Calculate Recall@k - most important metric (50% weight)"""
        if not relevant_docs:
            return 0.0
        
        retrieved_set = set([doc['doc_id'] for doc in retrieved_docs[:k]])
        relevant_set = set(relevant_docs)
        
        intersection = len(retrieved_set.intersection(relevant_set))
        return intersection / len(relevant_set)
    
    @staticmethod
    def calculate_precision_at_k(retrieved_docs: List, relevant_docs: List, k: int) -> float:
        """Calculate Precision@k (20% weight)"""
        if not retrieved_docs[:k]:
            return 0.0
        
        retrieved_set = set([doc['doc_id'] for doc in retrieved_docs[:k]])
        relevant_set = set(relevant_docs)
        
        intersection = len(retrieved_set.intersection(relevant_set))
        return intersection / min(k, len(retrieved_docs))
    
    @staticmethod
    def calculate_ndcg_at_k(retrieved_docs: List, relevant_docs: Dict, k: int) -> float:
        """Calculate NDCG@k (30% weight)"""
        def dcg_at_k(scores, k):
            scores = np.array(scores)[:k]
            return np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))
        
        # Get relevance scores for retrieved documents
        retrieved_relevance = []
        for doc in retrieved_docs[:k]:
            doc_id = doc.get('doc_id', doc.get('document', {}).get('id', ''))
            relevance = relevant_docs.get(doc_id, 0)  # 0 if not relevant
            retrieved_relevance.append(relevance)
        
        if not retrieved_relevance or max(retrieved_relevance) == 0:
            return 0.0
        
        # Calculate DCG
        dcg = dcg_at_k(retrieved_relevance, k)
        
        # Calculate IDCG (perfect ranking)
        ideal_relevance = sorted(relevant_docs.values(), reverse=True)[:k]
        idcg = dcg_at_k(ideal_relevance, k)
        
        return dcg / idcg if idcg > 0 else 0.0


# Example usage and testing
if __name__ == "__main__":
    # Initialize the competition system
    retriever = CompetitionRAGRetriever()
    evaluator = CompetitionEvaluator()
    
    # Mock documents and queries for testing
    documents = [
        {"id": "doc1", "text": "COVID-19 research shows significant impact on respiratory system"},
        {"id": "doc2", "text": "Machine learning algorithms for medical diagnosis"},
        {"id": "doc3", "text": "Legal framework for data protection in European Union"},
        {"id": "doc4", "text": "Climate change effects on global agriculture systems"},
        {"id": "doc5", "text": "Quantum computing applications in cryptography"}
    ]
    
    # Index the documents
    retriever.index_documents(documents)
    
    # Test query
    query = "What are the effects of COVID-19 on lungs?"
    results = retriever.retrieve(query, k=100, final_k=5)
    
    print(f"Query: {query}")
    print(f"Retrieved {len(results)} documents:")
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result.get('final_score', 'N/A'):.3f}")
        print(f"   Text: {result['document']['text'][:100]}...")
        print()
    
    # Example evaluation (mock relevant documents)
    relevant_docs = ["doc1"]  # doc1 is relevant to COVID query
    relevant_scores = {"doc1": 3, "doc2": 1}  # Graded relevance
    
    recall_5 = evaluator.calculate_recall_at_k(results, relevant_docs, 5)
    precision_5 = evaluator.calculate_precision_at_k(results, relevant_docs, 5)
    ndcg_5 = evaluator.calculate_ndcg_at_k(results, relevant_scores, 5)
    
    print(f"Evaluation Metrics:")
    print(f"Recall@5: {recall_5:.3f}")
    print(f"Precision@5: {precision_5:.3f}")
    print(f"NDCG@5: {ndcg_5:.3f}")
    
    # Competition score calculation
    competition_score = (0.5 * recall_5 + 0.2 * precision_5 + 0.3 * ndcg_5)
    print(f"Competition Score: {competition_score:.3f}")