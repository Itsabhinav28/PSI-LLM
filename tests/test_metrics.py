"""
RAG Pipeline Evaluation Metrics - Stage 1 Assessment
Implements Recall@k, Precision@k, and NDCG@k for RAG evaluation
"""

import numpy as np
import json
import logging
import time
from typing import List, Dict, Any, Tuple
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.vector_store.chroma_store import ChromaStore
from src.retrieval.retriever import DocumentRetriever
from src.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

class RAGEvaluator:
    """Comprehensive RAG Pipeline Evaluator for Stage 1 Metrics"""
    
    def __init__(self):
        self.chroma_store = ChromaStore()
        self.retriever = DocumentRetriever(self.chroma_store)
        self.rag_pipeline = RAGPipeline()
        
    def recall_at_k(self, relevant_chunks: List[str], retrieved_chunks: List[str], k: int) -> float:
        """
        Calculate Recall@k: Fraction of relevant chunks retrieved in top-k results
        
        Args:
            relevant_chunks: List of chunk IDs that are actually relevant
            retrieved_chunks: List of chunk IDs returned by the retriever
            k: Number of top results to consider
            
        Returns:
            Recall@k score (0.0 to 1.0)
        """
        if not relevant_chunks:
            return 0.0
            
        relevant_set = set(relevant_chunks)
        retrieved_set = set(retrieved_chunks[:k])
        
        intersection = len(relevant_set & retrieved_set)
        return intersection / len(relevant_set)
    
    def precision_at_k(self, relevant_chunks: List[str], retrieved_chunks: List[str], k: int) -> float:
        """
        Calculate Precision@k: Fraction of top-k results that are relevant
        
        Args:
            relevant_chunks: List of chunk IDs that are actually relevant
            retrieved_chunks: List of chunk IDs returned by the retriever
            k: Number of top results to consider
            
        Returns:
            Precision@k score (0.0 to 1.0)
        """
        if k == 0:
            return 0.0
            
        relevant_set = set(relevant_chunks)
        retrieved_set = set(retrieved_chunks[:k])
        
        intersection = len(relevant_set & retrieved_set)
        return intersection / k
    
    def ndcg_at_k(self, relevant_chunks: List[str], retrieved_chunks: List[str], k: int) -> float:
        """
        Calculate NDCG@k: Normalized Discounted Cumulative Gain
        
        Args:
            relevant_chunks: List of chunk IDs that are actually relevant
            retrieved_chunks: List of chunk IDs returned by the retriever
            k: Number of top results to consider
            
        Returns:
            NDCG@k score (0.0 to 1.0)
        """
        if not relevant_chunks or k == 0:
            return 0.0
            
        # Calculate DCG
        dcg = 0.0
        for i, chunk_id in enumerate(retrieved_chunks[:k]):
            if chunk_id in relevant_chunks:
                dcg += 1.0 / np.log2(i + 2)
        
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(len(relevant_chunks), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_retrieval(self, test_queries: List[Dict[str, Any]], k_values: List[int] = [5, 10]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate retrieval performance on test queries
        
        Args:
            test_queries: List of test queries with relevant chunks
            k_values: List of k values to evaluate (e.g., [5, 10])
            
        Returns:
            Dictionary with metrics for each k value
        """
        results = {}
        
        for k in k_values:
            recall_scores = []
            precision_scores = []
            ndcg_scores = []
            
            for query_data in test_queries:
                query = query_data["query"]
                relevant_chunks = query_data["relevant_chunks"]
                
                try:
                    # Get retrieval results
                    search_results = self.retriever.retrieve_documents(
                        query=query,
                        n_results=k * 2  # Get more results than needed
                    )
                    
                    retrieved_chunks = [result.metadata.get("chunk_id", f"chunk_{i}") for i, result in enumerate(search_results)]
                    
                    # Calculate metrics
                    recall = self.recall_at_k(relevant_chunks, retrieved_chunks, k)
                    precision = self.precision_at_k(relevant_chunks, retrieved_chunks, k)
                    ndcg = self.ndcg_at_k(relevant_chunks, retrieved_chunks, k)
                    
                    recall_scores.append(recall)
                    precision_scores.append(precision)
                    ndcg_scores.append(ndcg)
                    
                    logger.info(f"Query: {query[:50]}... | R@{k}: {recall:.3f} | P@{k}: {precision:.3f} | NDCG@{k}: {ndcg:.3f}")
                    
                except Exception as e:
                    logger.error(f"Error processing query '{query}': {e}")
                    recall_scores.append(0.0)
                    precision_scores.append(0.0)
                    ndcg_scores.append(0.0)
            
            # Calculate average metrics
            results[f"@{k}"] = {
                "recall": np.mean(recall_scores),
                "precision": np.mean(precision_scores),
                "ndcg": np.mean(ndcg_scores),
                "num_queries": len(test_queries)
            }
            
            logger.info(f"Average Metrics @{k}: R={np.mean(recall_scores):.3f}, P={np.mean(precision_scores):.3f}, NDCG={np.mean(ndcg_scores):.3f}")
        
        return results
    
    def evaluate_end_to_end(self, test_queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate end-to-end RAG pipeline performance
        
        Args:
            test_queries: List of test queries with expected answers
            
        Returns:
            Dictionary with end-to-end evaluation results
        """
        results = {
            "total_queries": len(test_queries),
            "successful_queries": 0,
            "failed_queries": 0,
            "average_response_time": 0.0,
            "response_quality_scores": []
        }
        
        response_times = []
        
        for i, query_data in enumerate(test_queries):
            query = query_data["query"]
            expected_answer = query_data.get("expected_answer", "")
            
            try:
                # Get RAG response
                start_time = time.time()
                response = self.rag_pipeline.query(query)
                end_time = time.time()
                
                response_time = end_time - start_time
                response_times.append(response_time)
                
                # Basic quality assessment
                answer = response.get("response", "")
                sources = response.get("sources", [])
                
                # Simple quality score based on answer length and source presence
                quality_score = min(len(answer) / 100, 1.0) if answer else 0.0
                if sources:
                    quality_score += 0.2
                
                results["response_quality_scores"].append(quality_score)
                results["successful_queries"] += 1
                
                logger.info(f"Query {i+1}: {query[:50]}... | Time: {response_time:.2f}s | Quality: {quality_score:.3f}")
                
            except Exception as e:
                logger.error(f"Error processing query '{query}': {e}")
                results["failed_queries"] += 1
                results["response_quality_scores"].append(0.0)
        
        # Calculate averages
        if response_times:
            results["average_response_time"] = np.mean(response_times)
        
        if results["response_quality_scores"]:
            results["average_quality_score"] = np.mean(results["response_quality_scores"])
        
        return results

def create_sample_test_data() -> List[Dict[str, Any]]:
    """
    Create sample test data for evaluation
    In a real scenario, this would be your ground-truth dataset
    """
    return [
        {
            "query": "What is machine learning?",
            "relevant_chunks": ["ml_chunk_1", "ml_chunk_2", "ai_chunk_1"],
            "expected_answer": "Machine learning is a subset of artificial intelligence..."
        },
        {
            "query": "How does neural network training work?",
            "relevant_chunks": ["nn_chunk_1", "training_chunk_1", "backprop_chunk_1"],
            "expected_answer": "Neural network training involves forward and backward propagation..."
        },
        {
            "query": "What are the benefits of deep learning?",
            "relevant_chunks": ["dl_chunk_1", "benefits_chunk_1", "performance_chunk_1"],
            "expected_answer": "Deep learning offers superior performance in complex tasks..."
        },
        {
            "query": "Explain gradient descent optimization",
            "relevant_chunks": ["optimization_chunk_1", "gradient_chunk_1", "descent_chunk_1"],
            "expected_answer": "Gradient descent is an optimization algorithm..."
        },
        {
            "query": "What is overfitting in machine learning?",
            "relevant_chunks": ["overfitting_chunk_1", "generalization_chunk_1", "validation_chunk_1"],
            "expected_answer": "Overfitting occurs when a model performs well on training data..."
        }
    ]

def run_evaluation():
    """Run comprehensive RAG evaluation"""
    import time
    
    logger.info("🚀 Starting RAG Pipeline Evaluation - Stage 1 Metrics")
    logger.info("=" * 60)
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    # Load test data
    test_queries = create_sample_test_data()
    logger.info(f"📊 Loaded {len(test_queries)} test queries")
    
    # Evaluate retrieval performance
    logger.info("\n🔍 Evaluating Retrieval Performance...")
    retrieval_results = evaluator.evaluate_retrieval(test_queries, k_values=[5, 10])
    
    # Print results table
    logger.info("\n📈 RETRIEVAL METRICS RESULTS")
    logger.info("=" * 50)
    logger.info("| Metric    | @5      | @10     |")
    logger.info("|-----------|---------|---------|")
    logger.info(f"| Recall    | {retrieval_results['@5']['recall']:.3f}    | {retrieval_results['@10']['recall']:.3f}    |")
    logger.info(f"| Precision | {retrieval_results['@5']['precision']:.3f}    | {retrieval_results['@10']['precision']:.3f}    |")
    logger.info(f"| NDCG      | {retrieval_results['@5']['ndcg']:.3f}    | {retrieval_results['@10']['ndcg']:.3f}    |")
    
    # Evaluate end-to-end performance
    logger.info("\n🔄 Evaluating End-to-End RAG Performance...")
    e2e_results = evaluator.evaluate_end_to_end(test_queries)
    
    logger.info("\n📊 END-TO-END RESULTS")
    logger.info("=" * 30)
    logger.info(f"Total Queries: {e2e_results['total_queries']}")
    logger.info(f"Successful: {e2e_results['successful_queries']}")
    logger.info(f"Failed: {e2e_results['failed_queries']}")
    logger.info(f"Avg Response Time: {e2e_results['average_response_time']:.2f}s")
    logger.info(f"Avg Quality Score: {e2e_results['average_quality_score']:.3f}")
    
    # Save results
    results = {
        "retrieval_metrics": retrieval_results,
        "end_to_end_metrics": e2e_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n💾 Results saved to evaluation_results.json")
    logger.info("✅ Evaluation Complete!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    run_evaluation()
