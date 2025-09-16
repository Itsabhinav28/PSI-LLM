"""
RAG Pipeline Integration Tests using FastAPI TestClient
Comprehensive end-to-end testing for Stage 1 evaluation
"""

import pytest
import json
import time
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.api.main import app

class RAGIntegrationTester:
    """Comprehensive RAG Pipeline Integration Tester"""
    
    def __init__(self):
        self.client = TestClient(app)
        self.test_documents = []
        self.test_queries = []
        
    def setup_test_environment(self):
        """Set up test environment with sample documents"""
        # Sample test documents (in real scenario, these would be actual files)
        self.test_documents = [
            {
                "filename": "machine_learning_basics.txt",
                "content": """
                Machine Learning Fundamentals
                
                Machine learning is a subset of artificial intelligence (AI) that focuses on the development 
                of algorithms and statistical models that enable computer systems to improve their performance 
                on a specific task through experience, without being explicitly programmed.
                
                Key Concepts:
                1. Supervised Learning: Learning with labeled training data
                2. Unsupervised Learning: Finding patterns in data without labels
                3. Reinforcement Learning: Learning through interaction with environment
                
                Applications include image recognition, natural language processing, 
                recommendation systems, and predictive analytics.
                """
            },
            {
                "filename": "neural_networks_deep_learning.txt",
                "content": """
                Neural Networks and Deep Learning
                
                A neural network is a computing system inspired by biological neural networks. 
                It consists of interconnected nodes (neurons) that process information using 
                a connectionist approach to computation.
                
                Deep Learning:
                - Uses multiple layers of neural networks
                - Excels at pattern recognition in complex data
                - Powers modern AI applications like computer vision and NLP
                
                Training Process:
                1. Forward Propagation: Data flows through the network
                2. Loss Calculation: Compare predicted vs actual output
                3. Backpropagation: Adjust weights to minimize loss
                4. Gradient Descent: Optimize the learning process
                
                Popular architectures include CNNs, RNNs, and Transformers.
                """
            },
            {
                "filename": "data_science_methodology.txt",
                "content": """
                Data Science Methodology
                
                Data science is an interdisciplinary field that uses scientific methods, 
                processes, algorithms and systems to extract knowledge and insights from 
                data in various forms, both structured and unstructured.
                
                CRISP-DM Process:
                1. Business Understanding: Define objectives and requirements
                2. Data Understanding: Collect and explore data
                3. Data Preparation: Clean and transform data
                4. Modeling: Apply machine learning algorithms
                5. Evaluation: Assess model performance
                6. Deployment: Implement the solution
                
                Key Skills: Statistics, Programming, Domain Knowledge, Communication
                Tools: Python, R, SQL, Jupyter, TensorFlow, PyTorch
                """
            }
        ]
        
        # Test queries with expected answers
        self.test_queries = [
            {
                "query": "What is machine learning?",
                "expected_keywords": ["artificial intelligence", "algorithms", "experience", "supervised", "unsupervised"],
                "expected_sources": ["machine_learning_basics.txt"]
            },
            {
                "query": "How do neural networks work?",
                "expected_keywords": ["neurons", "interconnected", "forward propagation", "backpropagation"],
                "expected_sources": ["neural_networks_deep_learning.txt"]
            },
            {
                "query": "What is the CRISP-DM process?",
                "expected_keywords": ["business understanding", "data preparation", "modeling", "deployment"],
                "expected_sources": ["data_science_methodology.txt"]
            },
            {
                "query": "What are the types of machine learning?",
                "expected_keywords": ["supervised", "unsupervised", "reinforcement"],
                "expected_sources": ["machine_learning_basics.txt"]
            },
            {
                "query": "What tools are used in data science?",
                "expected_keywords": ["python", "r", "sql", "tensorflow", "pytorch"],
                "expected_sources": ["data_science_methodology.txt"]
            }
        ]
    
    def test_health_check(self):
        """Test system health check endpoint"""
        response = self.client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print("✅ Health check passed")
    
    def test_document_upload(self):
        """Test document upload functionality"""
        for doc in self.test_documents:
            # Create a temporary file-like object
            files = {"file": (doc["filename"], doc["content"], "text/plain")}
            response = self.client.post("/upload", files=files)
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert "document_id" in data["data"]
            print(f"✅ Document upload successful: {doc['filename']}")
    
    def test_document_processing(self):
        """Test document processing pipeline"""
        response = self.client.post("/documents/process-all")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        print("✅ Document processing completed")
    
    def test_query_processing(self):
        """Test query processing and response generation"""
        results = []
        
        for i, query_data in enumerate(self.test_queries):
            query = query_data["query"]
            expected_keywords = query_data["expected_keywords"]
            
            response = self.client.post("/query", json={"question": query})
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "success"
            assert "response" in data["data"]
            assert "sources" in data["data"]
            
            # Check response quality
            answer = data["data"]["response"].lower()
            sources = data["data"]["sources"]
            
            # Count how many expected keywords are present
            keyword_matches = sum(1 for keyword in expected_keywords if keyword.lower() in answer)
            keyword_score = keyword_matches / len(expected_keywords)
            
            results.append({
                "query": query,
                "keyword_score": keyword_score,
                "response_length": len(answer),
                "sources_count": len(sources),
                "response_time": data["data"].get("processing_time", 0)
            })
            
            print(f"✅ Query {i+1}: {query[:30]}... | Keyword Score: {keyword_score:.2f}")
        
        return results
    
    def test_document_search(self):
        """Test document search functionality"""
        search_queries = [
            "machine learning algorithms",
            "neural network training",
            "data science process"
        ]
        
        for query in search_queries:
            response = self.client.get(f"/documents/search?query={query}")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "success"
            assert "documents" in data["data"]
            print(f"✅ Search successful for: {query}")
    
    def test_system_statistics(self):
        """Test system statistics and monitoring"""
        response = self.client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "total_documents" in data["data"]
        assert "total_chunks" in data["data"]
        print("✅ System statistics retrieved")
    
    def test_analytics(self):
        """Test analytics and performance metrics"""
        response = self.client.get("/analytics")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        print("✅ Analytics data retrieved")
    
    def run_comprehensive_test(self):
        """Run comprehensive integration test suite"""
        print("🚀 Starting RAG Pipeline Integration Tests")
        print("=" * 60)
        
        # Setup
        self.setup_test_environment()
        
        # Run tests
        test_results = {
            "start_time": time.time(),
            "tests_passed": 0,
            "tests_failed": 0,
            "query_results": []
        }
        
        try:
            # Health check
            self.test_health_check()
            test_results["tests_passed"] += 1
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            test_results["tests_failed"] += 1
        
        try:
            # Document upload
            self.test_document_upload()
            test_results["tests_passed"] += 1
        except Exception as e:
            print(f"❌ Document upload failed: {e}")
            test_results["tests_failed"] += 1
        
        try:
            # Document processing
            self.test_document_processing()
            test_results["tests_passed"] += 1
        except Exception as e:
            print(f"❌ Document processing failed: {e}")
            test_results["tests_failed"] += 1
        
        try:
            # Query processing
            query_results = self.test_query_processing()
            test_results["query_results"] = query_results
            test_results["tests_passed"] += 1
        except Exception as e:
            print(f"❌ Query processing failed: {e}")
            test_results["tests_failed"] += 1
        
        try:
            # Document search
            self.test_document_search()
            test_results["tests_passed"] += 1
        except Exception as e:
            print(f"❌ Document search failed: {e}")
            test_results["tests_failed"] += 1
        
        try:
            # System statistics
            self.test_system_statistics()
            test_results["tests_passed"] += 1
        except Exception as e:
            print(f"❌ System statistics failed: {e}")
            test_results["tests_failed"] += 1
        
        try:
            # Analytics
            self.test_analytics()
            test_results["tests_passed"] += 1
        except Exception as e:
            print(f"❌ Analytics failed: {e}")
            test_results["tests_failed"] += 1
        
        # Calculate results
        test_results["end_time"] = time.time()
        test_results["total_time"] = test_results["end_time"] - test_results["start_time"]
        
        # Print summary
        print("\n📊 INTEGRATION TEST RESULTS")
        print("=" * 40)
        print(f"Tests Passed: {test_results['tests_passed']}")
        print(f"Tests Failed: {test_results['tests_failed']}")
        print(f"Total Time: {test_results['total_time']:.2f}s")
        
        if test_results["query_results"]:
            avg_keyword_score = sum(r["keyword_score"] for r in test_results["query_results"]) / len(test_results["query_results"])
            avg_response_time = sum(r["response_time"] for r in test_results["query_results"]) / len(test_results["query_results"])
            print(f"Average Keyword Score: {avg_keyword_score:.3f}")
            print(f"Average Response Time: {avg_response_time:.2f}s")
        
        # Save results
        with open("integration_test_results.json", "w") as f:
            json.dump(test_results, f, indent=2)
        
        print(f"\n💾 Results saved to integration_test_results.json")
        print("✅ Integration tests complete!")
        
        return test_results

def run_integration_tests():
    """Run integration tests"""
    tester = RAGIntegrationTester()
    return tester.run_comprehensive_test()

if __name__ == "__main__":
    run_integration_tests()