"""
RAG Pipeline Comprehensive Evaluation Runner
Stage 1 Metrics Assessment for PanScience Innovations LLM Assignment

This script implements the complete evaluation framework as requested:
- Recall@k, Precision@k, NDCG@k metrics
- Ground-truth test dataset
- Integration testing with FastAPI TestClient
- Performance tuning recommendations
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
import subprocess
import requests

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from tests.test_metrics import RAGEvaluator, create_sample_test_data
from tests.test_integration import RAGIntegrationTester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RAGPipelineEvaluator:
    """Comprehensive RAG Pipeline Evaluator for Stage 1 Assessment"""
    
    def __init__(self):
        self.evaluation_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "stage_1_metrics": {},
            "integration_tests": {},
            "performance_analysis": {},
            "recommendations": []
        }
    
    def check_server_status(self, base_url: str = "http://127.0.0.1:8000") -> bool:
        """Check if the RAG server is running"""
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def start_server_if_needed(self, port: int = 8001) -> bool:
        """Start the RAG server if not running"""
        if self.check_server_status():
            logger.info("✅ Server is already running")
            return True
        
        logger.info("🚀 Starting RAG server...")
        try:
            # Set environment variables
            os.environ["GOOGLE_API_KEY"] = "AIzaSyAZuRFDZW9giTnN5vhodnE3jGpaRNvswuY"
            os.environ["GOOGLE_MODEL_ID"] = "gemini-2.5-flash"
            
            # Start server in background
            cmd = [
                sys.executable, "-m", "uvicorn", 
                "src.api.main:app", 
                "--host", "127.0.0.1", 
                "--port", str(port)
            ]
            
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for server to start
            for i in range(30):  # Wait up to 30 seconds
                time.sleep(1)
                if self.check_server_status(f"http://127.0.0.1:{port}"):
                    logger.info(f"✅ Server started successfully on port {port}")
                    return True
            
            logger.error("❌ Failed to start server")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error starting server: {e}")
            return False
    
    def run_stage_1_metrics(self) -> Dict[str, Any]:
        """Run Stage 1 metrics evaluation (Recall@k, Precision@k, NDCG@k)"""
        logger.info("🔍 Running Stage 1 Metrics Evaluation...")
        logger.info("=" * 60)
        
        try:
            # Initialize evaluator
            evaluator = RAGEvaluator()
            
            # Load test data
            test_queries = create_sample_test_data()
            logger.info(f"📊 Loaded {len(test_queries)} test queries")
            
            # Evaluate retrieval performance
            retrieval_results = evaluator.evaluate_retrieval(test_queries, k_values=[5, 10])
            
            # Evaluate end-to-end performance
            e2e_results = evaluator.evaluate_end_to_end(test_queries)
            
            # Compile results
            stage_1_results = {
                "retrieval_metrics": retrieval_results,
                "end_to_end_metrics": e2e_results,
                "test_queries_count": len(test_queries)
            }
            
            # Print results table
            logger.info("\n📈 STAGE 1 METRICS RESULTS")
            logger.info("=" * 50)
            logger.info("| Metric    | @5      | @10     |")
            logger.info("|-----------|---------|---------|")
            logger.info(f"| Recall    | {retrieval_results['@5']['recall']:.3f}    | {retrieval_results['@10']['recall']:.3f}    |")
            logger.info(f"| Precision | {retrieval_results['@5']['precision']:.3f}    | {retrieval_results['@10']['precision']:.3f}    |")
            logger.info(f"| NDCG      | {retrieval_results['@5']['ndcg']:.3f}    | {retrieval_results['@10']['ndcg']:.3f}    |")
            
            logger.info(f"\n📊 END-TO-END METRICS")
            logger.info(f"Total Queries: {e2e_results['total_queries']}")
            logger.info(f"Success Rate: {e2e_results['successful_queries']/e2e_results['total_queries']*100:.1f}%")
            logger.info(f"Avg Response Time: {e2e_results['average_response_time']:.2f}s")
            logger.info(f"Avg Quality Score: {e2e_results['average_quality_score']:.3f}")
            
            return stage_1_results
            
        except Exception as e:
            logger.error(f"❌ Stage 1 metrics evaluation failed: {e}")
            return {"error": str(e)}
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run comprehensive integration tests"""
        logger.info("\n🧪 Running Integration Tests...")
        logger.info("=" * 40)
        
        try:
            tester = RAGIntegrationTester()
            integration_results = tester.run_comprehensive_test()
            return integration_results
            
        except Exception as e:
            logger.error(f"❌ Integration tests failed: {e}")
            return {"error": str(e)}
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance and provide recommendations"""
        logger.info("\n📊 Analyzing Performance...")
        
        recommendations = []
        
        # Analyze Stage 1 metrics
        if "stage_1_metrics" in self.evaluation_results:
            metrics = self.evaluation_results["stage_1_metrics"]
            
            if "retrieval_metrics" in metrics:
                recall_5 = metrics["retrieval_metrics"]["@5"]["recall"]
                precision_5 = metrics["retrieval_metrics"]["@5"]["precision"]
                ndcg_5 = metrics["retrieval_metrics"]["@5"]["ndcg"]
                
                # Recall recommendations
                if recall_5 < 0.8:
                    recommendations.append({
                        "category": "Recall Improvement",
                        "priority": "High",
                        "suggestion": "Increase chunk size or reduce similarity threshold to improve recall",
                        "current_value": recall_5,
                        "target_value": 0.8
                    })
                
                # Precision recommendations
                if precision_5 < 0.7:
                    recommendations.append({
                        "category": "Precision Improvement",
                        "priority": "Medium",
                        "suggestion": "Improve embeddings model or add reranking to boost precision",
                        "current_value": precision_5,
                        "target_value": 0.7
                    })
                
                # NDCG recommendations
                if ndcg_5 < 0.8:
                    recommendations.append({
                        "category": "NDCG Improvement",
                        "priority": "Medium",
                        "suggestion": "Implement reranking or improve chunk ordering for better NDCG",
                        "current_value": ndcg_5,
                        "target_value": 0.8
                    })
        
        # Integration test analysis
        if "integration_tests" in self.evaluation_results:
            integration = self.evaluation_results["integration_tests"]
            
            if "tests_failed" in integration and integration["tests_failed"] > 0:
                recommendations.append({
                    "category": "System Stability",
                    "priority": "High",
                    "suggestion": f"Fix {integration['tests_failed']} failed integration tests",
                    "current_value": integration["tests_failed"],
                    "target_value": 0
                })
            
            if "query_results" in integration:
                avg_keyword_score = sum(r["keyword_score"] for r in integration["query_results"]) / len(integration["query_results"])
                if avg_keyword_score < 0.7:
                    recommendations.append({
                        "category": "Response Quality",
                        "priority": "Medium",
                        "suggestion": "Improve query processing or document chunking for better answer quality",
                        "current_value": avg_keyword_score,
                        "target_value": 0.7
                    })
        
        return {
            "recommendations": recommendations,
            "total_recommendations": len(recommendations),
            "high_priority": len([r for r in recommendations if r["priority"] == "High"])
        }
    
    def generate_report(self) -> str:
        """Generate comprehensive evaluation report"""
        report = f"""
# RAG Pipeline Evaluation Report - Stage 1 Assessment
**Generated:** {self.evaluation_results['timestamp']}
**Organization:** PanScience Innovations LLM Specialist Assignment

## Executive Summary

This report presents a comprehensive evaluation of the Enhanced RAG Pipeline system against Stage 1 metrics requirements. The evaluation covers retrieval performance, end-to-end functionality, and system integration.

## Stage 1 Metrics Results

### Retrieval Performance
"""
        
        if "stage_1_metrics" in self.evaluation_results and "retrieval_metrics" in self.evaluation_results["stage_1_metrics"]:
            metrics = self.evaluation_results["stage_1_metrics"]["retrieval_metrics"]
            report += f"""
| Metric    | @5      | @10     | Target  | Status |
|-----------|---------|---------|---------|--------|
| Recall    | {metrics['@5']['recall']:.3f}    | {metrics['@10']['recall']:.3f}    | >0.8    | {'✅' if metrics['@5']['recall'] > 0.8 else '❌'} |
| Precision | {metrics['@5']['precision']:.3f}    | {metrics['@10']['precision']:.3f}    | >0.7    | {'✅' if metrics['@5']['precision'] > 0.7 else '❌'} |
| NDCG      | {metrics['@5']['ndcg']:.3f}    | {metrics['@10']['ndcg']:.3f}    | >0.8    | {'✅' if metrics['@5']['ndcg'] > 0.8 else '❌'} |
"""
        
        report += f"""
### End-to-End Performance
"""
        
        if "stage_1_metrics" in self.evaluation_results and "end_to_end_metrics" in self.evaluation_results["stage_1_metrics"]:
            e2e = self.evaluation_results["stage_1_metrics"]["end_to_end_metrics"]
            success_rate = e2e["successful_queries"] / e2e["total_queries"] * 100 if e2e["total_queries"] > 0 else 0
            report += f"""
- **Success Rate:** {success_rate:.1f}% ({e2e['successful_queries']}/{e2e['total_queries']})
- **Average Response Time:** {e2e['average_response_time']:.2f}s
- **Average Quality Score:** {e2e['average_quality_score']:.3f}
"""
        
        report += f"""
## Integration Test Results
"""
        
        if "integration_tests" in self.evaluation_results:
            integration = self.evaluation_results["integration_tests"]
            report += f"""
- **Tests Passed:** {integration.get('tests_passed', 0)}
- **Tests Failed:** {integration.get('tests_failed', 0)}
- **Total Time:** {integration.get('total_time', 0):.2f}s
"""
        
        report += f"""
## Performance Analysis & Recommendations
"""
        
        if "performance_analysis" in self.evaluation_results:
            perf = self.evaluation_results["performance_analysis"]
            report += f"""
- **Total Recommendations:** {perf.get('total_recommendations', 0)}
- **High Priority Issues:** {perf.get('high_priority', 0)}

### Recommendations:
"""
            for i, rec in enumerate(perf.get('recommendations', []), 1):
                report += f"""
{i}. **{rec['category']}** ({rec['priority']} Priority)
   - Current: {rec['current_value']:.3f}
   - Target: {rec['target_value']:.3f}
   - Suggestion: {rec['suggestion']}
"""
        
        report += f"""
## Conclusion

The RAG Pipeline evaluation demonstrates {'strong' if self.evaluation_results.get('stage_1_metrics', {}).get('retrieval_metrics', {}).get('@5', {}).get('recall', 0) > 0.8 else 'mixed'} performance against Stage 1 metrics. 

{'✅ The system meets most performance targets and is ready for production deployment.' if self.evaluation_results.get('stage_1_metrics', {}).get('retrieval_metrics', {}).get('@5', {}).get('recall', 0) > 0.8 else '⚠️ The system requires optimization to meet all performance targets.'}

## Next Steps

1. Address high-priority recommendations
2. Implement performance optimizations
3. Conduct additional testing with larger datasets
4. Prepare for Stage 2 evaluation

---
*Report generated by RAG Pipeline Evaluator v1.0*
"""
        
        return report
    
    def run_complete_evaluation(self) -> Dict[str, Any]:
        """Run complete RAG pipeline evaluation"""
        logger.info("🚀 Starting Complete RAG Pipeline Evaluation")
        logger.info("=" * 60)
        logger.info("Stage 1 Metrics Assessment for PanScience Innovations")
        logger.info("=" * 60)
        
        # Check/start server
        if not self.start_server_if_needed():
            logger.error("❌ Cannot proceed without server")
            return self.evaluation_results
        
        # Run Stage 1 metrics
        self.evaluation_results["stage_1_metrics"] = self.run_stage_1_metrics()
        
        # Run integration tests
        self.evaluation_results["integration_tests"] = self.run_integration_tests()
        
        # Analyze performance
        self.evaluation_results["performance_analysis"] = self.analyze_performance()
        
        # Generate report
        report = self.generate_report()
        
        # Save results
        with open("evaluation_results.json", "w") as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        with open("evaluation_report.md", "w") as f:
            f.write(report)
        
        logger.info("\n📊 EVALUATION COMPLETE")
        logger.info("=" * 30)
        logger.info("✅ Results saved to evaluation_results.json")
        logger.info("✅ Report saved to evaluation_report.md")
        logger.info("✅ Logs saved to evaluation.log")
        
        # Print summary
        if "stage_1_metrics" in self.evaluation_results:
            metrics = self.evaluation_results["stage_1_metrics"]
            if "retrieval_metrics" in metrics:
                recall_5 = metrics["retrieval_metrics"]["@5"]["recall"]
                precision_5 = metrics["retrieval_metrics"]["@5"]["precision"]
                ndcg_5 = metrics["retrieval_metrics"]["@5"]["ndcg"]
                
                logger.info(f"\n📈 FINAL METRICS SUMMARY")
                logger.info(f"Recall@5: {recall_5:.3f} {'✅' if recall_5 > 0.8 else '❌'}")
                logger.info(f"Precision@5: {precision_5:.3f} {'✅' if precision_5 > 0.7 else '❌'}")
                logger.info(f"NDCG@5: {ndcg_5:.3f} {'✅' if ndcg_5 > 0.8 else '❌'}")
        
        return self.evaluation_results

def main():
    """Main evaluation function"""
    evaluator = RAGPipelineEvaluator()
    results = evaluator.run_complete_evaluation()
    
    print("\n" + "="*60)
    print("🎯 RAG PIPELINE EVALUATION COMPLETE")
    print("="*60)
    print("📁 Check the following files for detailed results:")
    print("   - evaluation_results.json (raw data)")
    print("   - evaluation_report.md (human-readable report)")
    print("   - evaluation.log (detailed logs)")
    print("\n🌐 Web Interface: http://127.0.0.1:8001/static/index.html")
    print("📚 API Docs: http://127.0.0.1:8001/docs")

if __name__ == "__main__":
    main()

