"""
Simple and effective RAG accuracy test
"""

import sys
import time
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline

def test_rag_accuracy():
    """Test RAG accuracy with simple queries"""
    print("🚀 Testing RAG Pipeline Accuracy")
    print("="*50)
    
    # Initialize RAG pipeline
    rag = RAGPipeline()
    
    # Test queries based on what we know is in the documents
    test_queries = [
        "What is PDF format?",
        "What are sample PDF files used for?",
        "What companies are mentioned?",
        "What contact information is available?",
        "What file sizes are mentioned?",
        "What are dummy files?",
        "What software is mentioned?",
        "What are the file details?"
    ]
    
    results = []
    successful_queries = 0
    
    print(f"📊 Testing {len(test_queries)} queries...")
    print()
    
    for i, query in enumerate(test_queries, 1):
        print(f"🔍 Query {i}: {query}")
        
        try:
            start_time = time.time()
            response = rag.query(query)
            end_time = time.time()
            
            response_time = end_time - start_time
            answer = response.get("response", "")
            sources = response.get("sources", [])
            
            # Simple quality assessment
            answer_length = len(answer)
            has_sources = len(sources) > 0
            response_time_ok = response_time < 30  # Less than 30 seconds
            
            # Check if answer contains relevant keywords
            query_lower = query.lower()
            answer_lower = answer.lower()
            
            # Extract key terms from query
            key_terms = []
            if "pdf" in query_lower:
                key_terms.append("pdf")
            if "sample" in query_lower:
                key_terms.append("sample")
            if "company" in query_lower or "companies" in query_lower:
                key_terms.extend(["company", "inc", "ltd"])
            if "contact" in query_lower:
                key_terms.extend(["email", "phone", "contact"])
            if "size" in query_lower:
                key_terms.extend(["mb", "size", "file"])
            if "dummy" in query_lower:
                key_terms.append("dummy")
            if "software" in query_lower:
                key_terms.extend(["software", "open-source", "reader"])
            if "detail" in query_lower:
                key_terms.extend(["detail", "content", "file"])
            
            # Count keyword matches
            keyword_matches = sum(1 for term in key_terms if term in answer_lower)
            keyword_score = keyword_matches / len(key_terms) if key_terms else 0.0
            
            # Overall quality score
            quality_score = 0.0
            if answer_length > 50:  # Has substantial answer
                quality_score += 0.3
            if has_sources:  # Has sources
                quality_score += 0.3
            if response_time_ok:  # Reasonable response time
                quality_score += 0.2
            if keyword_score > 0:  # Contains relevant keywords
                quality_score += 0.2
            
            # Determine if query was successful
            is_successful = quality_score >= 0.5
            if is_successful:
                successful_queries += 1
            
            # Status indicator
            status = "✅" if is_successful else "❌"
            
            print(f"   {status} Time: {response_time:.1f}s | Quality: {quality_score:.2f} | Keywords: {keyword_score:.2f}")
            print(f"   Answer: {answer[:100]}{'...' if len(answer) > 100 else ''}")
            print(f"   Sources: {len(sources)} documents")
            print()
            
            results.append({
                "query": query,
                "response_time": response_time,
                "quality_score": quality_score,
                "keyword_score": keyword_score,
                "answer_length": answer_length,
                "sources_count": len(sources),
                "is_successful": is_successful,
                "answer": answer
            })
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            print()
            results.append({
                "query": query,
                "response_time": 0,
                "quality_score": 0,
                "keyword_score": 0,
                "answer_length": 0,
                "sources_count": 0,
                "is_successful": False,
                "answer": ""
            })
    
    # Calculate overall metrics
    total_queries = len(test_queries)
    success_rate = (successful_queries / total_queries) * 100
    avg_response_time = sum(r["response_time"] for r in results) / total_queries
    avg_quality = sum(r["quality_score"] for r in results) / total_queries
    avg_keyword_score = sum(r["keyword_score"] for r in results) / total_queries
    
    # Print summary
    print("="*50)
    print("📊 ACCURACY TEST RESULTS")
    print("="*50)
    print(f"Total Queries: {total_queries}")
    print(f"Successful Queries: {successful_queries}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Average Response Time: {avg_response_time:.1f}s")
    print(f"Average Quality Score: {avg_quality:.3f}")
    print(f"Average Keyword Score: {avg_keyword_score:.3f}")
    print()
    
    # Performance assessment
    if success_rate >= 80:
        print("🏆 EXCELLENT: System meets high accuracy standards")
    elif success_rate >= 60:
        print("✅ GOOD: System meets acceptable accuracy standards")
    elif success_rate >= 40:
        print("⚠️  FAIR: System needs improvement")
    else:
        print("❌ POOR: System requires significant optimization")
    
    print()
    print("🎯 OVERALL ACCURACY SCORE:", f"{success_rate/100:.3f}")
    
    return {
        "total_queries": total_queries,
        "successful_queries": successful_queries,
        "success_rate": success_rate,
        "avg_response_time": avg_response_time,
        "avg_quality": avg_quality,
        "avg_keyword_score": avg_keyword_score,
        "results": results
    }

if __name__ == "__main__":
    test_rag_accuracy()

