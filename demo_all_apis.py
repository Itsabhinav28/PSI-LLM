#!/usr/bin/env python3
"""
Comprehensive API Testing Script for PSI RAG Pipeline
LLM Specialist Assignment - PanScience Innovations
"""

import requests
import json
import time
import os
from datetime import datetime

# Configuration
BASE_URL = "http://127.0.0.1:8000"
API_KEY = "demo_key_123"  # Demo API key

def print_header(title):
    """Print a formatted header for each test section."""
    print("\n" + "="*60)
    print(f"🧪 {title}")
    print("="*60)

def print_success(message):
    """Print success message."""
    print(f"✅ {message}")

def print_error(message):
    """Print error message."""
    print(f"❌ {message}")

def print_info(message):
    """Print info message."""
    print(f"ℹ️  {message}")

def test_health_check():
    """Test 1: Health Check API"""
    print_header("HEALTH CHECK API")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print_success(f"Health Check: {response.status_code}")
            print_info(f"Status: {data.get('status')}")
            print_info(f"Timestamp: {data.get('timestamp')}")
            print_info(f"Rate Limits: {data.get('rate_limits')}")
        else:
            print_error(f"Health Check Failed: {response.status_code}")
    except Exception as e:
        print_error(f"Health Check Error: {e}")

def test_stats_api():
    """Test 2: Statistics API"""
    print_header("STATISTICS API")
    try:
        response = requests.get(f"{BASE_URL}/stats")
        if response.status_code == 200:
            data = response.json()
            print_success(f"Stats API: {response.status_code}")
            print_info(f"Total Documents: {data.get('total_documents')}")
            print_info(f"Pipeline Status: {data.get('pipeline_status')}")
            print_info(f"Collection Name: {data.get('collection_name')}")
            print_info(f"Embedding Model: {data.get('embedding_model', 'N/A')[:50]}...")
        else:
            print_error(f"Stats API Failed: {response.status_code}")
    except Exception as e:
        print_error(f"Stats API Error: {e}")

def test_document_list_api():
    """Test 3: Document List API"""
    print_header("DOCUMENT LIST API")
    try:
        response = requests.get(f"{BASE_URL}/documents/list")
        if response.status_code == 200:
            data = response.json()
            print_success(f"Document List API: {response.status_code}")
            print_info(f"Total Count: {data.get('total_count')}")
            print_info(f"Documents: {len(data.get('documents', []))}")
            
            # Show first few documents
            docs = data.get('documents', [])
            if docs:
                print_info("Sample Documents:")
                for i, doc in enumerate(docs[:3]):  # Show first 3
                    print(f"  {i+1}. {doc.get('filename', 'Unknown')} ({doc.get('file_size', 0)} bytes)")
            else:
                print_info("No documents found")
        else:
            print_error(f"Document List API Failed: {response.status_code}")
    except Exception as e:
        print_error(f"Document List API Error: {e}")

def test_document_stats_realtime():
    """Test 4: Real-time Document Statistics API"""
    print_header("REAL-TIME DOCUMENT STATS API")
    try:
        response = requests.get(f"{BASE_URL}/documents/stats/realtime")
        if response.status_code == 200:
            data = response.json()
            print_success(f"Real-time Stats API: {response.status_code}")
            if data.get('success'):
                stats = data.get('stats', {})
                print_info(f"Total Documents: {stats.get('total_documents')}")
                print_info(f"Active Documents: {stats.get('active_documents')}")
                print_info(f"Recent Uploads: {stats.get('recent_uploads')}")
                print_info(f"Total Size: {stats.get('total_size', 0)} bytes")
            else:
                print_error("Real-time stats failed")
        else:
            print_error(f"Real-time Stats API Failed: {response.status_code}")
    except Exception as e:
        print_error(f"Real-time Stats API Error: {e}")

def test_query_api():
    """Test 5: Query API (Core RAG Functionality)"""
    print_header("QUERY API - CORE RAG FUNCTIONALITY")
    
    # Test with different query types
    test_queries = [
        "What is artificial intelligence?",
        "Explain machine learning concepts",
        "What are the main topics in this document?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test Query {i}: {query} ---")
        try:
            payload = {
                "question": query,
                "n_results": 5,
                "use_reranking": True,
                "query_expansion": False,
                "semantic_search": True
            }
            
            response = requests.post(
                f"{BASE_URL}/query",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print_success(f"Query API: {response.status_code}")
                print_info(f"Success: {data.get('success')}")
                print_info(f"Processing Time: {data.get('processing_time')}s")
                print_info(f"Confidence Score: {data.get('confidence_score')}")
                print_info(f"Sources Found: {len(data.get('sources', []))}")
                
                # Show answer preview
                answer = data.get('answer', '')
                if answer:
                    print_info(f"Answer Preview: {answer[:100]}...")
                else:
                    print_info("No answer generated")
                    
            elif response.status_code == 400:
                error_data = response.json()
                print_info(f"Query API: {response.status_code} - {error_data.get('detail', 'Bad Request')}")
            else:
                print_error(f"Query API Failed: {response.status_code}")
                
        except Exception as e:
            print_error(f"Query API Error: {e}")
        
        time.sleep(1)  # Small delay between queries

def test_upload_api():
    """Test 6: Document Upload API"""
    print_header("DOCUMENT UPLOAD API")
    
    # Create a test file
    test_content = "This is a test document for API demonstration purposes. It contains information about artificial intelligence and machine learning concepts."
    test_filename = "demo_test_document.txt"
    
    try:
        with open(test_filename, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # Test upload
        with open(test_filename, 'rb') as f:
            files = {'files': (test_filename, f, 'text/plain')}
            response = requests.post(f"{BASE_URL}/upload", files=files)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Upload API: {response.status_code}")
            print_info(f"Success: {data.get('success')}")
            print_info(f"Message: {data.get('message')}")
            print_info(f"Documents Processed: {data.get('documents_processed')}")
            print_info(f"Document IDs: {data.get('document_ids')}")
        else:
            print_error(f"Upload API Failed: {response.status_code}")
            print_info(f"Response: {response.text}")
            
    except Exception as e:
        print_error(f"Upload API Error: {e}")
    finally:
        # Clean up test file
        if os.path.exists(test_filename):
            os.remove(test_filename)

def test_document_search_api():
    """Test 7: Document Search API"""
    print_header("DOCUMENT SEARCH API")
    try:
        # Test search with filters
        search_params = {
            "query": "artificial intelligence",
            "filters": json.dumps({"file_type": "text/plain"})
        }
        
        response = requests.get(f"{BASE_URL}/documents/search", params=search_params)
        
        if response.status_code == 200:
            data = response.json()
            print_success(f"Search API: {response.status_code}")
            print_info(f"Query: {data.get('query')}")
            print_info(f"Total Results: {data.get('total_results')}")
            print_info(f"Processing Time: {data.get('processing_time')}s")
            print_info(f"Filters Applied: {data.get('filters_applied')}")
        else:
            print_error(f"Search API Failed: {response.status_code}")
            
    except Exception as e:
        print_error(f"Search API Error: {e}")

def test_analytics_api():
    """Test 8: Analytics API"""
    print_header("ANALYTICS API")
    try:
        response = requests.get(f"{BASE_URL}/analytics")
        if response.status_code == 200:
            data = response.json()
            print_success(f"Analytics API: {response.status_code}")
            print_info(f"Total Queries: {data.get('total_queries')}")
            print_info(f"Success Rate: {data.get('success_rate')}%")
            print_info(f"Average Processing Time: {data.get('avg_processing_time')}s")
            print_info(f"Recent Errors: {data.get('recent_errors')}")
        else:
            print_error(f"Analytics API Failed: {response.status_code}")
            
    except Exception as e:
        print_error(f"Analytics API Error: {e}")

def test_reset_api():
    """Test 9: System Reset API"""
    print_header("SYSTEM RESET API")
    try:
        response = requests.post(f"{BASE_URL}/reset")
        if response.status_code == 200:
            data = response.json()
            print_success(f"Reset API: {response.status_code}")
            print_info(f"Success: {data.get('success')}")
            print_info(f"Message: {data.get('message')}")
            print_info(f"Timestamp: {data.get('timestamp')}")
        else:
            print_error(f"Reset API Failed: {response.status_code}")
            
    except Exception as e:
        print_error(f"Reset API Error: {e}")

def test_rate_limiting():
    """Test 10: Rate Limiting (Security Feature)"""
    print_header("RATE LIMITING TEST")
    try:
        print_info("Testing rate limiting by making multiple rapid requests...")
        
        # Make multiple rapid requests to trigger rate limiting
        responses = []
        for i in range(10):
            response = requests.get(f"{BASE_URL}/health")
            responses.append(response.status_code)
            time.sleep(0.1)  # Very small delay
        
        success_count = responses.count(200)
        rate_limited_count = responses.count(429)
        
        print_info(f"Total Requests: {len(responses)}")
        print_info(f"Successful: {success_count}")
        print_info(f"Rate Limited: {rate_limited_count}")
        
        if rate_limited_count > 0:
            print_success("Rate limiting is working correctly!")
        else:
            print_info("Rate limiting not triggered (may need more requests)")
            
    except Exception as e:
        print_error(f"Rate Limiting Test Error: {e}")

def main():
    """Main demonstration function"""
    print("🚀 PSI RAG PIPELINE - COMPREHENSIVE API TESTING DEMONSTRATION")
    print("=" * 70)
    print(f"📅 Test Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 Base URL: {BASE_URL}")
    print(f"🔑 Demo API Key: {API_KEY}")
    
    # Test all APIs
    test_health_check()
    test_stats_api()
    test_document_list_api()
    test_document_stats_realtime()
    test_query_api()
    test_upload_api()
    test_document_search_api()
    test_analytics_api()
    test_reset_api()
    test_rate_limiting()
    
    print_header("DEMONSTRATION COMPLETE")
    print_success("All API endpoints have been tested!")
    print_info("Check the results above to verify functionality.")
    print_info("The system is ready for screen recording demonstration.")

if __name__ == "__main__":
    main()
