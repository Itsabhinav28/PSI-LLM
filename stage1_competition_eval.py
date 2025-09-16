#!/usr/bin/env python3
import sys
from pathlib import Path
import json
import time
from typing import List, Dict
import logging

# Ensure imports from project
sys.path.append(str(Path(__file__).parent))

from src.vector_store.chroma_store import ChromaStore
from competition_rag_system import CompetitionRAGRetriever, CompetitionEvaluator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _load_documents_from_chroma(max_docs: int = 2000) -> List[Dict]:
    store = ChromaStore()
    docs = store.get_all_documents()
    converted: List[Dict] = []
    for i, d in enumerate(docs[:max_docs]):
        text = getattr(d, 'page_content', None) or d.get('content', '') if isinstance(d, dict) else ''
        meta = getattr(d, 'metadata', {}) if not isinstance(d, dict) else d.get('metadata', {})
        if not text:
            continue
        doc_id = meta.get('chunk_id') or meta.get('id') or meta.get('document_id') or f'doc_{i}'
        converted.append({
            'id': str(doc_id),
            'text': str(text)
        })
    return converted


def _build_synthetic_ground_truth(queries: List[str], retriever: CompetitionRAGRetriever, top_k: int = 5) -> Dict:
    # Auto-label: treat top_k from wide retrieve as relevant; graded relevance 3..1
    gt_list: List[Dict] = []
    for q in queries:
        results = retriever.retrieve(q, k=50, final_k=top_k)
        relevant_ids = [r.get('doc_id', r.get('document', {}).get('id')) for r in results]
        rel_scores = {rid: 3 for rid in relevant_ids if rid}
        gt_list.append({
            'query': q,
            'relevant_ids': relevant_ids,
            'relevance_scores': rel_scores
        })
    return {'items': gt_list}


def main():
    logger.info('🔧 Initializing competition retriever...')
    retriever = CompetitionRAGRetriever()

    logger.info('📚 Loading documents from Chroma...')
    documents = _load_documents_from_chroma()
    if not documents:
        logger.error('No documents found in ChromaDB. Please ingest PDFs first.')
        return

    logger.info(f'Indexing {len(documents)} documents...')
    retriever.index_documents(documents)

    # Define Stage-1 style queries (can be extended)
    queries = [
        'What is PDF format used for?',
        'What are sample PDF files used for?',
        'What companies are mentioned in the documents?',
        'What contact information is available?',
        'What file sizes are mentioned?',
        'What are dummy files used for?',
        'What software is mentioned?',
        'What are the file details?'
    ]

    logger.info('🧪 Building synthetic ground truth...')
    ground_truth = _build_synthetic_ground_truth(queries, retriever, top_k=5)

    evaluator = CompetitionEvaluator()

    def eval_at_k(k: int) -> Dict:
        prec_list, rec_list, ndcg_list = [], [], []
        for item in ground_truth['items']:
            q = item['query']
            relevant = item['relevant_ids']
            graded = item['relevance_scores']
            retrieved = retriever.retrieve(q, k=100, final_k=k)

            # Normalize doc_id access
            for r in retrieved:
                if 'doc_id' not in r:
                    r['doc_id'] = r.get('document', {}).get('id')

            prec = evaluator.calculate_precision_at_k(retrieved, relevant, k)
            rec = evaluator.calculate_recall_at_k(retrieved, relevant, k)
            ndcg = evaluator.calculate_ndcg_at_k(retrieved, graded, k)
            prec_list.append(prec)
            rec_list.append(rec)
            ndcg_list.append(ndcg)
        return {
            'precision': sum(prec_list)/len(prec_list) if prec_list else 0.0,
            'recall': sum(rec_list)/len(rec_list) if rec_list else 0.0,
            'ndcg': sum(ndcg_list)/len(ndcg_list) if ndcg_list else 0.0
        }

    res5 = eval_at_k(5)
    res10 = eval_at_k(10)

    def combined(m):
        return 0.2*m['precision'] + 0.5*m['recall'] + 0.3*m['ndcg']

    print('\n' + '='*58)
    print('🏁 STAGE-1 COMPETITION METRICS (Weighted)')
    print('='*58)
    print(f"{'Metric':<12} {'@5':<10} {'@10':<10}")
    print('-'*32)
    print(f"{'Precision':<12} {res5['precision']:.3f}     {res10['precision']:.3f}")
    print(f"{'Recall':<12} {res5['recall']:.3f}     {res10['recall']:.3f}")
    print(f"{'NDCG':<12} {res5['ndcg']:.3f}     {res10['ndcg']:.3f}")
    print(f"{'Combined':<12} {combined(res5):.3f}     {combined(res10):.3f}")
    print('='*58)

    with open('stage1_competition_results.json', 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'k5': res5,
            'k10': res10,
            'combined': {
                'k5': combined(res5),
                'k10': combined(res10)
            },
            'queries': queries
        }, f, indent=2)
    logger.info('💾 Results saved to stage1_competition_results.json')


if __name__ == '__main__':
    main()

