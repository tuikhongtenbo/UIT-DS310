"""
Stage 1 Evaluation Script: Test Retrievers on Private Test Set
Generate predictions in JSON format for BM25, Dense, and Hybrid retrievers
Output format: [{"qid": <int>, "relevant_laws": [<int>, ...]}, ...]
Also creates candidate index file for reranker stage 2
"""

import os
import json
import argparse
import numpy as np
import re
from typing import List, Dict, Tuple, Any, Union
from tqdm import tqdm
import torch

# Add parent directory to path
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Import modules from src/
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.ensemble import EnsembleRetriever
from src.embeddings.chroma_index import ChromaIndex
from src.embeddings.embedder import VietnameseEmbedder
from src.utils.logger import setup_logger

logger = setup_logger("stage_1_eval")


# ==========================================
# Convert AIDs to Int
# ==========================================
def convert_aids_to_int(aids: List[str]) -> List[int]:
    """Convert list of aid strings/ints to integers, filtering out invalid values"""
    result = []
    for aid in aids:
        if isinstance(aid, int):
            result.append(aid)
            continue
        
        aid_str = str(aid).strip()
        try:
            result.append(int(aid_str))
        except ValueError:
            numbers = re.findall(r'\d+', aid_str)
            if numbers:
                result.append(int(numbers[-1]))
    return result


def main():
    parser = argparse.ArgumentParser(description='Evaluate retrievers on test set')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Path to test JSON file (e.g., private_test.json)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for prediction JSON files')
    parser.add_argument('--bm25_index', type=str, required=True,
                        help='Path to BM25 index pickle file')
    parser.add_argument('--chroma_db_retriever', type=str, required=True,
                        help='Path to ChromaDB retriever database')
    parser.add_argument('--chroma_db_reranker', type=str, required=True,
                        help='Path to ChromaDB reranker database (for candidate index)')
    parser.add_argument('--collection_retriever', type=str, default='retriever_legal_articles',
                        help='Collection name for retriever')
    parser.add_argument('--collection_reranker', type=str, default='reranker_legal_articles',
                        help='Collection name for reranker')
    parser.add_argument('--embedding_model', type=str, default='AITeamVN/Vietnamese_Embedding',
                        help='Embedding model name')
    parser.add_argument('--top_k', type=int, default=200,
                        help='Top-k output from retriever (number of relevant_laws per query)')
    parser.add_argument('--bm25_weight', type=float, default=0.3,
                        help='BM25 weight for hybrid retrieval')
    parser.add_argument('--dense_weight', type=float, default=0.7,
                        help='Dense weight for hybrid retrieval')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda:0, cuda:1, cpu, etc.)')
    
    args = parser.parse_args()
    
    # GPU Configuration
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        if args.device:
            DEVICE = args.device
        else:
            DEVICE = "cuda:0"
        logger.info(f"\nUsing device: {DEVICE}")
    else:
        DEVICE = "cpu"
        logger.info("No GPU available, using CPU")
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Top-k: {args.top_k}")
    logger.info(f"  BM25 weight: {args.bm25_weight}, Dense weight: {args.dense_weight}")
    logger.info(f"  Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data
    logger.info(f"\nLoading test data from {args.test_file}...")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Load BM25 index
    logger.info(f"Loading BM25 index from {args.bm25_index}...")
    bm25_retriever = BM25Retriever(embedder=None)
    bm25_retriever.load(args.bm25_index)
    
    # Load ChromaDB for retriever
    logger.info(f"Loading ChromaDB retriever from {args.chroma_db_retriever}...")
    chroma_index_retriever = ChromaIndex(
        persist_directory=args.chroma_db_retriever,
        collection_name=args.collection_retriever
    )
    
    # Load ChromaDB for reranker (to get document content for candidate index)
    logger.info(f"Loading ChromaDB reranker from {args.chroma_db_reranker}...")
    chroma_index_reranker = ChromaIndex(
        persist_directory=args.chroma_db_reranker,
        collection_name=args.collection_reranker
    )
    
    # Load embedding model
    logger.info(f"Loading embedding model: {args.embedding_model}...")
    embedder = VietnameseEmbedder(model_name=args.embedding_model, device=DEVICE)
    
    # Initialize retrievers
    logger.info("Initializing retrievers...")
    dense_retriever = DenseRetriever(chroma_index_retriever, embedder)
    hybrid_retriever = EnsembleRetriever(
        bm25_retriever, 
        dense_retriever, 
        weights=(args.bm25_weight, args.dense_weight)
    )
    logger.info(f"Hybrid retriever initialized: BM25 weight={args.bm25_weight}, Dense weight={args.dense_weight}")
    
    # Processing
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing {len(test_data)} queries...")
    logger.info(f"{'='*80}")
    
    # Storage for results for each method
    results = {
        'bm25': [],
        'dense': [],
        'hybrid': []
    }
    
    # Storage for candidate index (for reranker stage 2)
    # Format: {qid: [{"aid": ..., "content": ..., "text": ...}, ...]}
    candidate_index = {}
    
    for idx, sample in enumerate(tqdm(test_data, desc="Processing")):
        qid = sample['qid']
        question = sample.get('question', '')
        
        if not question:
            # Empty question, add empty results for all methods
            for method in results.keys():
                results[method].append({
                    'qid': qid,
                    'relevant_laws': []
                })
            candidate_index[qid] = []
            continue
        
        # Step 1: Retrieve with BM25
        bm25_aids = []
        try:
            bm25_results = bm25_retriever.retrieve(question, top_k=args.top_k)
            bm25_aids = [str(aid) for aid, _ in bm25_results]
        except Exception as e:
            logger.error(f"Error in BM25 retrieval for sample {qid}: {e}")
        
        # Step 2: Retrieve with Dense
        dense_aids = []
        try:
            dense_results = dense_retriever.retrieve(question, top_k=args.top_k)
            dense_aids = [str(aid) for aid, _ in dense_results]
        except Exception as e:
            logger.error(f"Error in Dense retrieval for sample {qid}: {e}")
        
        # Step 3: Retrieve with Hybrid
        hybrid_aids = []
        try:
            hybrid_results = hybrid_retriever.retrieve(question, top_k=args.top_k)
            hybrid_aids = [str(aid) for aid, _ in hybrid_results]
        except Exception as e:
            logger.error(f"Error in Hybrid retrieval for sample {qid}: {e}")
        
        # Step 4: Get documents from ChromaDB reranker for candidate index
        # Use hybrid results for candidate index (best candidates)
        candidate_docs = []
        seen_aids = set()
        
        for aid in hybrid_aids:
            if aid in seen_aids:
                continue
            seen_aids.add(aid)
            
            try:
                query_results = chroma_index_reranker.collection.get(
                    where={"aid": str(aid)},
                    limit=1
                )
                if query_results['ids']:
                    doc_id = query_results['ids'][0]
                    doc_data = chroma_index_reranker.collection.get(
                        ids=[doc_id], 
                        include=['metadatas', 'documents']
                    )
                    if doc_data['documents']:
                        candidate_docs.append({
                            'aid': str(aid),
                            'content': doc_data['documents'][0],
                            'text': doc_data['documents'][0]
                        })
            except Exception as e:
                continue
        
        # Store candidate index
        candidate_index[qid] = candidate_docs
        
        # Step 5: Convert to results format
        # BM25 results
        bm25_laws = convert_aids_to_int(bm25_aids)
        bm25_laws = list(dict.fromkeys(bm25_laws))  # Remove duplicates
        results['bm25'].append({
            'qid': qid,
            'relevant_laws': bm25_laws[:args.top_k]
        })
        
        # Dense results
        dense_laws = convert_aids_to_int(dense_aids)
        dense_laws = list(dict.fromkeys(dense_laws))  # Remove duplicates
        results['dense'].append({
            'qid': qid,
            'relevant_laws': dense_laws[:args.top_k]
        })
        
        # Hybrid results
        hybrid_laws = convert_aids_to_int(hybrid_aids)
        hybrid_laws = list(dict.fromkeys(hybrid_laws))  # Remove duplicates
        results['hybrid'].append({
            'qid': qid,
            'relevant_laws': hybrid_laws[:args.top_k]
        })
    
    # Sort results by qid for all methods
    for method in results.keys():
        results[method].sort(key=lambda x: x.get('qid', 0))
    
    # Validate: ensure all qids from input are present in results
    input_qids = set(item.get('qid') for item in test_data)
    for method in results.keys():
        result_qids = set(result.get('qid') for result in results[method])
        missing_qids = input_qids - result_qids
        if missing_qids:
            for qid in sorted(missing_qids):
                results[method].append({
                    'qid': qid,
                    'relevant_laws': []
                })
            results[method].sort(key=lambda x: x.get('qid', 0))
    
    # Ensure candidate_index has all qids
    for qid in input_qids:
        if qid not in candidate_index:
            candidate_index[qid] = []
    
    # Save output files
    output_files = {
        'bm25': os.path.join(args.output_dir, f'results_bm25_k{args.top_k}.json'),
        'dense': os.path.join(args.output_dir, f'results_dense_k{args.top_k}.json'),
        'hybrid': os.path.join(args.output_dir, f'results_hybrid_k{args.top_k}.json')
    }
    
    logger.info(f"\n{'='*80}")
    logger.info("Saving output files...")
    logger.info(f"{'='*80}")
    
    for method, output_file in output_files.items():
        output_path = output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results[method], f, ensure_ascii=False, indent=2)
        
        # Statistics
        total_queries = len(results[method])
        queries_with_results = sum(1 for r in results[method] if len(r.get('relevant_laws', [])) > 0)
        avg_laws_per_query = np.mean([len(r.get('relevant_laws', [])) for r in results[method]])
        
        logger.info(f"\n{method.upper()}:")
        logger.info(f"  Output file: {output_path}")
        logger.info(f"  Total queries: {total_queries}")
        logger.info(f"  Queries with results: {queries_with_results} ({queries_with_results/total_queries*100:.1f}%)")
        logger.info(f"  Avg laws per query: {avg_laws_per_query:.2f}")
    
    # Save candidate index for reranker stage 2
    candidate_index_path = os.path.join(args.output_dir, 'candidate_index_for_reranker.json')
    logger.info(f"\n{'='*80}")
    logger.info("Saving candidate index for reranker stage 2...")
    logger.info(f"{'='*80}")
    
    # Convert to list format for easier loading: [{qid: ..., candidates: [...]}, ...]
    candidate_index_list = [
        {
            'qid': qid,
            'candidates': candidates
        }
        for qid, candidates in sorted(candidate_index.items())
    ]
    
    with open(candidate_index_path, 'w', encoding='utf-8') as f:
        json.dump(candidate_index_list, f, ensure_ascii=False, indent=2)
    
    total_candidates = sum(len(candidates) for candidates in candidate_index.values())
    avg_candidates = np.mean([len(candidates) for candidates in candidate_index.values()])
    
    logger.info(f"\nCandidate Index:")
    logger.info(f"  Output file: {candidate_index_path}")
    logger.info(f"  Total queries: {len(candidate_index)}")
    logger.info(f"  Total candidates: {total_candidates}")
    logger.info(f"  Avg candidates per query: {avg_candidates:.2f}")
    logger.info(f"  Format: [{{\"qid\": <int>, \"candidates\": [{{\"aid\": <str>, \"content\": <str>, \"text\": <str>}}, ...]}}, ...]")
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Stage 1 evaluation completed!")
    logger.info(f"Output files ready for submission:")
    for method, output_file in output_files.items():
        logger.info(f"  - {output_file}")
    logger.info(f"\nCandidate index ready for stage 2 reranker:")
    logger.info(f"  - {candidate_index_path}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()