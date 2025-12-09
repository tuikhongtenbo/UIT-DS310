"""
Stage 2 Evaluation Script: Test Rerankers on Private Test Set
Generate predictions in JSON format for GTE, BGE, Jina, and Ensemble rerankers
Output format: [{"qid": <int>, "relevant_laws": [<int>, ...]}, ...]
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
from src.reranking.single_reranker import SingleReranker
from src.reranking.ensemble_reranker import EnsembleReranker
from src.embeddings.chroma_index import ChromaIndex
from src.embeddings.embedder import VietnameseEmbedder
from src.utils.logger import setup_logger

logger = setup_logger("stage_2_eval")


# ==========================================
# Evaluation Metrics
# ==========================================
def calculate_precision_at_k(predicted: List[int], relevant: List[int], k: int = 20) -> float:
    """Calculate Precision@k"""
    if k == 0:
        return 0.0
    predicted_set = set(predicted[:k])
    relevant_set = set(relevant)
    if len(predicted_set) == 0:
        return 0.0
    relevant_retrieved = len(predicted_set & relevant_set)
    precision = relevant_retrieved / len(predicted_set)
    return precision


def calculate_recall_at_k(predicted: List[int], relevant: List[int], k: int = 20) -> float:
    """Calculate Recall@k"""
    if k == 0:
        return 0.0
    predicted_set = set(predicted[:k])
    relevant_set = set(relevant)
    if len(relevant_set) == 0:
        return 0.0
    relevant_retrieved = len(predicted_set & relevant_set)
    recall = relevant_retrieved / len(relevant_set)
    return recall


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


# ==========================================
# Clear GPU Cache
# ==========================================
def clear_gpu_cache():
    """Clear GPU cache to free memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("GPU cache cleared")


# ==========================================
# Evaluate Single Model on Train Set
# ==========================================
def evaluate_model_on_train(
    model_name: str,
    reranker: SingleReranker,
    train_data: List[Dict],
    hybrid_retriever: EnsembleRetriever,
    chroma_index_reranker: ChromaIndex,
    retriever_top_k: int,
    reranker_top_k: int,
    eval_k: int = 20
) -> Dict[str, float]:
    """
    Evaluate a single reranker model on train set.
    Returns metrics dictionary with 'precision', 'recall', 'f1', 'score' (average of P@k and R@k).
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating {model_name.upper()} on train set...")
    logger.info(f"{'='*80}")
    
    precisions = []
    recalls = []
    
    for sample in tqdm(train_data, desc=f"Eval {model_name}"):
        qid = sample.get('qid')
        question = sample.get('question', '')
        relevant_laws = [int(law_id) for law_id in sample.get('relevant_laws', [])]
        
        if not question or not relevant_laws:
            continue
        
        # Step 1: Retrieve candidates
        try:
            retrieved_results = hybrid_retriever.retrieve(question, top_k=retriever_top_k)
            retrieved_aids = [str(aid) for aid, _ in retrieved_results]
        except Exception as e:
            logger.error(f"Error in retrieval for sample {qid}: {e}")
            continue
        
        if not retrieved_aids:
            continue
        
        # Step 2: Get documents from ChromaDB
        candidate_docs = []
        for aid in retrieved_aids:
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
        
        if not candidate_docs:
            continue
        
        # Step 3: Rerank
        try:
            reranked_results = reranker.rerank(
                question, 
                candidate_docs, 
                top_k=reranker_top_k
            )
            predicted_laws = convert_aids_to_int([str(aid) for aid, _ in reranked_results])
            
            # Calculate metrics
            precision = calculate_precision_at_k(predicted_laws, relevant_laws, k=eval_k)
            recall = calculate_recall_at_k(predicted_laws, relevant_laws, k=eval_k)
            
            precisions.append(precision)
            recalls.append(recall)
        except Exception as e:
            logger.error(f"Error in reranking for sample {qid}: {e}")
            continue
    
    if len(precisions) == 0:
        logger.warning(f"No valid samples for {model_name}")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'score': 0.0}
    
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
    score = (avg_precision + avg_recall) / 2  # Average of P@k and R@k
    
    logger.info(f"\n{model_name.upper()} Results:")
    logger.info(f"  P@{eval_k}: {avg_precision:.4f}")
    logger.info(f"  R@{eval_k}: {avg_recall:.4f}")
    logger.info(f"  F1: {f1:.4f}")
    logger.info(f"  Score (avg P@k + R@k): {score:.4f}")
    
    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': f1,
        'score': score
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate rerankers on test set')
    parser.add_argument('--test_file', type=str, required=True,
                        help='Path to test JSON file (e.g., private_test.json)')
    parser.add_argument('--hybrid_results_file', type=str, default=None,
                        help='Path to hybrid results JSON file from stage_1_eval.py (e.g., results_hybrid_k200.json). If provided, will use relevant_laws from this file as candidates.')
    parser.add_argument('--candidate_index_file', type=str, default=None,
                        help='Path to candidate index JSON file from stage_1_eval.py (optional, if provided will use this instead of retrieving)')
    parser.add_argument('--train_file', type=str, default=None,
                        help='Path to train JSON file for evaluation (optional, to select best models)')
    parser.add_argument('--models', type=str, nargs='+', default=['gte', 'bge'],
                        choices=['gte', 'bge'],
                        help='Models to evaluate (e.g., --models gte bge). Default: gte and bge only')
    parser.add_argument('--ensemble_only', action='store_true',
                        help='Only run ensemble reranker, skip individual model evaluation (faster)')
    parser.add_argument('--eval_k', type=int, default=20,
                        help='K value for evaluation metrics (P@k, R@k) on train set')
    parser.add_argument('--auto_ensemble', action='store_true',
                        help='Automatically select top 2 models for ensemble based on train evaluation')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for prediction JSON files')
    parser.add_argument('--bm25_index', type=str, default=None,
                        help='Path to BM25 index pickle file (required if candidate_index_file not provided)')
    parser.add_argument('--chroma_db_retriever', type=str, default=None,
                        help='Path to ChromaDB retriever database (required if candidate_index_file not provided)')
    parser.add_argument('--chroma_db_reranker', type=str, default=None,
                        help='Path to ChromaDB reranker database (required if candidate_index_file not provided)')
    parser.add_argument('--collection_retriever', type=str, default='retriever_legal_articles',
                        help='Collection name for retriever')
    parser.add_argument('--collection_reranker', type=str, default='reranker_legal_articles',
                        help='Collection name for reranker')
    parser.add_argument('--embedding_model', type=str, default='AITeamVN/Vietnamese_Embedding',
                        help='Embedding model name')
    parser.add_argument('--retriever_top_k', type=int, default=300,
                        help='Top-k candidates from hybrid retriever (only used if candidate_index_file not provided)')
    parser.add_argument('--reranker_top_k', type=int, default=50,
                        help='Top-k output from reranker (number of relevant_laws per query)')
    parser.add_argument('--bm25_weight', type=float, default=0.3,
                        help='BM25 weight for hybrid retrieval')
    parser.add_argument('--dense_weight', type=float, default=0.7,
                        help='Dense weight for hybrid retrieval')
    parser.add_argument('--rrf_k', type=int, default=60,
                        help='RRF k parameter for ensemble reranker')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda:0, cuda:1, cpu, etc.)')
    parser.add_argument('--use_multi_gpu', action='store_true',
                        help='Use multiple GPUs if available')
    
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
        
        if args.use_multi_gpu and num_gpus > 1:
            logger.info(f"\nMulti-GPU mode: Distributing models across {num_gpus} GPUs")
            MODEL_DEVICES = {
                'gte': 'cuda:0',
                'bge': 'cuda:1' if num_gpus > 1 else 'cuda:0'
            }
        else:
            logger.info(f"\nSingle GPU mode: Using {DEVICE}")
            MODEL_DEVICES = {model: DEVICE for model in ['gte', 'bge']}
    else:
        DEVICE = "cpu"
        MODEL_DEVICES = {model: "cpu" for model in ['gte', 'bge']}
        logger.info("No GPU available, using CPU")
    
    # Reranker models (only GTE and BGE for stage 2)
    ALL_RERANKER_MODELS = {
        'gte': "Booooooooooooo/gte-reranker-10e",
        'bge': "Booooooooooooo/bge-v2-reranker-10e"
    }
    
    # Filter models based on user selection
    RERANKER_MODELS = {name: model_name for name, model_name in ALL_RERANKER_MODELS.items() 
                       if name in args.models}
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Models to evaluate: {list(RERANKER_MODELS.keys())}")
    logger.info(f"  Reranker top_k: {args.reranker_top_k}")
    logger.info(f"  Output directory: {args.output_dir}")
    logger.info(f"  Auto ensemble: {args.auto_ensemble}")
    logger.info(f"  Use candidate index: {args.candidate_index_file is not None}")
    logger.info(f"  Use hybrid results: {args.hybrid_results_file is not None}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load test data
    logger.info(f"\nLoading test data from {args.test_file}...")
    with open(args.test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # Load hybrid results file if provided (from stage_1_eval.py)
    hybrid_results_map = {}
    use_hybrid_results = False
    if args.hybrid_results_file:
        logger.info(f"Loading hybrid results from {args.hybrid_results_file}...")
        with open(args.hybrid_results_file, 'r', encoding='utf-8') as f:
            hybrid_results_list = json.load(f)
            hybrid_results_map = {item['qid']: item.get('relevant_laws', []) for item in hybrid_results_list}
        use_hybrid_results = True
        logger.info(f"Loaded hybrid results for {len(hybrid_results_map)} queries")
        logger.info(f"  Format: qid -> list of aid integers (candidates for reranking)")
    
    # Load candidate index if provided
    candidate_index_map = {}
    use_candidate_index = False
    if args.candidate_index_file:
        logger.info(f"Loading candidate index from {args.candidate_index_file}...")
        with open(args.candidate_index_file, 'r', encoding='utf-8') as f:
            candidate_index_list = json.load(f)
            candidate_index_map = {item['qid']: item['candidates'] for item in candidate_index_list}
        use_candidate_index = True
        logger.info(f"Loaded candidate index for {len(candidate_index_map)} queries")
    
    # Load train data if provided for evaluation
    train_data = None
    if args.train_file:
        logger.info(f"Loading train data from {args.train_file}...")
        with open(args.train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
    
    # Validate arguments
    if not use_candidate_index and not use_hybrid_results:
        if not args.bm25_index or not args.chroma_db_retriever or not args.chroma_db_reranker:
            raise ValueError("If neither candidate_index_file nor hybrid_results_file is provided, bm25_index, chroma_db_retriever, and chroma_db_reranker are required")
    
    # Load ChromaDB for reranker (needed if using hybrid_results_file or not using candidate_index)
    chroma_index_reranker = None
    if use_hybrid_results or (not use_candidate_index):
        if not args.chroma_db_reranker:
            raise ValueError("chroma_db_reranker is required when using hybrid_results_file or when not using candidate_index_file")
        logger.info(f"Loading ChromaDB reranker from {args.chroma_db_reranker}...")
        chroma_index_reranker = ChromaIndex(
            persist_directory=args.chroma_db_reranker,
            collection_name=args.collection_reranker
        )
    
    # Only load retrievers if not using candidate index or hybrid results, or if evaluating on train
    bm25_retriever = None
    chroma_index_retriever = None
    embedder = None
    hybrid_retriever = None
    
    if (not use_candidate_index and not use_hybrid_results) or train_data:
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
    
    # Evaluate models on train set if provided
    model_scores = {}
    single_rerankers = {}
    ensemble_models = []
    
    if train_data and args.auto_ensemble:
        logger.info(f"\n{'='*80}")
        logger.info("Step 1: Evaluating models on train set to select best 2 for ensemble")
        logger.info(f"{'='*80}")
        
        # Evaluate each model one by one, clearing GPU after each
        for name, model_name in RERANKER_MODELS.items():
            model_device = MODEL_DEVICES.get(name, DEVICE)
            logger.info(f"\nLoading {name} ({model_name}) on {model_device}...")
            
            # Load model
            reranker = SingleReranker(
                model_name=model_name,
                device=model_device,
                trust_remote_code=True,
                max_length=512
            )
            
            # Evaluate on train set
            metrics = evaluate_model_on_train(
                model_name=name,
                reranker=reranker,
                train_data=train_data,
                hybrid_retriever=hybrid_retriever,
                chroma_index_reranker=chroma_index_reranker,
                retriever_top_k=args.retriever_top_k,
                reranker_top_k=args.reranker_top_k,
                eval_k=args.eval_k
            )
            
            model_scores[name] = metrics['score']
            single_rerankers[name] = reranker
            
            # Clear GPU cache after each model evaluation
            logger.info(f"Clearing GPU cache after {name} evaluation...")
            clear_gpu_cache()
        
        # Select top 2 models for ensemble
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        top_2_models = [name for name, _ in sorted_models[:2]]
        
        logger.info(f"\n{'='*80}")
        logger.info("Model Evaluation Summary:")
        logger.info(f"{'='*80}")
        for name, score in sorted_models:
            logger.info(f"  {name.upper()}: Score = {score:.4f}")
        
        logger.info(f"\nTop 2 models selected for ensemble: {top_2_models}")
        ensemble_models = [single_rerankers[name] for name in top_2_models]
        
    else:
        # Load all models without evaluation
        if args.ensemble_only:
            # Only load models for ensemble, don't store individual rerankers
            logger.info("Initializing rerankers for ensemble only...")
            ensemble_models = []
            for name, model_name in RERANKER_MODELS.items():
                model_device = MODEL_DEVICES.get(name, DEVICE)
                logger.info(f"  Loading {name} ({model_name}) on {model_device}...")
                reranker = SingleReranker(
                    model_name=model_name,
                    device=model_device,
                    trust_remote_code=True,
                    max_length=512
                )
                ensemble_models.append(reranker)
                # Clear GPU cache after each model to avoid OOM
                clear_gpu_cache()
        else:
            # Load models for both individual evaluation and ensemble
            logger.info("Initializing rerankers...")
            for name, model_name in RERANKER_MODELS.items():
                model_device = MODEL_DEVICES.get(name, DEVICE)
                logger.info(f"  Loading {name} ({model_name}) on {model_device}...")
                single_rerankers[name] = SingleReranker(
                    model_name=model_name,
                    device=model_device,
                    trust_remote_code=True,
                    max_length=512
                )
                # Clear GPU cache after each model to avoid OOM
                clear_gpu_cache()
            
            # Use all models for ensemble (gte + bge)
            if len(RERANKER_MODELS) >= 2:
                ensemble_models = [single_rerankers[name] for name in RERANKER_MODELS.keys()]
            else:
                ensemble_models = []
                logger.warning(f"Need at least 2 models for ensemble, but only {len(RERANKER_MODELS)} model(s) provided")
    
    # Initialize ensemble reranker (always ensemble gte + bge)
    if len(ensemble_models) >= 2:
        logger.info(f"\nInitializing ensemble reranker (RRF with {len(ensemble_models)} models: {list(RERANKER_MODELS.keys())})...")
        ensemble_reranker = EnsembleReranker(
            reranker_models=ensemble_models,
            rrf_k=args.rrf_k
        )
    else:
        ensemble_reranker = None
        logger.warning("No ensemble created (need at least 2 models)")
    
    # Processing
    logger.info(f"\n{'='*80}")
    logger.info(f"Step 2: Processing {len(test_data)} queries on test set...")
    logger.info(f"{'='*80}")
    
    # Storage for results for each method
    if args.ensemble_only:
        # Only store ensemble results
        results = {}
        if ensemble_reranker:
            results['ensemble'] = []
    else:
        # Store results for individual models and ensemble
        results = {name: [] for name in RERANKER_MODELS.keys()}
        if ensemble_reranker:
            results['ensemble'] = []
    
    for idx, sample in enumerate(tqdm(test_data, desc="Processing")):
        qid = sample['qid']
        question = sample.get('question', '')
        
        if not question:
            # Empty question, add empty results
            if args.ensemble_only:
                if ensemble_reranker:
                    results['ensemble'].append({
                        'qid': qid,
                        'relevant_laws': []
                    })
            else:
                for method in results.keys():
                    results[method].append({
                        'qid': qid,
                        'relevant_laws': []
                    })
            continue
        
        # Step 1: Get candidate documents
        candidate_docs = []
        
        if use_candidate_index:
            # Use candidate index from stage_1_eval.py (has documents already)
            candidate_docs = candidate_index_map.get(qid, [])
        elif use_hybrid_results:
            # Use hybrid results file: get aid list, then query ChromaDB for documents
            candidate_aids = hybrid_results_map.get(qid, [])
            if candidate_aids:
                # Query ChromaDB reranker to get documents for each aid
                for aid in candidate_aids:
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
        else:
            # Retrieve candidates using hybrid retriever
            try:
                retrieved_results = hybrid_retriever.retrieve(question, top_k=args.retriever_top_k)
                retrieved_aids = [str(aid) for aid, _ in retrieved_results]
            except Exception as e:
                logger.error(f"Error in retrieval for sample {qid}: {e}")
                retrieved_aids = []
            
            if not retrieved_aids:
                # No candidates found, add empty results
                if args.ensemble_only:
                    if ensemble_reranker:
                        results['ensemble'].append({
                            'qid': qid,
                            'relevant_laws': []
                        })
                else:
                    for method in results.keys():
                        results[method].append({
                            'qid': qid,
                            'relevant_laws': []
                        })
                continue
            
            # Get documents from ChromaDB reranker
            for aid in retrieved_aids:
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
        
        if not candidate_docs:
            # No documents found, add empty results
            if args.ensemble_only:
                if ensemble_reranker:
                    results['ensemble'].append({
                        'qid': qid,
                        'relevant_laws': []
                    })
            else:
                for method in results.keys():
                    results[method].append({
                        'qid': qid,
                        'relevant_laws': []
                    })
            continue
        
        # Step 3: Rerank with each method (skip if ensemble_only)
        if not args.ensemble_only:
            for method_name in RERANKER_MODELS.keys():
                try:
                    reranked_results = single_rerankers[method_name].rerank(
                        question, 
                        candidate_docs, 
                        top_k=args.reranker_top_k
                    )
                    final_aids = [str(aid) for aid, _ in reranked_results]
                    relevant_laws = convert_aids_to_int(final_aids)
                    relevant_laws = list(dict.fromkeys(relevant_laws))  # Remove duplicates
                    
                    results[method_name].append({
                        'qid': qid,
                        'relevant_laws': relevant_laws
                    })
                except Exception as e:
                    logger.error(f"Error in {method_name} reranking for sample {qid}: {e}")
                    results[method_name].append({
                        'qid': qid,
                        'relevant_laws': []
                    })
        
        # Step 4: Rerank with ensemble (if available)
        if ensemble_reranker:
            try:
                reranked_results = ensemble_reranker.rerank(
                    question,
                    candidate_docs,
                    top_k=args.reranker_top_k
                )
                final_aids = [str(aid) for aid, _ in reranked_results]
                relevant_laws = convert_aids_to_int(final_aids)
                relevant_laws = list(dict.fromkeys(relevant_laws))  # Remove duplicates
                
                results['ensemble'].append({
                    'qid': qid,
                    'relevant_laws': relevant_laws
                })
            except Exception as e:
                logger.error(f"Error in ensemble reranking for sample {qid}: {e}")
                results['ensemble'].append({
                    'qid': qid,
                    'relevant_laws': []
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
    
    # Save output files
    output_files = {}
    if not args.ensemble_only:
        # Save individual model results
        for name in RERANKER_MODELS.keys():
            output_files[name] = os.path.join(args.output_dir, f'results_{name}_k{args.reranker_top_k}.json')
    if ensemble_reranker:
        output_files['ensemble'] = os.path.join(args.output_dir, f'results_ensemble_k{args.reranker_top_k}.json')
    
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
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Format: [{{\"qid\": <int>, \"relevant_laws\": [<int>, ...]}}, ...]")
    logger.info(f"Top-k: {args.reranker_top_k}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()