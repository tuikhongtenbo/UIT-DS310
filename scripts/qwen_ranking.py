"""
Qwen Ranking Script

Formats reranker results for Qwen LLM reranking.
Takes output from stage_2_eval.py and formats it for Qwen reranker.
Supports multiple top-k cases (top-1 to top-5) and exp_7 filter mode.
"""

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.reranking.qwen_reranker import QwenReranker, QwenRerankerVLLM
from src.utils.logger import setup_logger

logger = setup_logger("qwen_ranking")


def convert_aids_to_int(aids: List[Any]) -> List[int]:
    """
    Convert list of aid strings/ints to integers, filtering out invalid values.
    
    Args:
        aids: List of aid values (can be strings, ints, or mixed)
        
    Returns:
        List of integer aids
    """
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


def setup_device(args: argparse.Namespace) -> str:
    """
    Setup and configure GPU/CPU device.
    
    Args:
        args: Command line arguments
        
    Returns:
        Device string ('cuda:0' or 'cpu')
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"Available GPUs: {num_gpus}")
        for i in range(num_gpus):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        device = args.device if args.device else "cuda:0"
        logger.info(f"\nUsing device: {device}")
        return device
    else:
        logger.info("No GPU available, using CPU")
        return "cpu"


def load_data(test_file: str, reranker_results_file: str, legal_corpus_file: str) -> tuple:
    """
    Load test data, reranker results, and legal corpus.
    
    Args:
        test_file: Path to test JSON file
        reranker_results_file: Path to reranker results JSON file
        legal_corpus_file: Path to legal corpus JSON file
        
    Returns:
        Tuple of (qid_to_question dict, reranker_results_list, legal_corpus_dict)
    """
    logger.info(f"\nLoading test data from {test_file}...")
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    qid_to_question = {item['qid']: item.get('question', '') for item in test_data}
    
    logger.info(f"Loading reranker results from {reranker_results_file}...")
    with open(reranker_results_file, 'r', encoding='utf-8') as f:
        reranker_results_list = json.load(f)
    
    logger.info(f"Loading legal corpus from {legal_corpus_file}...")
    with open(legal_corpus_file, 'r', encoding='utf-8') as f:
        legal_corpus = json.load(f)
    
    legal_corpus_dict = {}
    for law_entry in legal_corpus:
        content_list = law_entry.get('content', [])
        for article in content_list:
            aid = article.get('aid')
            content_article = article.get('content_Article', '')
            if aid is not None and content_article:
                legal_corpus_dict[str(aid)] = content_article
    
    logger.info(f"Loaded {len(legal_corpus_dict)} articles from legal corpus")
    
    return qid_to_question, reranker_results_list, legal_corpus_dict


def initialize_reranker(args: argparse.Namespace, device: str):
    """
    Initialize Qwen reranker (either vLLM or transformers backend).
    
    Args:
        args: Command line arguments
        device: Device string
        
    Returns:
        Initialized QwenReranker or QwenRerankerVLLM instance
    """
    if args.use_vllm:
        logger.info(f"\nInitializing Qwen reranker with vLLM ({args.qwen_model})...")
        logger.info(f"  Tensor parallel size: {args.tensor_parallel_size}")
        logger.info(f"  GPU memory utilization: {args.gpu_memory_utilization}")
        logger.info(f"  Max new tokens: {args.max_new_tokens}")
        
        tensor_parallel_size = args.tensor_parallel_size
        if tensor_parallel_size == 1 and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            tensor_parallel_size = min(2, torch.cuda.device_count())
            logger.info(f"  Auto-detected tensor parallel size: {tensor_parallel_size}")
        
        return QwenRerankerVLLM(
            model_name=args.qwen_model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            max_content_length=2000,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            exp_num=None,
            verbose=args.verbose
        )
    else:
        logger.info(f"\nInitializing Qwen reranker with transformers ({args.qwen_model})...")
        logger.info(f"  Max new tokens: {args.max_new_tokens}")
        logger.info(f"  Use 4-bit: {args.use_4bit}")
        
        return QwenReranker(
            model_name=args.qwen_model,
            device=device,
            threshold=0.0,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            use_4bit=args.use_4bit,
            exp_num=None,
            verbose=args.verbose
        )


def set_exp_num(qwen_reranker, exp_7: bool, top_k: Optional[int]) -> None:
    """
    Set exp_num based on top_k or exp_7 mode.
    
    Args:
        qwen_reranker: Qwen reranker instance
        exp_7: Whether exp_7 mode is enabled
        top_k: Top-k value (None for exp_7 mode)
    """
    if exp_7:
        qwen_reranker.exp_num = 7
    elif top_k == 1:
        qwen_reranker.exp_num = 10
    elif top_k == 2:
        qwen_reranker.exp_num = 11
    elif top_k == 3:
        qwen_reranker.exp_num = 12
    elif top_k == 4:
        qwen_reranker.exp_num = 13
    elif top_k == 5:
        qwen_reranker.exp_num = 14
    else:
        qwen_reranker.exp_num = None


def process_sample(
    item: Dict[str, Any],
    qid_to_question: Dict[int, str],
    legal_corpus_dict: Dict[str, str],
    qwen_reranker,
    exp_7: bool,
    top_k: Optional[int]
) -> Dict[str, Any]:
    """
    Process a single sample through Qwen reranker.
    
    Args:
        item: Sample item with qid and relevant_laws
        qid_to_question: Mapping from qid to question
        legal_corpus_dict: Mapping from aid to article content
        qwen_reranker: Qwen reranker instance
        exp_7: Whether exp_7 mode is enabled
        top_k: Top-k value (None for exp_7 mode)
        
    Returns:
        Result dictionary with qid and relevant_laws
    """
    qid = item['qid']
    question = qid_to_question.get(qid, '')
    relevant_laws = item.get('relevant_laws', [])
    
    if not question or not relevant_laws:
        return {'qid': qid, 'relevant_laws': []}
    
    candidate_docs = []
    for aid in relevant_laws:
        aid_str = str(aid)
        content_article = legal_corpus_dict.get(aid_str)
        if content_article:
            candidate_docs.append({
                'aid': aid_str,
                'content': content_article,
                'text': content_article
            })
    
    if not candidate_docs:
        return {'qid': qid, 'relevant_laws': []}
    
    reranker_results_for_qwen = [(str(doc['aid']), 1.0) for doc in candidate_docs]
    documents_dict = qwen_reranker.build_documents_dict(candidate_docs, id_key="aid")
    
    set_exp_num(qwen_reranker, exp_7, top_k)
    
    try:
        rerank_top_k = None if exp_7 else top_k
        qwen_results = qwen_reranker.rerank(
            query=question,
            reranker_results=reranker_results_for_qwen,
            documents_dict=documents_dict,
            top_k=rerank_top_k
        )
        
        selected_aids = [str(aid) for aid, _ in qwen_results]
        relevant_laws_int = convert_aids_to_int(selected_aids)
        relevant_laws_int = list(dict.fromkeys(relevant_laws_int))
        
        if exp_7:
            return {'qid': qid, 'relevant_laws': relevant_laws_int}
        else:
            return {
                'qid': qid,
                'relevant_laws': relevant_laws_int[:top_k] if top_k else relevant_laws_int
            }
    except Exception as e:
        logger.error(f"Error in Qwen reranking for sample {qid}: {e}")
        return {
            'qid': qid,
            'relevant_laws': convert_aids_to_int([str(aid) for aid in relevant_laws])[:top_k] if top_k else []
        }


def save_results(results: List[Dict[str, Any]], output_file: str, exp_7: bool, top_k: Optional[int]) -> None:
    """
    Save results to JSON file and print statistics.
    
    Args:
        results: List of result dictionaries
        output_file: Output file path
        exp_7: Whether exp_7 mode is enabled
        top_k: Top-k value (for logging)
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    total_queries = len(results)
    queries_with_results = sum(1 for r in results if len(r.get('relevant_laws', [])) > 0)
    avg_laws_per_query = (
        sum(len(r.get('relevant_laws', [])) for r in results) / total_queries
        if total_queries > 0 else 0
    )
    
    mode_name = "Exp-7 (Filter Mode)" if exp_7 else f"Top-{top_k}"
    logger.info(f"\n{mode_name} Results:")
    logger.info(f"  Output file: {output_file}")
    logger.info(f"  Total queries: {total_queries}")
    logger.info(f"  Queries with results: {queries_with_results} ({queries_with_results/total_queries*100:.1f}%)")
    logger.info(f"  Avg laws per query: {avg_laws_per_query:.2f}")


def main():
    """Main function to run Qwen ranking script."""
    parser = argparse.ArgumentParser(
        description='Format reranker results for Qwen LLM reranking'
    )
    parser.add_argument(
        '--test_file', type=str, required=True,
        help='Path to test JSON file (e.g., private_test.json)'
    )
    parser.add_argument(
        '--reranker_results_file', type=str, required=True,
        help='Path to reranker results JSON file (e.g., results_ensemble_k50.json)'
    )
    parser.add_argument(
        '--legal_corpus_file', type=str, required=True,
        help='Path to legal corpus JSON file (e.g., legal_corpus.json)'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./results',
        help='Output directory for Qwen prediction JSON files'
    )
    parser.add_argument(
        '--qwen_model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
        help='Qwen model name'
    )
    parser.add_argument(
        '--device', type=str, default=None,
        help='Device to use (cuda:0, cuda:1, cpu, etc.)'
    )
    parser.add_argument(
        '--use_4bit', action='store_true',
        help='Use 4-bit quantization for Qwen model (only for transformers backend)'
    )
    parser.add_argument(
        '--use_vllm', action='store_true',
        help='Use vLLM backend for faster inference (recommended)'
    )
    parser.add_argument(
        '--tensor_parallel_size', type=int, default=1,
        help='Tensor parallel size for vLLM (number of GPUs to use)'
    )
    parser.add_argument(
        '--gpu_memory_utilization', type=float, default=0.9,
        help='GPU memory utilization for vLLM (0.0-1.0)'
    )
    parser.add_argument(
        '--max_new_tokens', type=int, default=24,
        help='Max new tokens for Qwen generation'
    )
    parser.add_argument(
        '--temperature', type=float, default=0.1,
        help='Temperature for Qwen generation'
    )
    parser.add_argument(
        '--top_k_cases', type=int, nargs='+', default=[1, 2, 3, 4, 5],
        help='Top-k cases to generate (e.g., --top_k_cases 1 2 3 4 5). Ignored if --exp_7 is used.'
    )
    parser.add_argument(
        '--exp_7', action='store_true',
        help='Use exp_7 mode: filter incorrect aids, keep only correct ones. If all are correct, keep all.'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Print model output for each query (for debugging)'
    )
    
    args = parser.parse_args()
    
    if args.exp_7:
        args.top_k_cases = [None]
    
    device = setup_device(args)
    
    logger.info(f"\nConfiguration:")
    logger.info(f"  Reranker results file: {args.reranker_results_file}")
    logger.info(f"  Top-k cases: {args.top_k_cases}")
    logger.info(f"  Qwen model: {args.qwen_model}")
    logger.info(f"  Output directory: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    qid_to_question, reranker_results_list, legal_corpus_dict = load_data(
        args.test_file, args.reranker_results_file, args.legal_corpus_file
    )
    
    qwen_reranker = initialize_reranker(args, device)
    
    for top_k in args.top_k_cases:
        if args.exp_7:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing Exp-7 (Filter Mode)...")
            logger.info(f"{'='*80}")
        else:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing Top-{top_k} case...")
            logger.info(f"{'='*80}")
        
        results = []
        desc = "Exp-7" if args.exp_7 else f"Top-{top_k}"
        
        for item in tqdm(reranker_results_list, desc=desc):
            result = process_sample(
                item, qid_to_question, legal_corpus_dict, qwen_reranker, args.exp_7, top_k
            )
            results.append(result)
        
        results.sort(key=lambda x: x.get('qid', 0))
        
        input_qids = set(item.get('qid') for item in reranker_results_list)
        result_qids = set(result.get('qid') for result in results)
        missing_qids = input_qids - result_qids
        
        if missing_qids:
            for qid in sorted(missing_qids):
                results.append({'qid': qid, 'relevant_laws': []})
            results.sort(key=lambda x: x.get('qid', 0))
        
        if args.exp_7:
            output_file = os.path.join(args.output_dir, 'results_qwen_exp7.json')
        else:
            output_file = os.path.join(args.output_dir, f'results_qwen_top{top_k}_k50.json')
        
        save_results(results, output_file, args.exp_7, top_k)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Qwen ranking completed!")
    logger.info(f"Output files:")
    if args.exp_7:
        logger.info(f"  - results_qwen_exp7.json")
    else:
        for top_k in args.top_k_cases:
            logger.info(f"  - results_qwen_top{top_k}_k50.json")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    main()