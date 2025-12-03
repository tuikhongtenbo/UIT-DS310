"""
Main script to run experiments with different configurations
Loads config from exp_i.yaml, processes public_test.json, and outputs results
"""
import sys
import os
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from config.config import AppConfig
from src.utils.logger import setup_logger
from src.embeddings.embedder import VietnameseEmbedder
from src.embeddings.chroma_index import ChromaIndex
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.ensemble import EnsembleRetriever
from src.reranking.build_rerankers import build_ensemble_reranker, build_qwen_reranker
from src.utils.format_reranker_input import format_chroma_to_reranker, format_documents_for_reranker

logger = setup_logger("main")


class ExperimentPipeline:
    """Pipeline to run experiments with different configurations"""
    
    def __init__(self, config_path: str):
        """Initialize pipeline with config"""
        self.config = AppConfig(config_path).config
        self.retriever = None
        self.reranker = None
        self.qwen_reranker = None
        
        # Initialize components
        self._init_retriever()
        self._init_reranker()
    
    def _init_retriever(self):
        """Initialize retriever based on config"""
        retriever_config = self.config.get('retriever', {})
        embedder_config = retriever_config.get('embedder', {})
        mode = retriever_config.get('mode', 'ensemble')
        top_k = retriever_config.get('top_k', 50)
        
        # Initialize embedder
        model_name = embedder_config.get('model', "AITeamVN/Vietnamese_Embedding")
        device = embedder_config.get('device', 'cuda')
        self.embedder = VietnameseEmbedder(model_name=model_name, device=device)
        
        models_config = retriever_config.get('models', {})
        bm25_config = models_config.get('bm25', {})
        sentencebert_config = models_config.get('sentencebert', {})
        
        bm25_retriever = None
        dense_retriever = None
        
        # Initialize BM25 if enabled
        if bm25_config.get('enabled', False):
            logger.info("Initializing BM25 retriever...")
            bm25_retriever = BM25Retriever(
                embedder=self.embedder,
                k1=bm25_config.get('k1', 1.5),
                b=bm25_config.get('b', 0.75)
            )
            # Load BM25 index
            pipeline_config = self.config.get('pipeline', {})
            indexes_config = pipeline_config.get('indexes', {})
            bm25_index_path = indexes_config.get('bm25_index_path', './data/bm25_index.pkl')
            
            # Handle path: if absolute, use as-is; if relative, join with current_dir
            if os.path.isabs(bm25_index_path):
                # Absolute path - use directly
                bm25_index_path = os.path.normpath(bm25_index_path)
            else:
                # Relative path - remove leading ./ if present and join with current_dir
                bm25_index_path = bm25_index_path.lstrip('./')
                bm25_index_path = os.path.join(current_dir, bm25_index_path)
                bm25_index_path = os.path.normpath(bm25_index_path)
            
            if os.path.exists(bm25_index_path):
                bm25_retriever.load(bm25_index_path)
                logger.info(f"BM25 index loaded from {bm25_index_path}")
            else:
                logger.error(f"BM25 index not found at {bm25_index_path}")
                logger.error("BM25 retriever requires an index to work.")
                logger.error("For Experiment 1, BM25 index is required but not found.")
                logger.error("You can skip experiments that require BM25 (exp 1, 3, 4, 6, 7, 8, 9)")
                logger.error("or build the BM25 index first if you have the legal corpus.")
        
        # Initialize Dense retriever if enabled
        if sentencebert_config.get('enabled', False):
            logger.info("Initializing Dense retriever...")
            pipeline_config = self.config.get('pipeline', {})
            indexes_config = pipeline_config.get('indexes', {})
            chroma_path = indexes_config.get('chroma_db_retriever_path', './data/chroma_db_retriever')
            if not os.path.isabs(chroma_path):
                chroma_path = os.path.join(current_dir, chroma_path)
            
            collection_name = sentencebert_config.get('collection_name', 'retriever_legal_articles')
            chroma_index = ChromaIndex(persist_directory=chroma_path, collection_name=collection_name)
            dense_retriever = DenseRetriever(chroma_index=chroma_index, embedder=self.embedder)
            logger.info(f"Dense retriever initialized with ChromaDB at {chroma_path}")
        
        # Create retriever based on mode
        if mode == 'sparse' and bm25_retriever:
            self.retriever = bm25_retriever
        elif mode == 'dense' and dense_retriever:
            self.retriever = dense_retriever
        elif mode == 'ensemble' and (bm25_retriever or dense_retriever):
            ensemble_config = retriever_config.get('ensemble', {})
            weights = ensemble_config.get('weighted', {}).get('weights', {})
            bm25_weight = weights.get('bm25', 0.5)
            dense_weight = weights.get('sentencebert', 0.5)
            self.retriever = EnsembleRetriever(
                bm25_retriever=bm25_retriever,
                dense_retriever=dense_retriever,
                weights=(bm25_weight, dense_weight)
            )
        else:
            raise ValueError(f"Invalid retriever mode: {mode} or missing retrievers")
        
        self.retriever_top_k = top_k
        logger.info(f"Retriever initialized: mode={mode}, top_k={top_k}")
    
    def _init_reranker(self):
        """Initialize reranker based on config"""
        reranker_config = self.config.get('reranker', {})
        models_config = reranker_config.get('models', {})
        
        # Check if any reranker is enabled
        has_cross_encoder = any([
            models_config.get('gte', {}).get('enabled', False),
            models_config.get('bge_v2', {}).get('enabled', False),
            models_config.get('jina', {}).get('enabled', False)
        ])
        
        if has_cross_encoder:
            logger.info("Initializing Cross-Encoder reranker...")
            self.reranker = build_ensemble_reranker(self.config)
            if self.reranker:
                logger.info("Cross-Encoder reranker initialized")
        
        # Initialize Qwen reranker if enabled
        qwen_config = reranker_config.get('qwen', {})
        if qwen_config.get('enabled', False):
            logger.info("Initializing Qwen reranker...")
            self.qwen_reranker = build_qwen_reranker(self.config)
            if self.qwen_reranker:
                logger.info("Qwen reranker initialized")
        
        self.reranker_top_k = reranker_config.get('top_k', 2)
    
    def run_query(self, query: str) -> List[str]:
        """
        Run query through pipeline and return list of aid
        
        Args:
            query: Query string
            
        Returns:
            List of aid (article IDs)
        """
        # Step 1: Retrieve
        if not self.retriever:
            logger.error("Retriever not initialized")
            return []
        
        retrieved_results = self.retriever.retrieve(query, top_k=self.retriever_top_k)
        
        if not retrieved_results:
            return []
        
        # Step 2: Extract aids from retrieved results
        retrieved_aids = []
        
        # Check if we need to get metadata from ChromaDB (for Dense retriever)
        retriever_config = self.config.get('retriever', {})
        models_config = retriever_config.get('models', {})
        sentencebert_config = models_config.get('sentencebert', {})
        is_dense = sentencebert_config.get('enabled', False)
        
        if is_dense:
            # For Dense retriever, doc_id is ChromaDB ID, need to get metadata to extract aid
            pipeline_config = self.config.get('pipeline', {})
            indexes_config = pipeline_config.get('indexes', {})
            chroma_path = indexes_config.get('chroma_db_retriever_path', './data/chroma_db_retriever')
            if not os.path.isabs(chroma_path):
                chroma_path = os.path.join(current_dir, chroma_path)
            
            collection_name = sentencebert_config.get('collection_name', 'retriever_legal_articles')
            chroma_index = ChromaIndex(persist_directory=chroma_path, collection_name=collection_name)
            
            # Get all doc_ids
            doc_ids = [doc_id for doc_id, score in retrieved_results]
            
            # Get metadata for all doc_ids at once
            try:
                doc_data = chroma_index.collection.get(ids=doc_ids)
                if doc_data and doc_data.get('metadatas'):
                    metadatas = doc_data['metadatas']
                    for i, metadata in enumerate(metadatas):
                        aid = metadata.get('aid', doc_ids[i] if i < len(doc_ids) else '')
                        retrieved_aids.append(str(aid))
                else:
                    # Fallback: use doc_id as aid
                    retrieved_aids = [str(doc_id) for doc_id, score in retrieved_results]
            except Exception as e:
                logger.warning(f"Error getting metadata from ChromaDB: {e}")
                # Fallback: use doc_id as aid
                retrieved_aids = [str(doc_id) for doc_id, score in retrieved_results]
        else:
            # For BM25, doc_id is already aid
            retrieved_aids = [str(aid) for aid, score in retrieved_results]
        
        # Step 3: Rerank if reranker is enabled
        if self.reranker or self.qwen_reranker:
            # Get documents from reranker ChromaDB for reranking
            reranker_config = self.config.get('reranker', {})
            embedder_config = reranker_config.get('embedder', {})
            pipeline_config = self.config.get('pipeline', {})
            indexes_config = pipeline_config.get('indexes', {})
            chroma_path = indexes_config.get('chroma_db_reranker_path', './data/chroma_db_reranker')
            if not os.path.isabs(chroma_path):
                chroma_path = os.path.join(current_dir, chroma_path)
            
            collection_name = embedder_config.get('collection_name', 'reranker_legal_articles')
            reranker_chroma = ChromaIndex(persist_directory=chroma_path, collection_name=collection_name)
            
            # Query reranker ChromaDB with query to get documents
            query_embedding = self.embedder.encode(query)
            if isinstance(query_embedding, list):
                query_embedding = query_embedding[0]
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()
            
            # Query reranker ChromaDB to get documents (article-level)
            reranker_results = reranker_chroma.query(
                query_embedding=query_embedding,
                n_results=min(max(len(retrieved_aids), self.retriever_top_k), 100)
            )
            
            if reranker_results and reranker_results.get('documents'):
                # Format documents for reranker
                formatted_docs = format_chroma_to_reranker(reranker_results)
                
                # Filter to only include documents with aids in retrieved_aids
                # This ensures we rerank only the documents retrieved by BM25
                retrieved_aids_set = set(retrieved_aids)
                filtered_docs = [
                    doc for doc in formatted_docs 
                    if str(doc.get('aid', '')) in retrieved_aids_set
                ]
                
                # If no matching documents found, fallback to retrieved aids
                if not filtered_docs:
                    logger.warning(f"No matching documents found in reranker ChromaDB for retrieved_aids. Using retrieved_aids directly.")
                    return retrieved_aids[:self.reranker_top_k] if self.reranker_top_k else retrieved_aids
                
                # Rerank with Cross-Encoder
                if self.reranker:
                    reranked_results = self.reranker.rerank(query, filtered_docs, top_k=self.reranker_top_k)
                    # reranked_results is List[Tuple[str, float]] - (aid, score)
                else:
                    # If no cross-encoder, create dummy results from filtered_docs
                    # Only include aids that are in filtered_docs (available in reranker ChromaDB)
                    # Qwen reranker needs List[Tuple[str, float]] format
                    filtered_aids_set = {str(doc.get('aid', '')) for doc in filtered_docs}
                    reranked_results = [
                        (aid, 1.0) for aid in retrieved_aids[:self.reranker_top_k]
                        if aid in filtered_aids_set
                    ]
                
                # Apply Qwen reranker if enabled
                if self.qwen_reranker and reranked_results:
                    # Build documents_dict for Qwen reranker
                    # Qwen reranker needs Dict[str, Dict[str, Any]] with aid as key
                    documents_dict = self.qwen_reranker.build_documents_dict(filtered_docs, id_key="aid")
                    
                    # Qwen reranker takes: query, reranker_results (List[Tuple[str, float]]), documents_dict
                    qwen_results = self.qwen_reranker.rerank(
                        query, 
                        reranked_results, 
                        documents_dict, 
                        top_k=self.reranker_top_k
                    )
                    # Extract aids from Qwen results
                    final_aids = [str(aid) for aid, score in qwen_results]
                else:
                    # No Qwen reranker, use cross-encoder results
                    final_aids = [str(aid) for aid, score in reranked_results]
                
                return final_aids
            else:
                # Fallback to retrieved aids
                logger.warning("Reranker ChromaDB query returned no results. Using retrieved_aids directly.")
                return retrieved_aids[:self.reranker_top_k] if self.reranker_top_k else retrieved_aids
        else:
            # No reranker, return retrieved aids
            return retrieved_aids[:self.reranker_top_k] if self.reranker_top_k else retrieved_aids


def main():
    """Main function to run experiments"""
    parser = argparse.ArgumentParser(description="Run experiments with different configurations")
    parser.add_argument(
        '--exp', 
        type=int, 
        required=True, 
        choices=range(1, 10),
        help='Experiment number (1-9)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='./dataset/public_test.json',
        help='Input test file path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: ./outputs/exp_{exp}_results.json)'
    )
    args = parser.parse_args()
    
    # Load config
    config_path = f"config/exp_{args.exp}.yaml"
    if not os.path.isabs(config_path):
        config_path = os.path.join(current_dir, config_path)
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        logger.error(f"Current directory: {current_dir}")
        logger.error(f"Looking for: {config_path}")
        return
    
    logger.info(f"Running experiment {args.exp} with config: {config_path}")
    
    # Initialize pipeline
    pipeline = ExperimentPipeline(config_path)
    
    # Load test data
    input_path = args.input
    if not os.path.isabs(input_path):
        input_path = os.path.join(current_dir, input_path)
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return
    
    logger.info(f"Loading test data from: {input_path}")
    with open(input_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    logger.info(f"Processing {len(test_data)} queries...")
    
    # Helper function to convert aids to integers
    def convert_aids_to_int(aids):
        """Convert list of aid strings/ints to integers, filtering out invalid values"""
        result = []
        for aid in aids:
            try:
                # If already an int, use it directly
                if isinstance(aid, int):
                    result.append(aid)
                else:
                    # Try to convert string to int directly
                    aid_str = str(aid).strip()
                    try:
                        result.append(int(aid_str))
                    except ValueError:
                        # If direct conversion fails, extract numbers from string
                        # e.g., "law_1_article_5" -> extract 5
                        numbers = re.findall(r'\d+', aid_str)
                        if numbers:
                            result.append(int(numbers[-1]))
                        else:
                            logger.warning(f"Could not extract number from aid: {aid}")
            except Exception as e:
                logger.warning(f"Error converting aid to int: {aid}, error: {e}")
                continue
        return result
    
    # Process each query
    results = []
    for idx, item in enumerate(test_data):
        qid = item.get('qid')
        question = item.get('question', '')
        
        if not question:
            logger.warning(f"Empty question for qid {qid}, skipping...")
            results.append({
                'qid': qid,
                'relevant_laws': []
            })
            continue
        
        logger.info(f"Processing query {idx+1}/{len(test_data)}: qid={qid}")
        
        # Run query through pipeline
        try:
            relevant_aids = pipeline.run_query(question)
            # Convert aids to integers and remove duplicates
            relevant_laws = convert_aids_to_int(relevant_aids)
            # Remove duplicates while preserving order
            relevant_laws = list(dict.fromkeys(relevant_laws))
            results.append({
                'qid': qid,
                'relevant_laws': relevant_laws
            })
            logger.info(f"Query {qid}: Found {len(relevant_laws)} relevant laws")
        except Exception as e:
            logger.error(f"Error processing query {qid}: {e}")
            results.append({
                'qid': qid,
                'relevant_laws': []
            })
    
    # Sort results by qid to ensure consistent order
    results.sort(key=lambda x: x.get('qid', 0))
    
    # Validate: ensure all qids from input are present in results
    input_qids = set(item.get('qid') for item in test_data)
    result_qids = set(result.get('qid') for result in results)
    missing_qids = input_qids - result_qids
    if missing_qids:
        logger.warning(f"Missing qids in results: {sorted(missing_qids)}")
        # Add missing qids with empty relevant_laws
        for qid in sorted(missing_qids):
            results.append({
                'qid': qid,
                'relevant_laws': []
            })
        # Re-sort after adding missing qids
        results.sort(key=lambda x: x.get('qid', 0))
    
    # Save results
    output_path = args.output
    if not output_path:
        output_dir = os.path.join(current_dir, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        # Default to results.json for submission format
        output_path = os.path.join(output_dir, 'results.json')
    else:
        # Ensure output directory exists even if path is provided
        output_dir = os.path.dirname(output_path)
        if output_dir:  # If there's a directory component
            os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Saving results to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Experiment {args.exp} completed! Results saved to {output_path}")
    logger.info(f"Total queries processed: {len(results)}")
    logger.info(f"Format: [{{\"qid\": <int>, \"relevant_laws\": [<int>, ...]}}, ...]")


if __name__ == "__main__":
    main()