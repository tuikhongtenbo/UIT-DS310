# src/retrieval/ensemble.py
"""
Ensemble Retriever Module
Orchestrator for Hybrid Search (BM25 + Dense)
"""

from typing import Any, List, Tuple

from src.fusion.weighted_ensemble import WeightedEnsemble
from src.utils.logger import setup_logger

logger = setup_logger("ensemble_retriever")


class EnsembleRetriever:
    """
    Ensemble Retriever that combines results from BM25 and Dense Retriever.
    Uses WeightedEnsemble to calculate combined scores and rerank.
    """

    def __init__(self, bm25_retriever: Any, dense_retriever: Any, weights: tuple = (0.5, 0.5)):
        """
        Initialize Ensemble Retriever.

        Args:
            bm25_retriever: Instance of BM25Retriever.
            dense_retriever: Instance of DenseRetriever.
            weights: Tuple (weight_bm25, weight_dense).
        """
        self.bm25 = bm25_retriever
        self.dense = dense_retriever

        self.fusion = WeightedEnsemble(weights=weights)

        logger.info(f"EnsembleRetriever initialized with weights: BM25={weights[0]}, Dense={weights[1]}")

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """
        Perform query to both retrievers and combine results.

        Args:
            query: Query string.
            top_k: Number of final results desired.

        Returns:
            List of tuples (doc_id, combined_score) sorted by score.
        """
        search_k = max(top_k, 50)

        # 1. Retrieve from BM25
        try:
            bm25_results_list = self.bm25.retrieve(query, top_k=search_k)
        except Exception as e:
            logger.error(f"BM25 Retrieval failed: {e}")
            bm25_results_list = []

        # 2. Retrieve from Dense
        try:
            dense_results_list = self.dense.retrieve(query, top_k=search_k)
        except Exception as e:
            logger.error(f"Dense Retrieval failed: {e}")
            dense_results_list = []

        # 3. Convert List -> Dict for WeightedEnsemble processing
        bm25_scores = dict(bm25_results_list)
        dense_scores = dict(dense_results_list)

        # 4. Call Fusion module to calculate and rank
        return self.fusion.rank(bm25_scores, dense_scores, top_k=top_k)

    def set_weights(self, weights: Tuple[float, float]) -> None:
        """
        Update weights (for Grid Search or runtime tuning).

        Args:
            weights: Tuple (weight_bm25, weight_dense)
        """
        self.fusion.set_weights(weights)
        logger.info(f"Weights updated to: {weights}")