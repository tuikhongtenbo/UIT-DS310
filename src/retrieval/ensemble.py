# src/retrieval/ensemble.py
"""
Ensemble Retriever Module
Orchestrator for Hybrid Search (BM25 + Dense)
"""
from typing import List, Tuple, Any

from src.fusion.weighted_ensemble import WeightedEnsemble
from src.utils.logger import setup_logger

logger = setup_logger("ensemble_retriever")

class EnsembleRetriever:
    """
    Lớp điều phối (Orchestrator) kết hợp kết quả từ BM25 và Dense Retriever.
    Sử dụng WeightedEnsemble để tính điểm tổng hợp và xếp hạng lại.
    """

    def __init__(self, bm25_retriever: Any, dense_retriever: Any, weights: tuple = (0.5, 0.5)):
        """
        Khởi tạo Ensemble Retriever.

        Args:
            bm25_retriever: Instance của BM25Retriever.
            dense_retriever: Instance của DenseRetriever.
            weights: Tuple (weight_bm25, weight_dense).
        """
        self.bm25 = bm25_retriever
        self.dense = dense_retriever

        self.fusion = WeightedEnsemble(weights=weights)

        logger.info(f"EnsembleRetriever initialized with weights: BM25={weights[0]}, Dense={weights[1]}")

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """
        Thực hiện truy vấn tới cả 2 retriever con và gộp kết quả.

        Args:
            query: Câu truy vấn.
            top_k: Số lượng kết quả cuối cùng mong muốn.

        Returns:
            List các tuple (doc_id, combined_score) đã được sắp xếp.
        """

        search_k = max(top_k, 50)

        # 1. Gọi Retrieve từ BM25
        try:
            bm25_results_list = self.bm25.retrieve(query, top_k=search_k)
        except Exception as e:
            logger.error(f"BM25 Retrieval failed: {e}")
            bm25_results_list = []

        # 2. Gọi Retrieve từ Dense
        try:
            dense_results_list = self.dense.retrieve(query, top_k=search_k)
        except Exception as e:
            logger.error(f"Dense Retrieval failed: {e}")
            dense_results_list = []

        # 3. Chuyển đổi List -> Dict để WeightedEnsemble xử lý
        bm25_scores = dict(bm25_results_list)
        dense_scores = dict(dense_results_list)

        # 4. Gọi module Fusion để tính toán và xếp hạng
        return self.fusion.rank(bm25_scores, dense_scores, top_k=top_k)

    def set_weights(self, weights: tuple):
        """
        Cập nhật trọng số (Dùng cho Grid Search hoặc tinh chỉnh runtime).
        """
        self.fusion.set_weights(weights)
        logger.info(f"Weights updated to: {weights}")