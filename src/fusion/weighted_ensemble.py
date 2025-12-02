# src/fusion/weighted_ensemble.py
from typing import List, Dict, Tuple

class WeightedEnsemble:
    """
    Pure Logic for Weighted Ensemble Fusion.
    """

    def __init__(self, weights: tuple = (0.5, 0.5)):
        self.weights = weights

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        if not scores: return {}
        values = list(scores.values())
        min_score = min(values)
        max_score = max(values)
        if max_score == min_score: return {k: 1.0 for k in scores}
        return {k: (v - min_score) / (max_score - min_score) for k, v in scores.items()}

    def rank(self, bm25_scores: Dict[str, float], dense_scores: Dict[str, float], top_k: int = 100) -> List[Tuple[str, float]]:
        w1, w2 = self.weights
        norm_bm25 = self._normalize_scores(bm25_scores)
        norm_dense = self._normalize_scores(dense_scores)
        all_ids = set(norm_bm25.keys()) | set(norm_dense.keys())

        final_scores = {}
        for doc_id in all_ids:
            s1 = norm_bm25.get(doc_id, 0.0)
            s2 = norm_dense.get(doc_id, 0.0)
            final_scores[doc_id] = (w1 * s1) + (w2 * s2)

        sorted_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_docs[:top_k]

    def set_weights(self, weights: tuple):
        if len(weights) != 2: raise ValueError("Weights tuple must have exactly 2 elements.")
        self.weights = weights