import numpy as np
from typing import List, Dict, Any

class WeightedEnsembleRetriever:
    def __init__(self, retrievers: List[Any]):
        """
        retrievers: List chứa các object retriever (BM25, Dense, TFIDF...)
        Thứ tự trong list này ứng với thứ tự của weights w1, w2...
        """
        self.retrievers = retrievers

    def normalize_scores(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chuẩn hóa điểm số về [0, 1] để cộng được với nhau.
        BM25 score có thể là 20, Dense score là 0.8 -> Cần chuẩn hóa.
        """
        if not results:
            return []

        scores = [r['score'] for r in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score == min_score:
            return results # Tránh chia cho 0

        for r in results:
            r['normalized_score'] = (r['score'] - min_score) / (max_score - min_score)

        return results

    def search(self, query: str, weights: List[float], k: int = 10) -> List[Dict[str, Any]]:
        """
        weights: List [w1, w2, ...] tương ứng với self.retrievers
        """
        if len(weights) != len(self.retrievers):
            raise ValueError("Number of weights must match number of retrievers")

        all_results_map = {} # Map id -> {final_score, content, ...}

        # 1. Gọi từng retriever
        for idx, retriever in enumerate(self.retrievers):
            # Lấy nhiều hơn k một chút để tăng khả năng giao thoa (recall phase)
            raw_results = retriever.search(query, k=k*2)

            # Chuẩn hóa điểm số của retriever hiện tại
            norm_results = self.normalize_scores(raw_results)

            weight = weights[idx]

            # Cộng dồn điểm
            for res in norm_results:
                doc_id = res['id'] # ID phải thống nhất giữa các retriever (quan trọng)

                if doc_id not in all_results_map:
                    all_results_map[doc_id] = {
                        'id': doc_id,
                        'content': res.get('content', ''),
                        'final_score': 0.0,
                        'original_scores': {}
                    }

                # Weighted Sum
                # Nếu doc xuất hiện ở retriever này, cộng điểm * weight
                # Nếu không xuất hiện, coi như score = 0
                score_contribution = res['normalized_score'] * weight
                all_results_map[doc_id]['final_score'] += score_contribution
                all_results_map[doc_id]['original_scores'][res['type']] = res['score']

        # 2. Convert map to list & Sort
        final_results = list(all_results_map.values())
        final_results.sort(key=lambda x: x['final_score'], reverse=True)

        return final_results[:k]