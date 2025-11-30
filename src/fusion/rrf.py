"""
Reciprocal Rank Fusion Module
Combine multiple ranked lists using Reciprocal Rank Fusion (RRF)
"""
from collections import defaultdict
from typing import List, Tuple, Union


def reciprocal_rank_fusion(
        results_list: List[List[Tuple[Union[str, int], float]]], 
        k: int = 60
    ) -> List[Tuple[Union[str, int], float]]:
   
    rrf_scores = defaultdict(float)
    
    # For each ranked list
    for ranked_list in results_list:
        for rank, (item_id, _) in enumerate(ranked_list, start=1):
            # Add RRF score: 1 / (k + rank)
            rrf_scores[item_id] += 1.0 / (k + rank)
    
    # Sort by RRF score 
    results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return results