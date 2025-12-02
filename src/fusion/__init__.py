"""
Fusion Module
Weighted ensemble and Reciprocal Rank Fusion
"""
from .rrf import reciprocal_rank_fusion
from .weighted_ensemble import WeightedEnsemble

__all__ = ['reciprocal_rank_fusion', 'WeightedEnsemble']