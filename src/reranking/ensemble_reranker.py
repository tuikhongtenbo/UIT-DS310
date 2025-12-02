"""
Ensemble Reranker Module
Combine multiple rerankers using Reciprocal Rank Fusion (RRF)
"""
from typing import List, Tuple, Union, Dict, Any
from ..fusion.rrf import reciprocal_rank_fusion
from .single_reranker import SingleReranker


class EnsembleReranker:
    """
    Ensemble reranker that combines multiple rerankers using RRF.
    """
    def __init__(
        self, 
        reranker_models: List[Union[str, SingleReranker]], 
        rrf_k: int = 60,
        trust_remote_code: bool = True,
        model_configs: Dict[str, Dict[str, Any]] = None
    ):
        """
        Initialize ensemble reranker.
        """
        self.rrf_k = rrf_k
        self.rerankers = []
        
        # Initialize rerankers
        for model in reranker_models:
            if isinstance(model, str):
                # Get config for this model if available
                model_config = model_configs.get(model, {}) if model_configs else {}
                reranker = SingleReranker(
                    model_name=model,
                    device=model_config.get("device"),
                    trust_remote_code=trust_remote_code,
                    max_length=model_config.get("max_length", 512)
                )
                self.rerankers.append(reranker)
            elif isinstance(model, SingleReranker):
                self.rerankers.append(model)
            else:
                raise ValueError(f"Unsupported reranker type: {type(model)}")
    
    def rerank(
        self, 
        query: str, 
        documents: Union[List[str], List[Dict[str, Any]]], 
        top_k: int = None
    ) -> List[Tuple[Union[int, str], float]]:
        """
        Rerank documents using ensemble of rerankers with RRF.
        
        Args:
            query: Search query
            documents: List of document texts (str) or dicts with 'content'/'text' and 'aid'/'id'
            top_k: Number of top results to return 
        
        Returns:
            List of (document_id, rrf_score) tuples sorted by RRF score
            - If documents are strings: returns (index, score)
            - If documents are dicts: returns (aid/id, score)
        """
        if not documents or not self.rerankers:
            return []
        
        # SingleReranker now handles both str and dict formats and returns (id, score)
        ranked_lists = []
        for reranker in self.rerankers:
            # Rerank documents - SingleReranker will handle format conversion
            reranked = reranker.rerank(query, documents, top_k=None)
            # reranked is already in format [(doc_id, score), ...]
            ranked_lists.append(reranked)
        
        # Combine using RRF
        rrf_results = reciprocal_rank_fusion(ranked_lists, k=self.rrf_k)
        
        # Return top_k if specified
        if top_k is not None:
            return rrf_results[:top_k]
        return rrf_results
    
    def score(self, query: str, document: Union[str, Dict[str, Any]]) -> float:
        """
        Calculate relevance score using ensemble.
        """
        results = self.rerank(query, [document], top_k=1)
        if results:
            return results[0][1]
        return 0.0