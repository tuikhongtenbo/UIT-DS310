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
    def __init__(self, reranker_models: List[Union[str, SingleReranker]], rrf_k: int = 60):
        """
        Initialize ensemble reranker.
        """
        self.rrf_k = rrf_k
        self.rerankers = []
        
        # Initialize rerankers
        for model in reranker_models:
            if isinstance(model, str):
                reranker = SingleReranker(
                    model_name=model,
                    trust_remote_code=trust_remote_code
                )
                self.rerankers.append(reranker)
            elif isinstance(model, SingleReranker):
                self.rerankers.append(model)
            else:
                raise ValueError(f"Unsupported reranker type: {type(model)}")
    
    def rerank(self, query: str, documents: List[Union[str, Dict[str, Any]]], top_k: int = None) -> List[Tuple[Union[int, str], float]]:
        """
        Rerank documents using ensemble of rerankers with RRF.
        
        Args:
            query: Search query
            documents: List of document texts or dicts with 'content' key
            top_k: Number of top results to return 
        
        Returns:
            List of (document_index, rrf_score) tuples sorted by RRF score
        """
        if not documents or not self.rerankers:
            return []
        
        # Extract document texts if documents are dicts
        if documents and isinstance(documents[0], dict):
            doc_texts = [doc.get('content', doc.get('text', '')) for doc in documents]
            doc_ids = [
                doc.get('aid', doc.get('id', idx)) 
                for idx, doc in enumerate(documents)
            ]
        else:
            doc_texts = documents
            doc_ids = list(range(len(documents)))
        
        # Get ranked lists from each reranker
        ranked_lists = []
        for reranker in self.rerankers:
            # Rerank documents
            reranked = reranker.rerank(query, doc_texts, top_k=None)
            
            # Map indices to document IDs
            ranked_list = [
                (doc_ids[idx], score) 
                for idx, score in reranked
            ]
            ranked_lists.append(ranked_list)
        
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