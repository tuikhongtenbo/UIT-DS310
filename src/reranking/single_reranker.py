"""
Single Reranker Module
Generic reranker using transformers models for sequence classification
"""
from typing import List, Tuple, Union
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SingleReranker:
    """
    Single reranker model using transformers sequence classification models.
    Supports models like BAAI/bge-reranker-v2-m3, Alibaba-NLP/gte-multilingual-reranker-base, jinaai/jina-reranker-v2-base-multilingual rerankers.
    """
    
    def __init__(self, model_name: str, device: str = None, trust_remote_code: bool = True):
        """
        Initialize single reranker model.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code
        )
        self.model.to(device)
        self.model.eval()
    
    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Tuple[int, float]]:
        """
        Rerank documents for a given query.
        
        Args:
            query: Search query
            documents: List of document texts to rerank
            top_k: Number of top results to return (None for all)
        
        Returns:
            List of (document_index, score) tuples sorted by score (descending)
        """
        if not documents:
            return []
        
        # Get scores for all query-document pairs
        scores = self.score_batch(query, documents)
        
        # Create list of (index, score) tuples
        results = [(idx, score) for idx, score in enumerate(scores)]
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k if specified
        if top_k is not None:
            return results[:top_k]
        return results
    
    def score(self, query: str, document: str) -> float:
        """
        Calculate relevance score between query and document.
        
        Args:
            query: Search query
            document: Document text
        
        Returns:
            Relevance score
        """
        scores = self.score_batch(query, [document])
        return scores[0]
    
    def score_batch(self, query: str, documents: List[str]) -> List[float]:
        """
        Calculate relevance scores for query-document pairs in batch.
        
        Args:
            query: Search query
            documents: List of document texts
        
        Returns:
            List of relevance scores
        """
        if not documents:
            return []
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Tokenize and get scores
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            outputs = self.model(**inputs)
            scores = outputs.logits.float().view(-1).cpu().numpy()
        
        return scores.tolist()