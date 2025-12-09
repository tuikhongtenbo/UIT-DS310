"""
Single Reranker Module

Generic reranker using transformers models for sequence classification.
Supports models like BAAI/bge-reranker-v2-m3, Alibaba-NLP/gte-multilingual-reranker-base,
jinaai/jina-reranker-v2-base-multilingual rerankers.
"""

from typing import Any, Dict, List, Tuple, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ..utils.format_reranker_input import (
    extract_ids_from_documents,
    extract_texts_from_documents,
    format_documents_for_reranker
)


class SingleReranker:
    """
    Single reranker model using transformers sequence classification models.
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = None,
        trust_remote_code: bool = True,
        max_length: int = 512
    ):
        """
        Initialize single reranker model.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        self.max_length = max_length
        
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
    
    def rerank(
        self,
        query: str,
        documents: Union[List[str], List[Dict[str, Any]]],
        top_k: int = None
    ) -> List[Tuple[Union[int, str], float]]:
        """
        Rerank documents for a given query.
        
        Args:
            query: Search query
            documents: List of document texts (str) or document dicts with 'content'/'text' and 'aid'/'id'
            top_k: Number of top results to return (None for all)
        
        Returns:
            List of (document_id, score) tuples sorted by score (descending)
            - If documents are strings: returns (index, score)
            - If documents are dicts: returns (aid/id, score)
        """
        if not documents:
            return []
        
        # Format documents
        formatted_docs = format_documents_for_reranker(documents)
        
        # Extract texts for scoring
        doc_texts = extract_texts_from_documents(formatted_docs)
        
        # Extract IDs for result mapping
        doc_ids = extract_ids_from_documents(formatted_docs, id_key="aid")
        
        # Get scores for all query-document pairs
        scores = self.score_batch(query, doc_texts)
        
        # Create list of (doc_id, score) tuples
        results = [(doc_ids[idx], score) for idx, score in enumerate(scores)]
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k if specified
        if top_k is not None:
            return results[:top_k]
        return results
    
    def score(self, query: str, document: Union[str, Dict[str, Any]]) -> float:
        """
        Calculate relevance score between query and document.
        
        Args:
            query: Search query
            document: Document text (str) or document dict with 'content'/'text'
        
        Returns:
            Relevance score
        """
        # Extract text if document is a dict
        if isinstance(document, dict):
            doc_text = document.get("content", document.get("text", ""))
        else:
            doc_text = document
        
        scores = self.score_batch(query, [doc_text])
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
                max_length=self.max_length,
                return_tensors='pt'
            ).to(self.device)
            
            outputs = self.model(**inputs)
            scores = outputs.logits.float().view(-1).cpu().numpy()
        
        return scores.tolist()