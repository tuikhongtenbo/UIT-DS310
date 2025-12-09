# src/retrieval/dense_retriever.py
"""
Dense Retriever Module

Semantic retrieval using VietnameseEmbedder and ChromaIndex.
Wrapper class to connect Embedding model and Database.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from src.embeddings.chroma_index import ChromaIndex
from src.embeddings.embedder import VietnameseEmbedder
from src.utils.logger import setup_logger

logger = setup_logger("dense_retriever")


class DenseRetriever:
    """
    Dense Retriever wrapper.
    Uses VietnameseEmbedder for encoding queries and ChromaIndex for retrieval.
    """

    def __init__(self, chroma_index: ChromaIndex, embedder: VietnameseEmbedder):
        """
        Initialize the DenseRetriever.
        """
        self.chroma_index = chroma_index
        self.embedder = embedder
        logger.info("DenseRetriever initialized with provided ChromaIndex and Embedder.")

    def build_index(self, chunks: List[Dict[str, Any]], batch_size: int = 32):
        """
        Delegate indexing task to ChromaIndex.

        Args:
            chunks: List of dictionaries [{'text': '...', 'metadata': {...}}, ...]
            batch_size: Batch size for embedding generation.
        """
        logger.info(f"Delegating {len(chunks)} chunks to ChromaIndex for building...")
        self.chroma_index.build_index(chunks, self.embedder, batch_size=batch_size)

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """
        Retrieve top-k documents semantically.

        Args:
            query: The query string.
            top_k: Number of documents to return.

        Returns:
            List of tuples: (aid, score). Score is Cosine Similarity [0, 1].
        """
        if not query.strip():
            return []

        # 1. Encode query
        try:
            query_embedding = self.embedder.encode(query)
            if hasattr(query_embedding, 'tolist'):
                if len(query_embedding.shape) > 1:
                    query_vec = query_embedding[0].tolist()
                else:
                    query_vec = query_embedding.tolist()
            else:
                query_vec = query_embedding
        except Exception as e:
            logger.error(f"Error encoding query: {e}")
            raise e

        # 2. Query ChromaDB
        results = self.chroma_index.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["metadatas", "distances"]
        )

        if not results or not results.get('ids') or not results['ids'][0]:
            return []

        ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0] if results.get('metadatas') else []

        # Use dict to deduplicate if 1 article has multiple chunks
        retrieved_items = {}

        for i, doc_id in enumerate(ids):
            dist = distances[i]
            score = 1.0 - dist
            score = max(0.0, min(1.0, score))

            # Extract AID from Metadata
            meta = metadatas[i] if i < len(metadatas) else {}
            # Priority: 'aid', 'article_id', or fallback to doc_id
            real_id = str(meta.get('aid') or meta.get('article_id') or doc_id)

            # If 1 article has multiple chunks, keep the highest score chunk
            if real_id in retrieved_items:
                if score > retrieved_items[real_id]:
                    retrieved_items[real_id] = score
            else:
                retrieved_items[real_id] = score

        # Chuyển về list tuple và sort
        sorted_items = sorted(retrieved_items.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:top_k]