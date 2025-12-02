# src/retrieval/dense_retriever.py
"""
Dense Retriever Module
Semantic retrieval using VietnameseEmbedder and ChromaIndex.
Wrapper class to connect Embedding model and Database.
"""
from typing import List, Tuple, Dict, Any, Optional
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

    def __init__(
        self,
        chroma_index: ChromaIndex,
        embedder: VietnameseEmbedder
    ):
        """
        Initialize the DenseRetriever.

        Args:
            chroma_index: Instance of ChromaIndex (already initialized with DB path).
            embedder: Instance of VietnameseEmbedder (already loaded model).
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
            List of tuples: (doc_id, score). Score is Cosine Similarity [0, 1].
        """
        if not query.strip():
            return []

        # 1. Encode query using the centralized Embedder
        try:
            query_embedding = self.embedder.encode(query)

            if isinstance(query_embedding, np.ndarray):
                if len(query_embedding.shape) > 1:
                    query_vec = query_embedding[0].tolist()
                else:
                    query_vec = query_embedding.tolist()
            else:
                query_vec = query_embedding

        except Exception as e:
            logger.error(f"Error encoding query: {e}")
            raise e

        # 2. Query ChromaDB via ChromaIndex
        results = self.chroma_index.query(query_embedding=query_vec, n_results=top_k)

        # 3. Parse Results
        if not results or not results.get('ids') or not results['ids'][0]:
            return []

        ids = results['ids'][0]
        distances = results['distances'][0]

        retrieved_items = []

        for i, doc_id in enumerate(ids):
            dist = distances[i]
            score = 1.0 - dist

            score = max(0.0, min(1.0, score))

            retrieved_items.append((doc_id, float(score)))

        return retrieved_items