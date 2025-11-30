"""
BM25 Retriever
Statistical retrieval using BM25 algorithm
"""
import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Tuple, Any
from src.data.preprocessor import Preprocessor

class BM25Retriever:
    """
    BM25-based retriever for legal documents
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 retriever.

        Args:
            k1: BM25 parameter k1
            b: BM25 parameter b
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents = []
        self.preprocessor = Preprocessor()

    def fit(self, documents: list):
        """
        Build BM25 index from documents.

        Args:
            documents: List of document texts
        """

        if not documents:
            raise ValueError("Document list is empty.")

        self.documents = documents

        # Tokenize toàn bộ corpus
        # Lưu ý: Tokenizer của Preprocessor phải hoạt động tốt với tiếng Việt
        tokenized_corpus = [self.preprocessor.tokenize(doc) for doc in documents]

        # Khởi tạo BM25Okapi với các tham số đã config
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
        print(f"BM25 index built with {len(documents)} documents.")

    def retrieve(self, query: str, top_k: int = 100) -> list:
        """
        Retrieve top-k documents.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (document_id, score) tuples
        """
        if self.bm25 is None:
            raise ValueError("BM25 index is not built. Call fit() first.")

        # Tokenize query giống như lúc fit documents
        tokenized_query = self.preprocessor.tokenize(query)

        # Lấy điểm số cho tất cả document
        scores = self.bm25.get_scores(tokenized_query)

        # Sử dụng numpy để lấy top_k index nhanh hơn
        # argsort trả về index tăng dần -> lấy phần cuối -> đảo ngược
        top_n_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_n_indices:
            score = scores[idx]
            # Chỉ lấy các kết quả có điểm > 0
            if score > 0:
                results.append((int(idx), float(score)))

        return results