# src/retrieval/bm25_retriever.py
"""
BM25 Retriever (Optimized with Inverted Index)
Statistical retrieval using BM25 algorithm.
"""
import math
import os
import pickle
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any, Optional
from src.utils.logger import setup_logger
from src.utils.text_utils import tokenize_vietnamese

logger = setup_logger("bm25_retriever")

class BM25Retriever:
    """
    BM25-based retriever optimized with Inverted Index.
    Uses Vietnamese word tokenization for proper BM25 scoring.
    """

    def __init__(self, embedder: Any = None, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 Retriever.

        Args:
            embedder: DEPRECATED - No longer used. Kept for backward compatibility.
                      BM25 now uses Vietnamese word tokenization instead of BERT tokenizer.
            k1: BM25 k1 parameter (term frequency saturation).
            b: BM25 b parameter (length normalization).
        """
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0

        # Data structures for BM25
        self.doc_lengths = {}
        self.doc_ids = []
        self.index_to_id = {}
        self.inverted_index = defaultdict(dict)
        self.idf = {}

        # Use Vietnamese word tokenization instead of BERT tokenizer
        # This is critical for BM25 to work properly with Vietnamese text
        self.tokenizer = tokenize_vietnamese
        
        if embedder is not None:
            logger.warning("BM25: embedder parameter is deprecated. BM25 now uses Vietnamese word tokenization (pyvi) instead of BERT tokenizer.")
        
        logger.info("BM25: Using Vietnamese word tokenization for proper BM25 scoring.")

    def _calc_idf(self, doc_freq_count: Dict[str, int]):
        """Calculate IDF for all terms."""
        idf_dict = {}
        N = self.corpus_size
        for term, freq in doc_freq_count.items():
            n_q = freq
            idf_score = math.log(((N - n_q + 0.5) / (n_q + 0.5)) + 1)
            idf_dict[term] = idf_score
        return idf_dict

    def fit(self, documents: List[Dict[str, Any]]):
        """
        Build BM25 index from list of chunks.

        Args:
            documents: List of dicts.
                       Format: [{'text': '...', 'id': '...'}, ...]
                       Or: [{'content_Article': '...', 'aid': '...'}]
        """
        if not documents:
            logger.error("Document list is empty.")
            raise ValueError("Document list is empty.")

        self.corpus_size = len(documents)

        self.doc_lengths = {}
        self.doc_ids = []
        self.index_to_id = {}
        self.inverted_index = defaultdict(dict)

        global_term_doc_count = Counter()
        total_length = 0

        logger.info(f"Building BM25 index for {self.corpus_size} documents...")

        for idx, doc in enumerate(documents):
            # 1. Resolve ID
            d_id = str(doc.get('id',
                        doc.get('metadata', {}).get('doc_id',
                        doc.get('aid', str(idx)))))

            self.doc_ids.append(d_id)
            self.index_to_id[idx] = d_id

            # 2. Tokenize Text using the configured tokenizer
            text_content = doc.get('text', doc.get('content_Article', ''))

            tokens = self.tokenizer(text_content)

            length = len(tokens)
            self.doc_lengths[idx] = length
            total_length += length

            # 3. Build Inverted Index
            term_counts = Counter(tokens)

            global_term_doc_count.update(term_counts.keys())

            for term, count in term_counts.items():
                self.inverted_index[term][idx] = count

        self.avgdl = total_length / self.corpus_size if self.corpus_size > 0 else 0
        self.idf = self._calc_idf(global_term_doc_count)

        logger.info(f"BM25 index built successfully. Docs: {self.corpus_size}, AvgDL: {self.avgdl:.2f}")

    def retrieve(self, query: str, top_k: int = 100) -> List[Tuple[str, float]]:
        """
        Retrieves documents based on query.
        """
        if self.corpus_size == 0:
            logger.warning("BM25 index is empty. Please call fit() or load() first.")
            return []

        query_tokens = self.tokenizer(query)
        scores = defaultdict(float)

        for token in query_tokens:
            if token not in self.inverted_index:
                continue

            idf_score = self.idf[token]

            for doc_idx, term_freq in self.inverted_index[token].items():
                doc_len = self.doc_lengths[doc_idx]

                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl))
                score = idf_score * (numerator / denominator)

                scores[doc_idx] += score

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for doc_idx, score in sorted_scores:
            real_id = self.index_to_id[doc_idx]
            results.append((real_id, float(score)))

        return results

    def save(self, path: str):
        """Save BM25 index to file using pickle."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump({
                    'corpus_size': self.corpus_size,
                    'avgdl': self.avgdl,
                    'doc_lengths': self.doc_lengths,
                    'doc_ids': self.doc_ids,
                    'index_to_id': self.index_to_id,
                    'inverted_index': self.inverted_index,
                    'idf': self.idf
                }, f)
            logger.info(f"BM25 index saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")
            raise e

    def load(self, path: str):
        """Load BM25 index from file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"BM25 Index file not found: {path}")

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
                self.corpus_size = data['corpus_size']
                self.avgdl = data['avgdl']
                self.doc_lengths = data['doc_lengths']
                self.doc_ids = data['doc_ids']
                self.index_to_id = data.get('index_to_id', {})
                self.inverted_index = data['inverted_index']
                self.idf = data['idf']
            logger.info(f"BM25 index loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            raise e