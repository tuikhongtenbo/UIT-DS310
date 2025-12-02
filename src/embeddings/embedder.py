# src/embeddings/embedder.py
"""
Embedder Module
Vietnamese embedding models (AITeamVN, BGE, SentenceBERT)
"""
import sys
import os
import torch
import json
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import setup_logger
logger = setup_logger("embedder")


class VietnameseEmbedder:
    """
    Vietnamese text embedder using AITeamVN/Vietnamese_Embedding
    """

    def __init__(self, model_name: str = "AITeamVN/Vietnamese_Embedding", device: str = None):
        """
        Initialize Vietnamese embedder.

        Args:
            model_name: Name of the embedding model
        """
        self.model_name = model_name
        if not device:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        logger.info(f"Loading embedding model: {model_name} on {self.device}")
        try:
            self.model = SentenceTransformer(model_name, device=self.device)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to loaded model: {e}")
            raise e

    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: List of text strings

        Returns:
            List of embedding vectors
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True, # to keep logs clean in production
            convert_to_numpy=True,
            normalize_embeddings=True # important for consine similarity
        )
        return embeddings

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_path = os.path.join(project_root, "dataset", "legal_corpus.json")

    print(f"Checking data path: {data_path}")
    if os.path.exists(data_path):
        with open (data_path, "r", encoding = "utf-8") as f:
            data = json.load(f)
        sample_text = data[0]["content"][0]["content_Article"]
        print(f"Sample text: {sample_text[:100]}")
        embedder = VietnameseEmbedder()
        vector = embedder.encode(sample_text)
        print(f"Vector shape: {vector.shape}")
        print(F"First 5 values: {vector[0][:5]}")
    else:
        print("Dataset could not be found")