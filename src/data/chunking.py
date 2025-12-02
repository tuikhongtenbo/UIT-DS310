# src/data/chunking.py
"""
Chunking Module
Split documents into chunks with overlap for better retrieval
"""
import sys
import os
import json
from typing import List
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import setup_logger
from src.data.preprocessor import Preprocessor


class DocumentChunker:
    """
    Document chunker for splitting documents into chunks with overlap
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        """
        Initialize document chunker.

        Args:
            chunk_size: Maximum size of each chunk (characters, default: 2000)
            overlap: Number of overlapping characters between chunks (default: 200)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: str) -> List[str]:
        """
        Split document into chunks with overlap.

        Args:
            document: Input document text

        Returns:
            List of text chunks
        """
        if not document:
            return []
        words = document.split()
        if len(words) <= self.chunk_size:
            return [document]

        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)

            if end >= len(words):
                break
            start += (self.chunk_size - self.overlap)
        return chunks



    def chunk_with_overlap(self, document: str) -> List[str]:
        """
        Split document into chunks with overlap.

        Args:
            document: Input document text

        Returns:
            List of text chunks
        """
        return self.chunk(document)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_path = os.path.join(project_root, "dataset", "legal_corpus.json")

    print(f"Checking data path: {data_path}")
    if os.path.exists(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data and "content" in data[0] and len(data[0]["content"]) > 0:
            raw_text = data[0]['content'][3]['content_Article']
            print(f"Original data: {raw_text}")
            # 1. Preprocess
            preprocessor = Preprocessor()
            clean_text = preprocessor.preprocess(raw_text)

            # 2. Chunk
            chunker = DocumentChunker(chunk_size=512, overlap=64)
            chunks = chunker.chunk(clean_text)
            print(f"Original word count: {len(clean_text.split())}")
            print(f"Generated {len(chunks)} chunks")
            for i, c in enumerate(chunks):
                print(f"Chunk {i+1} - Words: {len(c.split())}")
                print(c[:100] + "..." + c[-50])
    else:
        print(f"Dataset could not be found")