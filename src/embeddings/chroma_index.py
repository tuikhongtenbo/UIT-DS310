# src/embeddings/chroma_index.py
"""
ChromaDB Index Module
ChromaDB storage & retrieval for embeddings
"""
import os
import sys
import uuid
import json
import shutil
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.logger import setup_logger

logger = setup_logger("chroma_index")


class ChromaIndex:
    """
    ChromaDB index for storing and querying embeddings
    """

    def __init__(self, persist_directory: str = "./data/chroma_db", collection_name: str = "legal_articles_chunks"):
        """
        Initialize ChromaDB index.

        Args:
            persist_directory: Directory to persist ChromaDB
            collection_name: Name of the collection
        """
        # Handle read-only paths by copying to working directory
        actual_path = self._resolve_db_path(persist_directory)
        
        self.persist_directory = actual_path
        self.collection_name = collection_name
        logger.info(f"Initialize ChromaDB at {actual_path}")

        # Initialize PersistentClient
        self.client = chromadb.PersistentClient(path=actual_path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Collection: {collection_name} is ready. Current count: {self.collection.count()}")
    
    def _resolve_db_path(self, path: str) -> str:
        """
        Resolve database path, copying from read-only location if needed.
        
        Args:
            path: Original database path
            
        Returns:
            Resolved path (writable location)
        """
        # If path is in Kaggle input (read-only), copy to working directory
        if '/kaggle/input/' in path:
            working_path = path.replace('/kaggle/input/', '/kaggle/working/')
            
            # If copy doesn't exist, create it
            if not os.path.exists(working_path):
                logger.info(f"Copying ChromaDB from read-only to writable location...")
                logger.info(f"  From: {path}")
                logger.info(f"  To:   {working_path}")
                try:
                    os.makedirs(os.path.dirname(working_path), exist_ok=True)
                    # Use dirs_exist_ok for Python 3.8+, fallback for older versions
                    try:
                        shutil.copytree(path, working_path, dirs_exist_ok=True)
                    except TypeError:
                        if os.path.exists(working_path):
                            shutil.rmtree(working_path)
                        shutil.copytree(path, working_path)
                    logger.info("Database copied successfully")
                except Exception as e:
                    logger.error(f"Failed to copy database: {e}")
                    raise RuntimeError(f"Cannot copy ChromaDB from {path} to {working_path}: {e}")
            else:
                logger.info(f"Using existing copy at: {working_path}")
            
            return working_path
        
        return path

    def build_index(self, chunks: List[Dict[str, Any]], embedder, batch_size: int = 32):
        """
        Build ChromaDB index from chunks.

        Args:
            chunks: List of chunk dictionaries
            embedder: Embedding model
            batch_size: Batch size for embedding
        """
        logger.info(f"Start to build index for {len(chunks)} chunks")
        total_chunks = len(chunks)
        # Process in batches
        for i in tqdm(range(0, total_chunks, batch_size), desc="Indexing batches"):
            batch = chunks[i: i + batch_size]
            # Prepare data lists
            batch_texts = [item['text'] for item in batch]
            batch_metadatas =[item['metadata'] for item in batch]
            batch_ids = [str(uuid.uuid4()) for _ in batch]

            # 1. Embed text
            try:
                embeddings = embedder.encode(batch_texts, batch_size=batch_size)
                embeddings_list = embeddings.tolist()

                # 2. Add to ChromaDB
                self.collection.add(
                    documents=batch_texts,
                    embeddings=embeddings_list,
                    metadatas=batch_metadatas,
                    ids=batch_ids
                )
            except Exception as e:
                logger.error(f"Error indexing batch {i}: {e}")
                raise
        logger.info(f"Indexing completed. Total documents in DB: {self.collection.count()}")


    def query(self, query_embedding: List[float], n_results: int = 10, where: dict = None) -> Dict:
        """
        Query ChromaDB with embedding.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            where: Optional metadata filter

        Returns:
            Dictionary with ids, distances, documents, metadatas
        """
        try:
            results = self.collection.query(
                query_embeddings= [query_embedding],
                n_results=n_results,
                where=where
            )
            return results
        except Exception as e:
            logger.error(f"Error during query: {e}")
            return {}


    def get_chunk_mapping(self) -> dict:
        """
        Get mapping from chunk_id to article_id.

        Returns:
            Dictionary mapping chunk_id -> article_id
        """
        pass

    def count(self) -> int:
        """Get number of chunks in collection"""
        return self.collection.count()

    def reset(self):
        "Delete and recreate collection"
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

if __name__ == "__main__":
    import shutil
    from src.data.preprocessor import Preprocessor
    from src.data.chunking import DocumentChunker
    from src.embeddings.embedder import VietnameseEmbedder

    TEST_DB_PATH = "./data/chroma_db_test"
    if os.path.exists(TEST_DB_PATH):
        shutil.rmtree(TEST_DB_PATH) # Clean up
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_path = os.path.join(project_root, "dataset", "legal_corpus.json")

    print(f"Checking data path: {data_path}")
    if os.path.exists(data_path):
        processor = Preprocessor()
        chunker = DocumentChunker(512, 64)
        embedder = VietnameseEmbedder(device="cpu")
        indexer = ChromaIndex(persist_directory=TEST_DB_PATH, collection_name="test_collection")

        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)[:2]

        # Prepare chunks
        all_chunks = []
        for law in raw_data:
            for article in law['content']:
                text = article['content_Article']
                aid = article['aid']

                # Preprocess & Chunk
                clean_text = processor.preprocess(text)
                text_chunks = chunker.chunk(clean_text)
                for chunk_text in text_chunks:
                    all_chunks.append(
                        {"text": chunk_text,
                        "metadata": {"aid": aid,
                                     "law_id": law["law_id"]}
                        }
                    )
        print(f"Generated {len(all_chunks)} chunks from sample")

        # 2. Build index
        indexer.build_index(all_chunks, embedder, batch_size=4)

        # 3. Test query
        query_text = "Trách nhiệm phòng chống tham nhũng"
        query_vec = embedder.encode(query_text)[0].tolist()
        results = indexer.query(query_vec, n_results=2)

        print(f"Query: {query_text}")
        print("Top results")
        for i, doc in enumerate(results["documents"][0]):
            meta = results['metadatas'][0][i]
            print(f"[{i+1}] Law {meta['law_id']}, Article {meta['aid']}: {doc[:100]}")

        # shutil.rmtree(TEST_DB_PATH)
    else:
        print("Dataset could not be found")

