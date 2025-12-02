"""
Build Index Script
Build BM25, and ChromaDB indexes
"""

import sys
import os
import argparse
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from config.config import cfg
from src.utils.logger import setup_logger
from.src.data.dataloader import VLQALoader
from src.data.preprocessor import Preprocessor
from src.data.chunking import DocumentChunker
from src.embeddings.embedder import VietnameseEmbedder
from src.embeddings.chroma_index import ChromaIndex

logger = setup_logger("build_index")

def build_chroma_index(data, config):
    logger.info("Start ChromaDB index construction")
    # 1. Initialize components
    chunk_size = config.get('retriever', {}).get('embedder', {}).get('chunk_size', 512)
    overlap = config.get('retriever', {}).get('embedder', {}).get('chunk_size', 64)
    model_name = config.get('retriever', {}).get('embedder', {}).get('model', "AITeamVN/Vietnamese_Embedding")

    # Paths
    chroma_path = config.get('pipeline', {}).get('indexes', {}).get('chroma_db_retriever_path', "./data/chroma_db_retriever")
    collection_name = config.get('retriever', {}).get('models', {}).get('sentencebert', {}).get('collection_name', "legal_articles")

    logger.info(f"Params: Chunk = {chunk_size}, Overlap = {overlap}, Model = {model_name}")
    preprocessor = Preprocessor()
    chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
    embedder = VietnameseEmbedder(model_name=model_name)
    chroma_index = ChromaIndex(persist_directory=chroma_path, collection_name=collection_name)

    # 2. Process data & Create chunks
    all_chunks = []
    logger.info("Processing and chunking documents")
    for law in tqdm(data, desc="Processing laws"):
        law_id = law.get("law_id", "unknown")
        articles = law.get("content", [])
        for article in articles:
            aid = article.get("aid", "unknown")
            raw_text = article.get("content_Article", "")
            if not raw_text:
                continue

            # a. Preprocess
            clean_text = preprocessor.preprocess(raw_text)

            # b. Chunking
            text_chunks = chunker.chunk(clean_text)

            # c. Create chunk objects with metadata
            for i, chunk_text in enumerate(text_chunks):
                chunk_record = {
                    "text": chunk_text,
                    "metadata": {
                        "law_id": str(law_id),
                        "aid": str(aid),
                        "chunk_index": i,
                        "source": "legal_corpus"
                    }
                }
                all_chunks.append(chunk_record)
    logger.info(f"Total chunks created: {len(all_chunks)}")

    # 3. Embedding & Indexing
    batch_size = config.get("retriever", {}).get("embedder", {}).get("batch_size", 32)

    # chroma_index.reset()
    chroma_index.build_index(all_chunks, embedder, batch_size=batch_size)
    logger.info(f"ChromaDB index build successfully")

def build_b25_index(data, config):
    pass

def main():
    """
    Main function to build all indexes.

    Steps:
    1. Load legal corpus
    2. Build BM25 index
    3. Build ChromaDB index (chunking + embedding)
    """
    parser = argparse.ArgumentParser(description="Build Search Indexes")
    parser.add_argument("--mode", type=str, default="all", choices=["all", "chroma", "bm25"], help="Which index to build")
    args = parser.parse_args()

    # 1. Load confiuration
    if not cfg:
        logger.error("Configuration is not loaded")
        return

    # 2. Load dataset
    corpus_path = cfg.config['pipeline']['data']['legal_corpus_path']
    if not os.path.isabs(corpus_path):
        corpus_path = os.path.join(project_root, corpus_path)

    loader = VLQALoader()
    try:
        data = loader.load(corpus_path)
    except Exception as e:
        logger.info(f"Failed to load corpus: {e}")
        return
    if not data:
        logger.error(f"Data is empty")
        return

    # 3. Build indexs
    if args.mode in ['all', 'chroma']:
        build_chroma_index(data, cfg.config)
    if args.mode in ['all', 'bm25']:
        build_b25_index(data, cfg.config)

if __name__ == "__main__":
    main()