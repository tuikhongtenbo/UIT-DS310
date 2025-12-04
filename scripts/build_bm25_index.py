"""
Build BM25 Index Script
Build BM25 index for retriever module
"""

import sys
import os
import argparse
import time
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from config.config import AppConfig
from src.utils.logger import setup_logger
from src.data.dataloader import VLQALoader
from src.data.preprocessor import Preprocessor
from src.data.chunking import DocumentChunker
from src.embeddings.embedder import VietnameseEmbedder
from src.retrieval.bm25_retriever import BM25Retriever

logger = setup_logger("build_bm25_index")


def build_bm25_index(data, config):
    """
    Build BM25 index for retriever module.
    
    Args:
        data: Legal corpus data
        config: Configuration dictionary
    """
    logger.info("=" * 60)
    logger.info("Building BM25 Index")
    logger.info("=" * 60)
    
    # 1. Get retriever configuration
    retriever_config = config.get('retriever', {})
    embedder_config = retriever_config.get('embedder', {})
    models_config = retriever_config.get('models', {})
    bm25_config = models_config.get('bm25', {})
    
    chunk_size = embedder_config.get('chunk_size', 512)
    overlap = embedder_config.get('overlap', 64)
    model_name = embedder_config.get('model', "AITeamVN/Vietnamese_Embedding")
    device = embedder_config.get('device', "cuda")
    
    k1 = bm25_config.get('k1', 1.5)
    b = bm25_config.get('b', 0.75)
    
    # Get paths from config
    pipeline_config = config.get('pipeline', {})
    indexes_config = pipeline_config.get('indexes', {})
    bm25_index_path = indexes_config.get('bm25_index_path', './data/bm25_index.pkl')
    
    # Make absolute path if relative
    if not os.path.isabs(bm25_index_path):
        bm25_index_path = os.path.join(project_root, bm25_index_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(bm25_index_path), exist_ok=True)
    
    logger.info(f"Configuration:")
    logger.info(f"  - Chunk size: {chunk_size}")
    logger.info(f"  - Overlap: {overlap}")
    logger.info(f"  - Model: {model_name}")
    logger.info(f"  - Device: {device}")
    logger.info(f"  - BM25 k1: {k1}")
    logger.info(f"  - BM25 b: {b}")
    logger.info(f"  - BM25 index path: {bm25_index_path}")
    
    # 2. Initialize components
    logger.info("\nInitializing components...")
    preprocessor = Preprocessor()
    chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
    bm25_retriever = BM25Retriever(embedder=None, k1=k1, b=b)
    logger.info("BM25: Using Vietnamese word tokenization (no embedder needed)")
    
    # 3. Process data & Create chunks
    all_chunks = []
    logger.info("\nProcessing and chunking documents...")
    
    # Count total articles for progress tracking
    total_articles = sum(len(law.get("content", [])) for law in data)
    logger.info(f"Total articles to process: {total_articles}")
    
    start_time = time.time()
    article_count = 0
    for law in tqdm(data, desc="Processing laws", unit="law"):
        law_id = law.get("law_id", "unknown")
        articles = law.get("content", [])
        for article in articles:
            article_count += 1
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
                    },
                    "aid": str(aid)  # BM25 needs aid as id
                }
                all_chunks.append(chunk_record)
    
    chunking_time = time.time() - start_time
    logger.info(f"\nTotal chunks created: {len(all_chunks)}")
    logger.info(f"Average chunks per article: {len(all_chunks) / max(article_count, 1):.2f}")
    logger.info(f"Chunking time: {chunking_time:.2f}s ({len(all_chunks)/chunking_time:.1f} chunks/s)")
    
    # 4. Build BM25 Index
    logger.info("\nBuilding BM25 index...")
    start_time = time.time()
    bm25_retriever.fit(all_chunks)
    indexing_time = time.time() - start_time
    logger.info(f"Indexing time: {indexing_time:.2f}s")
    
    # 5. Save BM25 index
    logger.info(f"\nSaving BM25 index to {bm25_index_path}...")
    bm25_retriever.save(bm25_index_path)
    
    total_time = chunking_time + indexing_time
    logger.info("=" * 60)
    logger.info("BM25 index built successfully!")
    logger.info(f"Total documents indexed: {bm25_retriever.corpus_size}")
    logger.info(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    logger.info(f"Throughput: {len(all_chunks)/total_time:.1f} chunks/second")
    logger.info("=" * 60)


def main():
    """
    Main function to build BM25 index.
    """
    parser = argparse.ArgumentParser(description="Build BM25 Index")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    # 1. Load configuration
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return
    
    try:
        cfg = AppConfig(config_path)
        config = cfg.config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return

    # 2. Load dataset
    corpus_path = config['pipeline']['data']['legal_corpus_path']
    if not os.path.isabs(corpus_path):
        corpus_path = os.path.join(project_root, corpus_path)

    # Ensure dataset path exists
    if not os.path.exists(corpus_path):
        logger.error(f"Dataset file not found: {corpus_path}")
        logger.info(f"Please ensure legal_corpus.json exists")
        return

    logger.info(f"Loading corpus from: {corpus_path}")
    loader = VLQALoader()
    try:
        data = loader.load(corpus_path)
    except Exception as e:
        logger.error(f"Failed to load corpus: {e}")
        return
    
    if not data:
        logger.error("Data is empty")
        return
    
    logger.info(f"Loaded {len(data)} laws from corpus")

    # 3. Build index
    build_bm25_index(data, config)


if __name__ == "__main__":
    main()