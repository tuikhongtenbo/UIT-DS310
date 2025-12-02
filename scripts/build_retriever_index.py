"""
Build Retriever Index Script
Build ChromaDB index for retriever module
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
from src.data.dataloader import VLQALoader
from src.data.preprocessor import Preprocessor
from src.data.chunking import DocumentChunker
from src.embeddings.embedder import VietnameseEmbedder
from src.embeddings.chroma_index import ChromaIndex

logger = setup_logger("build_retriever_index")


def build_retriever_index(data, config):
    """
    Build ChromaDB index for retriever module.
    
    Args:
        data: Legal corpus data
        config: Configuration dictionary
    """
    logger.info("=" * 60)
    logger.info("Building Retriever ChromaDB Index")
    logger.info("=" * 60)
    
    # 1. Get retriever configuration
    retriever_config = config.get('retriever', {})
    embedder_config = retriever_config.get('embedder', {})
    
    chunk_size = embedder_config.get('chunk_size', 512)
    overlap = embedder_config.get('overlap', 64)
    model_name = embedder_config.get('model', "AITeamVN/Vietnamese_Embedding")
    batch_size = embedder_config.get('batch_size', 32)
    device = embedder_config.get('device', "cuda")
    
    # Get paths from config
    pipeline_config = config.get('pipeline', {})
    indexes_config = pipeline_config.get('indexes', {})
    chroma_path = indexes_config.get('chroma_db_retriever_path', "./data/chroma_db_retriever")
    
    sentencebert_config = retriever_config.get('models', {}).get('sentencebert', {})
    collection_name = sentencebert_config.get('collection_name', "retriever_legal_articles")
    
    # Make absolute path if relative
    if not os.path.isabs(chroma_path):
        chroma_path = os.path.join(project_root, chroma_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(chroma_path, exist_ok=True)
    
    logger.info(f"Configuration:")
    logger.info(f"  - Chunk size: {chunk_size}")
    logger.info(f"  - Overlap: {overlap}")
    logger.info(f"  - Model: {model_name}")
    logger.info(f"  - Device: {device}")
    logger.info(f"  - Batch size: {batch_size}")
    logger.info(f"  - ChromaDB path: {chroma_path}")
    logger.info(f"  - Collection name: {collection_name}")
    
    # 2. Initialize components
    logger.info("\nInitializing components...")
    preprocessor = Preprocessor()
    chunker = DocumentChunker(chunk_size=chunk_size, overlap=overlap)
    embedder = VietnameseEmbedder(model_name=model_name, device=device)
    chroma_index = ChromaIndex(persist_directory=chroma_path, collection_name=collection_name)
    
    # 3. Process data & Create chunks
    all_chunks = []
    logger.info("\nProcessing and chunking documents...")
    
    # Count total articles for progress tracking
    total_articles = sum(len(law.get("content", [])) for law in data)
    logger.info(f"Total articles to process: {total_articles}")
    
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
                    }
                }
                all_chunks.append(chunk_record)
    
    logger.info(f"\nTotal chunks created: {len(all_chunks)}")
    logger.info(f"Average chunks per article: {len(all_chunks) / max(article_count, 1):.2f}")
    
    # 4. Embedding & Indexing
    logger.info("\nBuilding ChromaDB index...")
    chroma_index.build_index(all_chunks, embedder, batch_size=batch_size)
    
    logger.info("=" * 60)
    logger.info("Retriever ChromaDB index built successfully!")
    logger.info(f"Total documents indexed: {chroma_index.count()}")
    logger.info("=" * 60)


def main():
    """
    Main function to build retriever index.
    """
    parser = argparse.ArgumentParser(description="Build Retriever ChromaDB Index")
    parser.add_argument(
        "--reset", 
        action="store_true", 
        help="Reset existing index before building"
    )
    args = parser.parse_args()

    # 1. Load configuration
    if not cfg:
        logger.error("Configuration is not loaded")
        return

    # 2. Load dataset
    corpus_path = cfg.config['pipeline']['data']['legal_corpus_path']
    if not os.path.isabs(corpus_path):
        corpus_path = os.path.join(project_root, corpus_path)

    # Ensure dataset path exists
    if not os.path.exists(corpus_path):
        logger.error(f"Dataset file not found: {corpus_path}")
        logger.info(f"Please ensure legal_corpus.json exists in the dataset folder")
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
    if args.reset:
        logger.warning("Resetting existing index...")
        retriever_config = cfg.config.get('retriever', {})
        sentencebert_config = retriever_config.get('models', {}).get('sentencebert', {})
        collection_name = sentencebert_config.get('collection_name', "retriever_legal_articles")
        pipeline_config = cfg.config.get('pipeline', {})
        indexes_config = pipeline_config.get('indexes', {})
        chroma_path = indexes_config.get('chroma_db_retriever_path', "./data/chroma_db_retriever")
        if not os.path.isabs(chroma_path):
            chroma_path = os.path.join(project_root, chroma_path)
        os.makedirs(chroma_path, exist_ok=True)
        chroma_index = ChromaIndex(persist_directory=chroma_path, collection_name=collection_name)
        chroma_index.reset()
        logger.info("Index reset completed")
    
    build_retriever_index(data, cfg.config)


if __name__ == "__main__":
    main()