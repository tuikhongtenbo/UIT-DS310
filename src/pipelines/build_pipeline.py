"""
Build Pipeline Module
Build full retrieval and reranking system
"""


def build_full_pipeline(config_path: str = "configs/main_config.yaml"):
    """
    Build complete retrieval and reranking pipeline.
    
    Steps:
    1. Load configurations
    2. Build BM25 index
    3. Build TF/IDF index
    4. Build ChromaDB index (chunking + embedding)
    5. Initialize retrievers
    6. Initialize rerankers
    
    Args:
        config_path: Path to main configuration file
    
    Returns:
        Complete pipeline system
    """
    pass