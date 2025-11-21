"""
Inference Pipeline Module
Single function to run query through full pipeline
"""


def run(query: str, pipeline=None) -> dict:
    """
    Run query through full retrieval and reranking pipeline.
    
    Steps:
    1. Retrieve top-k using weighted ensemble
    2. Rerank using BGE/Qwen
    3. Select final articles (score > threshold or LLM inference)
    
    Args:
        query: Search query
        pipeline: Pipeline system (if None, will load from config)
    
    Returns:
        Dictionary with relevant articles and scores
    """
    pass