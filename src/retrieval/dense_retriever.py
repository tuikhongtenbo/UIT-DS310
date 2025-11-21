"""
Dense Retriever
Semantic retrieval using SentenceBERT embeddings from ChromaDB
"""


class DenseRetriever:
    """
    Dense retriever using SentenceBERT embeddings
    Queries from ChromaDB vector store
    """
    
    def __init__(
        self, 
        model_name: str = "AITeamVN/Vietnamese_Embedding",
        chroma_index=None
    ):
        """
        Initialize dense retriever.
        
        Args:
            model_name: Name of the SentenceBERT model
            chroma_index: ChromaIndex instance (optional)
        """
        pass
    
    def retrieve(self, query: str, top_k: int = 100) -> list:
        """
        Retrieve top-k documents using semantic similarity from ChromaDB.
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of (article_id, score) tuples (aggregated from chunks)
        """
        pass