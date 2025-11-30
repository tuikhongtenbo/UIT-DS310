"""
Dense Retriever
Semantic retrieval using SentenceBERT embeddings from ChromaDB
"""
from typing import List, Tuple, Any
from sentence_transformers import SentenceTransformer

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
        self.chroma_index = chroma_index

        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)

        if self.chroma_index is None:
            print("Warning: chroma_index is None. You must provide a valid ChromaIndex instance to retrieve.")


    def retrieve(self, query: str, top_k: int = 100) -> list:
        """
        Retrieve top-k documents using semantic similarity from ChromaDB.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (article_id, score) tuples (aggregated from chunks)
        """
        if not self.chroma_index:
            raise ValueError("ChromaIndex has not been initialized.")

        # 1. Encode query thành vector
        # BGE-M3 có thể trả về dense, sparse, colbert vec. Ở đây ta dùng dense vec mặc định.
        query_embedding = self.model.encode(query).tolist()

        # 2. Query từ ChromaDB
        # Giả định self.chroma_index có thuộc tính 'collection' là chromadb.Collection
        # Hoặc self.chroma_index chính là collection (tuỳ cách bạn implement file chroma_index.py)
        collection = getattr(self.chroma_index, "collection", self.chroma_index)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances", "documents"]
        )

        # 3. Xử lý kết quả trả về
        # ChromaDB trả về list lồng nhau (batch), ta lấy phần tử đầu tiên
        ids = results['ids'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(ids)

        retrieved_items = []

        for i in range(len(ids)):
            chunk_id = ids[i]
            distance = distances[i]
            metadata = metadatas[i]

            # Chuyển distance thành similarity score
            # Chroma mặc định dùng Cosine Distance (nếu config space='cosine')
            # Score = 1 - Distance
            score = 1.0 - distance

            # Lấy article_id từ metadata nếu có, nếu không thì dùng chunk_id
            # Giả sử metadata lưu field là "article_id" hoặc "doc_id"
            article_id = metadata.get('article_id', metadata.get('doc_id', chunk_id))

            retrieved_items.append((article_id, score))

        # Lưu ý: Vì đây là Dense Retriever trên chunks, kết quả trả về có thể chứa
        # nhiều chunks thuộc cùng 1 article.
        # Nếu bạn muốn aggregate (gộp điểm) theo article_id, bạn có thể xử lý thêm ở đây.
        # Hiện tại trả về raw list theo docstring yêu cầu.

        return retrieved_items