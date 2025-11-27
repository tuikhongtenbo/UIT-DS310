"""
Chunking Module
Split documents into chunks with overlap for better retrieval
"""


class DocumentChunker:
    """
    Document chunker for splitting documents into chunks with overlap
    """
    
    def __init__(self, chunk_size: int = 2000, overlap: int = 200):
        """
        Initialize document chunker.
        
        Args:
            chunk_size: Maximum size of each chunk (characters, default: 2000)
            overlap: Number of overlapping characters between chunks (default: 200)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, document: str) -> list:
        """
        Split document into chunks with overlap.
        
        Args:
            document: Input document text
        
        Returns:
            List of text chunks
        """
        pass
    
    def chunk_with_overlap(self, document: str) -> list:
        """
        Split document into chunks with overlap.
        
        Args:
            document: Input document text
        
        Returns:
            List of text chunks
        """
        return self.chunk(document)