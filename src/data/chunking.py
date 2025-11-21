"""
Chunking Module
Split documents into chunks with overlap for better retrieval
"""


def chunk_with_overlap(document: str, chunk_size: int = 2000, overlap: int = 200) -> list:
    """
    Split document into chunks with overlap.
    
    Args:
        document: Input document text
        chunk_size: Maximum size of each chunk (characters, default: 512)
        overlap: Number of overlapping characters between chunks (default: 50)
    
    Returns:
        List of text chunks
    """
    pass