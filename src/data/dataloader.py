"""
Data Loader Module
Load legal corpus from JSON files
"""


class VLQALoader:
    """
    VLQA dataset loader 
    """
    
    def __init__(self, encoding: str = "utf-8"):

        self.encoding = encoding
    
    def load(self, file_path: str) -> list:
        """
        Load legal corpus from file.
        
        Args:
            file_path: Path to legal corpus JSON file
        
        Returns:
            List of legal documents
        """
        pass
    
    def load_legal_corpus(self, file_path: str) -> list:
        """
        Load legal corpus from file.
        
        Args:
            file_path: Path to legal corpus JSON file
        
        Returns:
            List of legal documents
        """
        return self.load(file_path)