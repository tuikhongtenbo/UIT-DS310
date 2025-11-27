"""
Preprocessor Module
Clean and normalize Vietnamese legal text
"""


class Preprocessor:
    """
    Document preprocessor for cleaning and normalizing Vietnamese legal text
    """
    
    def __init__(self, normalize_unicode: bool = True, remove_special_chars: bool = False):
        
        self.normalize_unicode = normalize_unicode
        self.remove_special_chars = remove_special_chars
    
    def preprocess(self, document: str) -> str:
        """
        Preprocess document text: normalize, clean, remove special characters.
        
        Args:
            document: Raw document text
        
        Returns:
            Preprocessed document text
        """
        pass
    
    def preprocess_document(self, document: str) -> str:
        """
        Preprocess document text.
        
        Args:
            document: Raw document text
        
        Returns:
            Preprocessed document text
        """
        return self.preprocess(document)