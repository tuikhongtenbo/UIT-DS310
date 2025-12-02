# src/data/preprocessor.py
"""
Preprocessor Module
Clean and normalize Vietnamese legal text
"""
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.text_utils import normalize_text
from src.utils.logger import setup_logger

logger = setup_logger("preprocessor")
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
        if not document:
            return ""
        cleaned_text = document
        if self.normalize_unicode:
            cleaned_text = normalize_text(cleaned_text)
        return cleaned_text

    def preprocess_document(self, document: str) -> str:
        """
        Preprocess document text.

        Args:
            document: Raw document text

        Returns:
            Preprocessed document text
        """
        return self.preprocess(document)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_path = os.path.join(project_root, "dataset", "legal_corpus.json")

    print(f"Checking data path: {data_path}")
    if os.path.exists(data_path):
        with open (data_path, "r", encoding = "utf-8") as f:
            data = json.load(f)

        if data and "content" in data[0] and len(data[0]["content"]) > 0:
            raw_text = data[0]['content'][0]['content_Article']
            processor = Preprocessor()
            cleaned_text = processor.preprocess(raw_text)
            print(f"Original text: {repr(raw_text[:300])}")
            print(f"Cleaned text: {repr(cleaned_text[:300])}")
            if "\n" not in cleaned_text:
                print("No newlines found in processed text")
    else:
        print("Dataset not found to run test")