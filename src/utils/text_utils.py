# src/utils/text_utils.py
"""
Text Utilities Module
Text processing and normalization utilities
"""
import re
import unicodedata
from typing import List

from pyvi import ViTokenizer

def normalize_text(text: str) -> str:
    """
    Normalize Vietnamese text.

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    if not text:
        return ""
    cleaned_text = re.sub(r'\s+', ' ', text)
    return cleaned_text.strip()

def preprocess_for_bm25(sentence: str) -> str:
    """
    Preprocess text for BM25: normalize, clean, lowercase, remove punctuation.
    
    Args:
        sentence: Input text
        
    Returns:
        Preprocessed text ready for tokenization
    """
    if not isinstance(sentence, str):
        return ""
    
    # Normalize unicode (NFC)
    sentence = unicodedata.normalize("NFC", sentence)
    
    # Lowercase
    sentence = sentence.lower().strip()
    
    # Remove URLs and emails
    sentence = re.sub(r"https?://\S+|www\.\S+", "", sentence)
    sentence = re.sub(r"\S+@\S+", "", sentence)
    
    # Remove punctuation, keep only Vietnamese characters, numbers, and spaces
    # À-ỹ covers Vietnamese characters
    sentence = re.sub(r"[^a-zA-Z0-9À-ỹ ]+", " ", sentence)
    
    # Normalize whitespace
    sentence = " ".join(sentence.split())
    
    return sentence

def tokenize_vietnamese(text: str) -> List[str]:
    """
    Tokenize Vietnamese text into words.
    Uses pyvi if available, otherwise falls back to whitespace splitting.
    
    Args:
        text: Input text (should be preprocessed)
        
    Returns:
        List of tokens (words)
    """
    if not text:
        return []
    
    # Preprocess first
    text = preprocess_for_bm25(text)
    if not text:
        return []
    
    # Use pyvi for Vietnamese word tokenization
    tokens = ViTokenizer.tokenize(text).split()
    return tokens

if __name__ == "__main__":
    raw_text = """Điều 1.   Phạm vi điều chỉnh\n\nLuật này quy định về:\t
    1. Phòng, chống tham nhũng;\r2. Xử lý trách nhiệm."""
    print(f"Original text: {repr(raw_text)}")

    normalized = normalize_text(raw_text)
    print(f"Normalized text: {normalized}")

    tokens = tokenize_vietnamese(normalized)
    print(f"Tokens: {tokens[:10]}")