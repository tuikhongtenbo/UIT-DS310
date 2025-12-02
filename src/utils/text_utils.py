# src/utils/text_utils.py
"""
Text Utilities Module
Text processing and normalization utilities
"""
import re
from typing import List

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

def tokenizer(text: str) -> List[str]:
    if not text:
        return []
    return text.split()

if __name__ == "__main__":
    raw_text = """Điều 1.   Phạm vi điều chỉnh\n\nLuật này quy định về:\t
    1. Phòng, chống tham nhũng;\r2. Xử lý trách nhiệm."""
    print(f"Original text: {repr(raw_text)}")

    normalized = normalize_text(raw_text)
    print(f"Normalized text: {normalized}")

    tokens = tokenizer(normalized)
    print(f"Tokens: {tokens[:10]}")