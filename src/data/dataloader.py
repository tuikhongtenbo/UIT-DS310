# src/data/dataloader.py
"""
Data Loader Module
Load legal corpus from JSON files
"""
import json
import os
import sys
from typing import List, Dict, Any

# Add prroject root to path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import setup_logger

logger = setup_logger("dataloader")

class VLQALoader:
    """
    VLQA dataset loader
    """

    def __init__(self, encoding: str = "utf-8"):

        self.encoding = encoding

    def load(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load legal corpus from file.

        Args:
            file_path: Path to legal corpus JSON file

        Returns:
            List of legal documents
        """
        if not os.path.exists(file_path):
            logger.error(f"Find not found: {file_path}")
            raise FileNotFoundError(f"Find not found {file_path}")
        try:
            logger.info(f"Loading data from {file_path}")
            with open(file_path, "r", encoding=self.encoding) as f:
                data = json.load(f)
            logger.info(f"Successfully loaded {len(data)} records")
            return data
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return []

    def load_legal_corpus(self, file_path: str) -> list:
        """
        Load legal corpus from file.

        Args:
            file_path: Path to legal corpus JSON file

        Returns:
            List of legal documents
        """
        return self.load(file_path)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_path = os.path.join(project_root, "dataset", "legal_corpus.json")

    print(f"Checking data path: {data_path}")
    if os.path.exists(data_path):
        loader = VLQALoader()
        data = loader.load_legal_corpus(data_path)
        if data:
            print("Sample first record")
            print(json.dumps(data[0], ensure_ascii=False, indent=4))
            if "content" in data[0]:
                print(f" Sample Article content: {data[0]['content'][0]['content_Article']}")
    else:
        print(f"Error: Could not file file at: {data_path}")