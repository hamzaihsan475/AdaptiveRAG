"""
txt_loader.py

Loader for extracting text content from plain text (.txt) files.
"""

from ingestion.base_loader import BaseLoader


class TXTLoader(BaseLoader):
    """
    Loads and extracts text from plain text documents.
    """

    def load(self, file_path: str) -> str:
        """
        Read the full content of a plain text file.

        Args:
            file_path (str): Path to the .txt file.

        Returns:
            str: Full text content of the file.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()