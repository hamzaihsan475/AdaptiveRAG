"""
docx_loader.py

Loader for extracting text content from Word (.docx) files
using python-docx.
"""

from docx import Document
from ingestion.base_loader import BaseLoader


class DOCXLoader(BaseLoader):
    """
    Loads and extracts text from DOCX documents.
    """

    def load(self, file_path: str) -> str:
        """
        Extract text from all paragraphs of a DOCX file.

        Args:
            file_path (str): Path to the .docx file.

        Returns:
            str: Concatenated text content from all paragraphs.
        """
        document = Document(file_path)
        paragraphs = [para.text for para in document.paragraphs if para.text.strip()]
        return "\n".join(paragraphs)