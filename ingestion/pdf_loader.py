"""
pdf_loader.py

Loader for extracting text content from PDF files using pypdf.
"""

from pypdf import PdfReader
from ingestion.base_loader import BaseLoader


class PDFLoader(BaseLoader):
    """
    Loads and extracts text from PDF documents.
    """

    def load(self, file_path: str) -> str:
        """
        Extract text from all pages of a PDF file.

        Args:
            file_path (str): Path to the .pdf file.

        Returns:
            str: Concatenated text content from all pages.
        """
        reader = PdfReader(file_path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts)