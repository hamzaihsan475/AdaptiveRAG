"""
loader_factory.py

Provides a factory function that selects the correct document
loader based on file extension, so calling code doesn't need to
know which loader class to use.
"""

import os
from ingestion.base_loader import BaseLoader
from ingestion.pdf_loader import PDFLoader
from ingestion.docx_loader import DOCXLoader
from ingestion.txt_loader import TXTLoader
from ingestion.excel_loader import ExcelLoader


def get_loader(file_path: str) -> BaseLoader:
    """
    Return the appropriate loader instance for a given file,
    based on its extension.

    Args:
        file_path (str): Path to the document file.

    Returns:
        BaseLoader: An instance of the matching loader class.

    Raises:
        ValueError: If the file extension is not supported.
    """
    extension = os.path.splitext(file_path)[1].lower()

    loaders = {
        ".pdf": PDFLoader,
        ".docx": DOCXLoader,
        ".txt": TXTLoader,
        ".xlsx": ExcelLoader,
    }

    if extension not in loaders:
        raise ValueError(f"Unsupported file type: {extension}")

    return loaders[extension]()


if __name__ == "__main__":
    import os

    # Build an absolute path to sample_data, regardless of working directory
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_file = os.path.join(BASE_DIR, "sample_data", "system_error_log.txt")

    loader = get_loader(test_file)
    content = loader.load(test_file)

    print("--- Extracted Output ---")
    print(content[:500])