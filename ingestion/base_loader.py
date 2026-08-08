"""
base_loader.py

Defines the abstract base class that all document loaders must
implement, ensuring a consistent interface across formats.
"""

from abc import ABC, abstractmethod


class BaseLoader(ABC):
    """
    Abstract base class for all document loaders.

    Any loader for a specific file format (PDF, DOCX, TXT, Excel)
    must inherit from this class and implement the `load` method.
    """

    @abstractmethod
    def load(self, file_path: str) -> str:
        """
        Load and extract raw text content from a document.

        Args:
            file_path (str): Path to the document file.

        Returns:
            str: Extracted plain text content of the document.
        """
        raise NotImplementedError