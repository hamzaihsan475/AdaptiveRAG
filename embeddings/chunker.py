"""
chunker.py

Splits extracted document text into smaller overlapping chunks,
suitable for embedding and retrieval.
"""


from langchain_text_splitters import RecursiveCharacterTextSplitter
class TextChunker:
    """
    Splits raw text into smaller, overlapping chunks using
    LangChain's RecursiveCharacterTextSplitter.

    Attributes:
        chunk_size (int): Maximum characters per chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize the chunker with size and overlap settings.

        Args:
            chunk_size (int): Maximum characters per chunk. Defaults to 500.
            chunk_overlap (int): Overlap between consecutive chunks. Defaults to 50.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def split(self, text: str) -> list[str]:
        """
        Split a block of text into a list of smaller chunks.

        Args:
            text (str): The full text to split.

        Returns:
            list[str]: List of text chunks.
        """
        return self.splitter.split_text(text)


if __name__ == "__main__":
    sample_text = "This is a test sentence about API authentication. " * 30
    chunker = TextChunker(chunk_size=150, chunk_overlap=30)
    chunks = chunker.split(sample_text)
    print(f"Generated {len(chunks)} chunks")
    print("--- First chunk ---")
    print(chunks[0])