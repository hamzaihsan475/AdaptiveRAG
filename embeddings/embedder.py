"""
embedder.py

Generates vector embeddings for text chunks using a
sentence-transformers model, ready for storage in ChromaDB.
"""

from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """
    Wraps a sentence-transformers model to convert text chunks
    into vector embeddings.

    Attributes:
        model_name (str): HuggingFace model identifier for embeddings.
        model: Loaded SentenceTransformer model instance.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize and load the embedding model.

        Args:
            model_name (str): Sentence-transformers model name.
                Defaults to a small, fast, CPU-friendly model.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of text chunks.

        Args:
            texts (list[str]): List of text chunks to embed.

        Returns:
            list[list[float]]: List of embedding vectors, one per chunk.
        """
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()


if __name__ == "__main__":
    sample_chunks = [
        "This endpoint requires a Bearer token for authentication.",
        "Database failovers are handled automatically via replication.",
    ]
    embedder = EmbeddingGenerator()
    vectors = embedder.embed(sample_chunks)
    print(f"Generated {len(vectors)} embeddings, each of length {len(vectors[0])}")
    print("--- First few values of embedding 1 ---")
    print(vectors[0][:5])