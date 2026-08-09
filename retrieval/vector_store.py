"""
vector_store.py

Manages a persistent ChromaDB vector store — adding document
chunks with their embeddings, and querying for similar chunks
given a search query.
"""

import os
import chromadb
from embeddings.embedder import EmbeddingGenerator


class VectorStore:
    """
    Wraps a persistent ChromaDB collection for storing and
    retrieving text chunks by vector similarity.

    Attributes:
        persist_directory (str): Folder where ChromaDB data is stored.
        collection_name (str): Name of the ChromaDB collection.
        embedder (EmbeddingGenerator): Embedding generator instance.
    """

    def __init__(self, persist_directory: str = None,
                 collection_name: str = "adaptiverag_docs"):
        """
        Initialize the ChromaDB client, collection, and embedder.

        Args:
            persist_directory (str): Local folder for persistent storage.
                Defaults to a chroma_db/ folder at the project root,
                regardless of the current working directory.
            collection_name (str): Name of the collection to use/create.
        """
        if persist_directory is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            persist_directory = os.path.join(base_dir, "chroma_db")

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedder = EmbeddingGenerator()

        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    def add_chunks(self, chunks: list[str], source: str) -> None:
        """
        Embed and store a list of text chunks in the vector store.

        Args:
            chunks (list[str]): Text chunks to store.
            source (str): Identifier for where these chunks came from
                (e.g. the original file name), stored as metadata.
        """
        embeddings = self.embedder.embed(chunks)
        ids = [f"{source}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source} for _ in chunks]

        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
        )

    def query(self, query_text: str, top_k: int = 3) -> list[str]:
        """
        Retrieve the top-k most similar chunks to a query.

        Args:
            query_text (str): The search query.
            top_k (int): Number of top results to return.

        Returns:
            list[str]: The most relevant text chunks.
        """
        query_embedding = self.embedder.embed([query_text])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        return results["documents"][0]


if __name__ == "__main__":
    store = VectorStore()

    sample_chunks = [
        "The authentication endpoint requires a Bearer token in the header.",
        "Database failovers are handled automatically through replica promotion.",
        "Error 429 means the API rate limit has been exceeded, retry after 60 seconds.",
    ]
    store.add_chunks(sample_chunks, source="sample_api_docs")

    results = store.query("What happens on a database failover?", top_k=2)
    print("--- Top matching chunks ---")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r}")