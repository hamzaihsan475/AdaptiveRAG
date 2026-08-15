"""
document_indexer.py

Handles embedding and storing document chunks into the vector
database. Single responsibility: indexing only, not retrieval.
"""

import os
import chromadb
from embeddings.embedder import EmbeddingGenerator


class DocumentIndexer:
    """
    Embeds text chunks and stores them in a persistent ChromaDB
    collection.

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
                Defaults to a chroma_db/ folder at the project root.
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

    def add_chunks(self, chunks: list[str], source: str) -> "DocumentIndexer":
        """
        Embed and store a list of text chunks in the vector store.

        Args:
            chunks (list[str]): Text chunks to store.
            source (str): Identifier for where these chunks came from.

        Returns:
            DocumentIndexer: self, to allow method chaining.
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
        return self