"""
document_retriever.py

Handles searching a ChromaDB collection for chunks relevant to a
query. Single responsibility: retrieval only, not indexing.
"""

import os
import chromadb
from embeddings.embedder import EmbeddingGenerator


class DocumentRetriever:
    """
    Searches a persistent ChromaDB collection for the most similar
    chunks to a given query.

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

    def query(self, query_text: str, top_k: int = 3) -> list[dict]:
        """
        Retrieve the top-k most similar chunks to a query.

        Args:
            query_text (str): The search query.
            top_k (int): Number of top results to return.

        Returns:
            list[dict]: Each dict has 'text' and 'source' keys.
        """
        query_embedding = self.embedder.embed([query_text])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        chunks = results["documents"][0]
        sources = [m["source"] for m in results["metadatas"][0]]
        return [{"text": c, "source": s} for c, s in zip(chunks, sources)]


if __name__ == "__main__":
    from retrieval.document_indexer import DocumentIndexer

    indexer = DocumentIndexer()
    indexer.add_chunks(
        ["Database failovers are handled automatically through replica promotion."],
        source="test",
    )

    retriever = DocumentRetriever()
    results = retriever.query("What happens on a database failover?", top_k=1)
    print(results)