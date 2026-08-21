import os
from langchain_chroma import Chroma
from embeddings.embedder import EmbeddingGenerator


class DocumentIndexer:


    def __init__(self, persist_directory: str = None,
                 collection_name: str = "adaptiverag_docs"):

        if persist_directory is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            persist_directory = os.path.join(base_dir, "chroma_db")

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedder = EmbeddingGenerator()

        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedder.embeddings,
            persist_directory=self.persist_directory,
        )

    def add_chunks(self, chunks: list[str], source: str) -> "DocumentIndexer":

        metadatas = [{"source": source} for _ in chunks]
        self.vectorstore.add_texts(texts=chunks, metadatas=metadatas)
        return self

    @property
    def collection(self):
        """Expose the underlying Chroma collection for count checks."""
        return self.vectorstore._collection