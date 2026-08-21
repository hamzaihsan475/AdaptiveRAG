import os
from langchain_chroma import Chroma
from embeddings.embedder import EmbeddingGenerator


class DocumentRetriever:


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

    def query(self, query_text: str, top_k: int = 3) -> list[dict]:

        results = self.vectorstore.similarity_search(query_text, k=top_k)
        return [{"text": doc.page_content, "source": doc.metadata.get("source", "unknown")} for doc in results]


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