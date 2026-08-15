"""
main.py

Entry point for AdaptiveRAG. Ingests all documents from
sample_data/, indexes them in the vector database (if not
already done), then starts an interactive chat session.
"""

import os

from ingestion.loader_factory import get_loader
from embeddings.chunker import TextChunker
from retrieval.document_indexer import DocumentIndexer
from chat.conversation_orchestrator import ConversationOrchestrator


def ingest_all_documents(data_dir: str, indexer: DocumentIndexer) -> None:
    """
    Load, chunk, and index every supported document in a folder.

    Args:
        data_dir (str): Path to the folder containing source documents.
        indexer (DocumentIndexer): The indexer to populate.
    """
    if indexer.collection.count() > 0:
        print(f"Vector store already has {indexer.collection.count()} chunks — skipping ingestion.")
        return

    chunker = TextChunker(chunk_size=500, chunk_overlap=50)
    supported_extensions = (".pdf", ".docx", ".txt", ".xlsx")

    for filename in os.listdir(data_dir):
        if not filename.lower().endswith(supported_extensions):
            continue

        file_path = os.path.join(data_dir, filename)
        print(f"Ingesting {filename}...")

        loader = get_loader(file_path)
        text = loader.load(file_path)
        chunks = chunker.split(text)

        indexer.add_chunks(chunks, source=filename)
        print(f" -> Added {len(chunks)} chunks from {filename}")


def main() -> None:
    """
    Run the AdaptiveRAG pipeline: ingest documents, then start
    an interactive conversational session.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "sample_data")

    print("Initializing AdaptiveRAG...")
    orchestrator = ConversationOrchestrator()
    indexer = DocumentIndexer()

    ingest_all_documents(data_dir, indexer)

    print("\nAdaptiveRAG is ready. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            break
        print("Thinking...", flush=True)
        answer = orchestrator.ask(user_input)
        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    main()