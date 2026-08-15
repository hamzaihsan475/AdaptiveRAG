"""
conversation_orchestrator.py

Wires together retrieval, prompt building, generation, and
history management into one conversational flow. This is the
single orchestrating class — each stage it calls is its own
dedicated single-responsibility class.
"""

from llm.model_loader import ModelLoader
from llm.text_generator import TextGenerator
from retrieval.document_retriever import DocumentRetriever
from chat.prompt_builder import PromptBuilder
from chat.history_manager import HistoryManager


class ConversationOrchestrator:
    """
    Orchestrates a full conversational RAG turn by calling each
    dedicated stage class in order: retrieve, build prompt,
    generate, record history.

    Attributes:
        retriever (DocumentRetriever): Retrieves relevant chunks.
        prompt_builder (PromptBuilder): Builds prompts.
        generator (TextGenerator): Generates responses.
        history (HistoryManager): Tracks conversation history.
        top_k (int): Number of chunks to retrieve per query.
    """

    def __init__(self, top_k: int = 4):
        """
        Initialize each stage class needed for a conversation.

        Args:
            top_k (int): Number of chunks to retrieve per query.
        """
        loader = ModelLoader()
        self.generator = TextGenerator(loader.model, loader.tokenizer, loader.device)
        self.retriever = DocumentRetriever()
        self.prompt_builder = PromptBuilder()
        self.history = HistoryManager()
        self.top_k = top_k

    def ask(self, query: str) -> str:
        """
        Process one user query through the full pipeline: retrieve,
        build prompt, generate, record history.

        Args:
            query (str): The user's question.

        Returns:
            str: The generated response.
        """
        retrieved_chunks = self.retriever.query(query, top_k=self.top_k)
        recent_history = self.history.get_recent(3)
        system_prompt, user_prompt = self.prompt_builder.build(
            query, retrieved_chunks, recent_history
        )
        response = self.generator.generate_chat(system_prompt, user_prompt, max_new_tokens=200)

        self.history.add_turn(query, response)
        return response


if __name__ == "__main__":
    orchestrator = ConversationOrchestrator()

    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            break
        print("Thinking...", flush=True)
        answer = orchestrator.ask(user_input)
        print(f"Assistant: {answer}\n")