"""
conversation.py

Orchestrates the full conversational RAG flow: takes a user
query, retrieves relevant context from the vector store, builds
a prompt with chat history, generates a response via the LLM,
and updates the conversation history.
"""

from llm.llm_wrapper import TransformerLLM
from retrieval.vector_store import VectorStore


class ConversationManager:
    """
    Manages a single conversational session, combining retrieval,
    generation, and chat history into one interface.

    Attributes:
        llm (TransformerLLM): The language model wrapper.
        vector_store (VectorStore): The retrieval vector store.
        history (list[dict]): List of past turns, each with
            'user' and 'assistant' keys.
        top_k (int): Number of retrieved chunks to use per query.
    """

    def __init__(self, top_k: int = 4):
        """
        Initialize the conversation manager with an LLM, vector
        store, and empty chat history.

        Args:
            top_k (int): Number of chunks to retrieve per query.
        """
        self.llm = TransformerLLM()
        self.vector_store = VectorStore()
        self.history: list[dict] = []
        self.top_k = top_k

    def _build_prompt(self, query: str, retrieved_chunks: list[str]) -> tuple[str, str]:
        """
        Construct the system and user prompts, combining retrieved
        context and recent chat history with the current query.

        Args:
            query (str): The current user question.
            retrieved_chunks (list[str]): Relevant chunks from retrieval.

        Returns:
            tuple[str, str]: (system_prompt, user_prompt)
        """
        context = "\n".join(retrieved_chunks)

        history_text = ""
        for turn in self.history[-3:]:  # last 3 turns for brevity
            history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

        system_prompt = (
            "You are a technical assistant answering questions about software "
            "architecture, API documentation, and system logs. Always answer "
            "using ONLY the provided context below. If the context doesn't "
            "contain the answer, say so directly instead of asking the user "
            "to provide the document."
        )

        user_prompt = (
            f"Context:\n{context}\n\n"
            f"{history_text}"
            f"Question: {query}"
        )

        return system_prompt, user_prompt

    def ask(self, query: str) -> str:
        retrieved_chunks = self.vector_store.query(query, top_k=self.top_k)
        print("--- DEBUG: Retrieved chunks ---")
        for i, c in enumerate(retrieved_chunks, 1):
            print(f"[{i}] {c}")
        print("--- END DEBUG ---")

        system_prompt, user_prompt = self._build_prompt(query, retrieved_chunks)
        response = self.llm.generate_chat(system_prompt, user_prompt, max_new_tokens=200)

        self.history.append({"user": query, "assistant": response})
        return response


if __name__ == "__main__":
    manager = ConversationManager()

    print("Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            break
        print("Thinking...", flush=True)
        answer = manager.ask(user_input)
        print(f"Assistant: {answer}\n")