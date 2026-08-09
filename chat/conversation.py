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

    def __init__(self, top_k: int = 3):
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

    def _build_prompt(self, query: str, retrieved_chunks: list[str]) -> str:
        """
        Construct a prompt combining retrieved context, recent
        chat history, and the current user query.

        Args:
            query (str): The current user question.
            retrieved_chunks (list[str]): Relevant chunks from retrieval.

        Returns:
            str: The final prompt to send to the LLM.
        """
        context = "\n".join(retrieved_chunks)

        history_text = ""
        for turn in self.history[-3:]:  # last 3 turns for brevity
            history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

        prompt = (
            f"You are a technical assistant answering questions about "
            f"software architecture, API documentation, and system logs.\n\n"
            f"Relevant context:\n{context}\n\n"
            f"Conversation so far:\n{history_text}\n"
            f"User: {query}\nAssistant:"
        )
        return prompt

    def ask(self, query: str) -> str:
        """
        Process a single user query end-to-end: retrieve context,
        build the prompt, generate a response, and update history.

        Args:
            query (str): The user's question.

        Returns:
            str: The generated response.
        """
        retrieved_chunks = self.vector_store.query(query, top_k=self.top_k)
        prompt = self._build_prompt(query, retrieved_chunks)
        response = self.llm.generate(prompt, max_new_tokens=200)

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