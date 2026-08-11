"""
conversation.py

Orchestrates the full conversational RAG flow: takes a user
query, retrieves relevant context from the vector store, builds
a prompt with chat history and a detailed system prompt, generates
a response via the LLM, and updates the conversation history.
"""

from llm.llm_wrapper import TransformerLLM
from retrieval.vector_store import VectorStore

SYSTEM_PROMPT = """You are AdaptiveRAG, an expert AI Technical Assistant specializing in software architecture, API documentation, and system logs.

### CORE DUTIES & ROLE:
- Your primary objective is to answer technical queries accurately using ONLY the provided document context.
- Maintain a professional, concise, and structured tone suited for software engineers and system architects.

### RULES & COMPLIANCE:
1. STRICT CONTEXT ADHERENCE: Answer using ONLY the information provided in the Context section below. Do NOT use outside knowledge or make assumptions.
2. CITATION & SOURCES: Always reference the source document or section where the information was found (e.g., [Source: API Guide - Section 2]).
3. ZERO HALLUCINATION: If the provided context does not contain enough information to answer the question, state clearly: "I cannot answer this question based on the provided documents."
4. STRUCTURED OUTPUT: Use bullet points, code blocks, or tables wherever applicable to make technical answers readable.
5. NO SPECULATION: Never guess parameters, endpoints, or error codes that are not explicitly written in the context.
"""


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

    def _build_prompt(self, query: str, retrieved_chunks: list[dict]) -> tuple[str, str]:
        """
        Construct the system and user prompts, combining retrieved
        context (with source labels) and recent chat history with
        the current query.

        Args:
            query (str): The current user question.
            retrieved_chunks (list[dict]): Chunks with 'text' and 'source' keys.

        Returns:
            tuple[str, str]: (system_prompt, user_prompt)
        """
        context = "\n".join(
            f"[Source: {c['source']}] {c['text']}" for c in retrieved_chunks
        )

        history_text = ""
        for turn in self.history[-3:]:  # last 3 turns for brevity
            history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

        user_prompt = (
            f"Context:\n{context}\n\n"
            f"{history_text}"
            f"Question: {query}"
        )

        return SYSTEM_PROMPT, user_prompt

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