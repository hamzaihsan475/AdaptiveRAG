"""
prompt_builder.py

Constructs system and user prompts from retrieved context and
chat history. Single responsibility: prompt construction only.
"""

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


class PromptBuilder:
    """
    Builds the system and user prompts for a single query, given
    retrieved context chunks and recent chat history.
    """

    def build(self, query: str, retrieved_chunks: list[dict],
              recent_history: list[dict]) -> tuple[str, str]:
        """
        Construct the system and user prompts.

        Args:
            query (str): The current user question.
            retrieved_chunks (list[dict]): Chunks with 'text' and 'source' keys.
            recent_history (list[dict]): Recent past turns.

        Returns:
            tuple[str, str]: (system_prompt, user_prompt)
        """
        context = "\n".join(
            f"[Source: {c['source']}] {c['text']}" for c in retrieved_chunks
        )

        history_text = ""
        for turn in recent_history:
            history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

        user_prompt = (
            f"Context:\n{context}\n\n"
            f"{history_text}"
            f"Question: {query}"
        )

        return SYSTEM_PROMPT, user_prompt