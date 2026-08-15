"""
history_manager.py

Manages conversation turn history for a session. Single
responsibility: history storage/retrieval only.
"""


class HistoryManager:
    """
    Stores and retrieves past conversation turns for one session.

    Attributes:
        history (list[dict]): List of past turns with 'user' and
            'assistant' keys.
    """

    def __init__(self):
        """Initialize an empty history list."""
        self.history: list[dict] = []

    def add_turn(self, user_message: str, assistant_message: str) -> "HistoryManager":
        """
        Record one completed conversational turn.

        Args:
            user_message (str): The user's message.
            assistant_message (str): The assistant's response.

        Returns:
            HistoryManager: self, to allow method chaining.
        """
        self.history.append({"user": user_message, "assistant": assistant_message})
        return self

    def get_recent(self, n: int = 3) -> list[dict]:
        """
        Get the most recent n turns.

        Args:
            n (int): Number of recent turns to return.

        Returns:
            list[dict]: The last n turns.
        """
        return self.history[-n:]