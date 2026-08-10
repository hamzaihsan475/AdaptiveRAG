"""
models.py

Defines the database model for storing chat conversation history.
"""

from django.db import models


class ChatMessage(models.Model):
    """
    Represents a single message in a chat conversation, either
    from the user or the assistant.

    Attributes:
        session_id (str): Identifier grouping messages into one session.
        role (str): Either 'user' or 'assistant'.
        message (str): The message content.
        timestamp (datetime): When the message was created.
    """

    ROLE_CHOICES = [
        ("user", "User"),
        ("assistant", "Assistant"),
    ]

    session_id = models.CharField(max_length=100)
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"[{self.session_id}] {self.role}: {self.message[:50]}"