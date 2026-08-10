"""
views.py

Handles the chat web interface: renders the chat page and
processes user messages through the AdaptiveRAG pipeline,
persisting each exchange to the database.
"""

from django.shortcuts import render
from .models import ChatMessage
from chat.conversation import ConversationManager

_manager = None


def get_manager() -> ConversationManager:
    """
    Lazily initialize and cache a single ConversationManager
    instance, since loading the LLM and vector store is expensive
    and should only happen once per server process, not per request.

    Returns:
        ConversationManager: The shared conversation manager instance.
    """
    global _manager
    if _manager is None:
        _manager = ConversationManager()
    return _manager


def chat_view(request):
    """
    Render the chat page and handle new user messages via POST.

    Args:
        request: The Django HTTP request object.

    Returns:
        HttpResponse: The rendered chat page with updated history.
    """
    if not request.session.session_key:
        request.session.create()
    session_id = request.session.session_key

    if request.method == "POST":
        user_message = request.POST.get("message", "").strip()
        if user_message:
            manager = get_manager()
            answer = manager.ask(user_message)

            ChatMessage.objects.create(session_id=session_id, role="user", message=user_message)
            ChatMessage.objects.create(session_id=session_id, role="assistant", message=answer)

    messages = ChatMessage.objects.filter(session_id=session_id).order_by("timestamp")
    return render(request, "chatbot/chat.html", {"messages": messages})