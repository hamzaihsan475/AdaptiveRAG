"""
views.py

Handles the chat web interface: renders the chat page, processes
user messages through the AdaptiveRAG pipeline, and handles
document uploads that get added to the shared vector store.
"""

import os
from django.conf import settings
from django.shortcuts import render, redirect
from .models import ChatMessage
from chat.conversation import ConversationManager
from ingestion.loader_factory import get_loader
from embeddings.chunker import TextChunker

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

    upload_message = request.session.pop("upload_message", None)
    messages = ChatMessage.objects.filter(session_id=session_id).order_by("timestamp")
    return render(request, "chatbot/chat.html", {"messages": messages, "upload_message": upload_message})


def upload_view(request):
    """
    Handle user-uploaded documents: save temporarily, extract text
    via the existing ingestion loaders, chunk it, and add it to
    the shared vector store.

    Args:
        request: The Django HTTP request object.

    Returns:
        HttpResponse: Redirects back to the chat page with a status message.
    """
    if request.method == "POST" and request.FILES.get("document"):
        uploaded_file = request.FILES["document"]

        upload_dir = os.path.join(settings.BASE_DIR, "sample_data", "uploads")
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, uploaded_file.name)

        with open(file_path, "wb+") as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        try:
            loader = get_loader(file_path)
            text = loader.load(file_path)
            chunker = TextChunker(chunk_size=500, chunk_overlap=50)
            chunks = chunker.split(text)

            manager = get_manager()
            manager.vector_store.add_chunks(chunks, source=uploaded_file.name)

            request.session["upload_message"] = f"'{uploaded_file.name}' uploaded and indexed ({len(chunks)} chunks)."
        except ValueError as e:
            request.session["upload_message"] = f"Upload failed: {e}"

    return redirect("chat")