"""
urls.py

URL routing for the chatbot app.
"""

from django.urls import path
from . import views

urlpatterns = [
    path("", views.chat_view, name="chat"),
]