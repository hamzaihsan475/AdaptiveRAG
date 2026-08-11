"""
urls.py

URL routing for the chatbot app.
"""

from django.urls import path
from . import views

urlpatterns = [
    path("", views.chat_view, name="chat"),
    path("upload/", views.upload_view, name="upload"),
]