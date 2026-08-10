"""
urls.py

Root URL configuration for the AdaptiveRAG web project.
"""

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("chatbot.urls")),
]