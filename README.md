# AdaptiveRAG — DevTech Assistant

A modular Retrieval-Augmented Generation (RAG) chatbot that answers technical questions grounded in real software architecture documents, API references, and system error logs — built with Hugging Face Transformers, ChromaDB, and Django.

## Problem It Solves

Developers waste significant time jumping between fragmented internal wikis, API documentation, and system logs to find a single answer — e.g. "what headers does the auth endpoint require" or "how do we handle database failovers per our design guidelines." AdaptiveRAG lets developers, system architects, and technical support staff ask these questions in plain language and get a direct, cited answer pulled from the actual documentation, instead of manually searching multiple sources.

## Case Study: Dev Tech — Software Architecture & API Documentation

- **Target domain:** DevOps & Software Engineering
- **Target users:** Software engineers, system architects, technical support
- **Data sources:** System architecture docs (PDF/TXT), API reference manuals (PDF/TXT), system error logs (TXT)
- **Example queries:**
  - "What are the required headers and error responses for the authentication endpoint?"
  - "What kind of authentication failures appear in the logs?"
  - "How do we handle database failovers per the system design guidelines?"

## Architecture

The system has two pipelines feeding a shared vector store, plus a conversational layer with memory:

- **Ingestion pipeline:** Document loaders (PDF/DOCX/TXT/Excel) → chunking → embedding → ChromaDB
- **Query pipeline:** User query → retrieval → LLM generation (with chat history) → response

See docs/AdaptiveRAG_Diagrams.drawio for the full flow diagram, state diagram, and workflow diagram.

## Tech Stack

- **LLM:** Hugging Face Transformers (Qwen2.5-1.5B-Instruct), with a PEFT/LoRA hook for optional fine-tuning
- **Vector store:** ChromaDB (persistent, local)
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Chunking:** LangChain (RecursiveCharacterTextSplitter)
- **Web framework:** Django, with SQLite for chat history persistence
- **Document parsing:** pypdf, python-docx, openpyxl

## Project Structure

Each pipeline stage is implemented as its own single-responsibility class:
AdaptiveRAG/
├── ingestion/ # PDFLoader, DOCXLoader, TXTLoader, ExcelLoader, LoaderFactory
├── embeddings/ # TextChunker, EmbeddingGenerator
├── retrieval/ # DocumentIndexer, DocumentRetriever
├── llm/ # ModelLoader, TextGenerator
├── chat/ # PromptBuilder, HistoryManager, ConversationOrchestrator
├── chatbot/ # Django app: views, models, templates
├── adaptiverag_web/ # Django project settings
├── sample_data/ # Source documents for ingestion
├── main.py # Terminal entry point (ingests sample_data/, starts chat loop)
├── manage.py # Django entry point
├── requirements.txt
└── README.md


## Setup

1. Clone the repo and create a virtual environment:

python -m venv venv
venv\Scripts\activate


2. Install dependencies:

pip install -r requirements.txt


3. Run Django migrations:

python manage.py migrate


## Running the App

**Option 1 — Django web UI (recommended):**

python manage.py runserver

Open `http://127.0.0.1:8000/` in your browser. Upload a document via the upload form, then ask questions in the chat box.

**Option 2 — Terminal (auto-ingests `sample_data/`):**

python main.py


## Features

- Multi-format document ingestion: PDF, DOCX, TXT, Excel
- User document upload via the web UI, indexed on the fly
- Conversational memory — follow-up questions retain context
- Source-cited answers — responses reference which document they came from
- Persistent chat history (SQLite)
- Grounded generation — system prompt enforces context-only answers, explicitly refuses to guess when information isn't available

## Notes on Model Choice

The system uses `Qwen/Qwen2.5-1.5B-Instruct` for CPU-friendly local inference (no GPU required). This is a small model, so response times on CPU-only hardware can take 1-2 minutes per query — a known tradeoff of running fully offline without paid API costs.

## Future Work

- Persist memory/session state across restarts
- Multi-file batch upload
- Optional PEFT/LoRA fine-tuning on domain-specific data
- Upgrade database to PostgreSQL/MySQL for production use