# 🦁 Leo RAG Assistant

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

An Retrieval-Augmented Generation (RAG) assistant specifically designed for the [Leo programming language](https://developer.aleo.org/leo/), featuring configurable chunking strategies, hybrid search, and intelligent document processing.

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- [Git](https://git-scm.com/)
- [Google Cloud account](https://cloud.google.com/)
- [ChromaDB](https://www.trychroma.com/) - for local vector store. Consider using [Vector Search](https://cloud.google.com/vertex-ai/docs/vector-search/overview) for production.
- [Google ADK](https://google.github.io/adk-docs/) - for building agents.

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/leo-code-assist.git
   cd leo-code-assist
   ```

2. **Set up Python environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   Copy the example environment file and update it with your credentials:
   ```bash
   cp .env.example .env
   ```

4. **Run Ingestion and Indexing**:
   ```bash
   python scripts/run_indexing.py
   ```

5. **Run Agent**:
   ```bash
   # Make sure your current directory is the root of the project
   adk web
   ```

## 📚 Documentation

See examples on how to run each function under `scripts` directory.

## Project Structure

```
leo_code_assist/
├── data/                    # Data storage
├── src/
│   ├── ingestion/          # Data collection and preprocessing
│   ├── indexing/            # Vector store and embeddings
│   ├── retrieval/           # Retrieval logic
│   ├── generation/          # LLM integration and response generation
│   └── utils/               # Utility functions
├── .env.example             # Example environment variables
├── requirements.txt         # Project dependencies
└── README.md               # This file
```
