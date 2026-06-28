Markdown
# 🔍 Local Modular RAG Document Search Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: Streamlit](https://img.shields.io/badge/Framework-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Model: Gemini 2.5 Flash](https://img.shields.io/badge/Model-Gemini%202.5%20Flash-orange.svg)](https://ai.google.dev/)

A production-grade, highly modular Retrieval-Augmented Generation (RAG) Document Search Engine engineered in Python. The system processes complex PDF documentation locally, generates semantic vector representations on your CPU via `sentence-transformers`, compiles an optimized local L2 index using `faiss`, and synthesizes grounded answers using the official, modern `google-genai` SDK with `gemini-2.5-flash`.

---

## 🚀 Key Architectural Features

- **Page-Aware Metadata Tracking**: Extracts text content page-by-page, mapping absolute page boundaries into chunk metadata to support pinpoint source tracking and global range queries.
- **Recursive Text Chunking**: Leverages `RecursiveCharacterTextSplitter` to parse raw strings into 600-character segments with a 120-character semantic overlap, keeping parent context intact.
- **Zero-Cost Local Embeddings**: Computes 384-dimensional dense vectors completely offline using the HuggingFace `all-MiniLM-L6-v2` model.
- **High-Performance Vector Index**: Uses Facebook AI Similarity Search (`faiss`) to create memory-mapped flat L2 coordinates for microscopic lookup times.
- **Strict Factual Guardrails**: Implements zero-temperature deterministic reasoning tied to explicit system instructions, forcing the LLM to completely drop hallucinations and answer *only* if the facts exist within the local context.
- **Dual Interface Framework**: Supports both an interactive, low-latency command-line loop (`app.py`) and an elegant, reactive dashboard UI (`ui.py`) built with Streamlit.

---

## 📂 Codebase Architecture

```text
local-rag-search/
│
├── core/
│   ├── __init__.py          # Package initialization marker
│   ├── ingestion.py         # Page-boundary tracking & chunk splitting
│   ├── indexing.py          # Local FAISS index lifecycle & similarity search
│   └── generator.py         # Google GenAI client binding & system instructions
│
├── app.py                   # Continuous CLI orchestration loop
├── ui.py                    # Drag-and-drop Streamlit Web application Dashboard
├── test_rag.py              # Headless continuous verification testing script
├── requirements.txt         # Package dependency manifest
└── .gitignore               # System cache and index state exclusion
🛠️ Installation & Setup
1. Clone the Workspace
Bash
git clone [https://github.com/Arshu30/Local-RAG-Document-Search.git](https://github.com/Arshu30/Local-RAG-Document-Search.git)
cd Local-RAG-Document-Search
2. Install Package Dependencies
Ensure your local Python environment is initialized, then install the system manifest:

Bash
pip install -r requirements.txt
3. Configure Authentication Environment
The google-genai engine looks dynamically for standard environment keys. Export your credential string based on your terminal layout:

Windows PowerShell:

PowerShell
$env:GEMINI_API_KEY="AIzaSyYourActualSecretGeminiKeyHere"
Windows Command Prompt (cmd):

DOS
set GEMINI_API_KEY=AIzaSyYourActualSecretGeminiKeyHere
Linux / macOS terminal:

Bash
export GEMINI_API_KEY="AIzaSyYourActualSecretGeminiKeyHere"
💻 Running the Application
Option A: The Streamlit Graphical Dashboard UI (Recommended)
Launch the responsive browser interface to drag and drop PDFs, view expandable vector context distances, and chat visually:

Bash
streamlit run ui.py
Option B: The Terminal CLI Orchestration Loop
Run the continuous terminal execution stream directly within your console:

Bash
python app.py
Workflow Loop: On boot, input your file path (e.g., FastAPI Complete Course CampusX.pdf). The console will stream progress tickers as it extracts pages, updates indices, and shifts into the prompt loop.

Option C: Automated Headless Validation Test
Execute the self-contained pipeline validation test to verify math bounds and vector integrity:

Bash
python test_rag.py
This script injects a mock text layer, tracks token structures across the indexing workflow, checks distance calculations, and verifies system sanity before cleanly removing temporary evaluation artifacts.

⚖️ License
Distributed under the permissive MIT License. See LICENSE for complete details.
