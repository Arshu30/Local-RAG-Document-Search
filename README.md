# Local Modular RAG Document Search Engine

A local, modular Retrieval-Augmented Generation (RAG) Document Search Engine built using Python. This engine processes PDF documents, indexes their chunks into a local FAISS vector database using `sentence-transformers` embeddings, and queries Google Gemini 2.5 using the official `google-genai` SDK.

## Features
- **Raw Text Extraction**: Extracts text content from local PDF files page-by-page.
- **Recursive Chunking**: Splits document text into chunks of 600 characters with a 120-character overlap.
- **Local Embeddings**: Generates vector representations locally using the HuggingFace `all-MiniLM-L6-v2` SentenceTransformer model (no API key needed for embedding).
- **FAISS Vector Store**: Fast L2 index search for retrieving relevant contexts locally.
- **Context-Restricted Answers**: Utilizes the modern `google.genai` SDK with `gemini-2.5-flash` and strict system instructions to force the model to answer *only* from the retrieved document context.

## File Structure
- `core/ingestion.py`: Handles PDF text extraction and character chunk splitting.
- `core/indexing.py`: Manages the FAISS index building, similarity search, and persistence.
- `core/generator.py`: Connects to Gemini API using the `google-genai` client.
- `app.py`: Simple CLI orchestrator with an interactive prompt query loop.
- `test_rag.py`: Automated pipeline test verification script (runs headless tests).
- `requirements.txt`: Python package requirements.
- `.gitignore`: Prevents temporary log and cache tracking.

## Setup Instructions

### 1. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Configure Gemini API Key
Provide your Google Gemini API Key by setting the `GEMINI_API_KEY` environment variable.

- **Windows PowerShell**:
  ```powershell
  $env:GEMINI_API_KEY="your-gemini-api-key-here"
  ```
- **Windows Command Prompt**:
  ```cmd
  set GEMINI_API_KEY=your-gemini-api-key-here"
  ```
- **Linux/macOS**:
  ```bash
  export GEMINI_API_KEY="your-gemini-api-key-here"
  ```

## Usage

### Run the Interactive App
Start the terminal interface by executing:
```bash
python app.py
```
On startup:
- The app will automatically prompt you for a PDF file path to ingest. (If you press enter, it defaults to the file `FastAPI Complete Course CampusX.pdf` if available).
- Once indexed, you can ask queries continuously in the interactive terminal loop.
- Type `exit` or `quit` to leave.

### Run Verification Test Loop
You can run the headless verification test script to verify components:
```bash
python test_rag.py
```
This script builds a temporary PDF `sample_paper.pdf` containing explicit key facts, runs the entire ingestion/indexing loop, queries the RAG generator (using a mock if no `GEMINI_API_KEY` is present), and verifies response accuracy before cleaning up after itself.
