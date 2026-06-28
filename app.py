import os
import sys
import logging
from typing import List

from core.ingestion import extract_text_from_pdf, split_text
from core.indexing import FAISSVectorStore
from core.generator import GeminiGenerator

# Configure logging to write to a log file instead of stdout to keep CLI clean
log_file = "rag_engine.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file, encoding="utf-8")]
)
logger = logging.getLogger(__name__)

# Constants for index persistence
DB_DIR = "db"
INDEX_PATH = os.path.join(DB_DIR, "index.faiss")
DOC_MAP_PATH = os.path.join(DB_DIR, "doc_map.json")

def print_separator():
    print("=" * 60)

def main():
    print_separator()
    print("      Local Modular RAG Document Search Engine (Gemini & FAISS)      ")
    print_separator()

    # Early check for API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        print("Please set it in your environment before running this app.")
        print("Example (Windows PowerShell): $env:GEMINI_API_KEY='your-key-here'")
        print("Example (Windows Cmd): set GEMINI_API_KEY=your-key-here")
        print_separator()
        sys.exit(1)

    # Ensure database directory exists
    os.makedirs(DB_DIR, exist_ok=True)

    # Initialize indexing and generation modules
    try:
        vector_store = FAISSVectorStore()
        generator = GeminiGenerator(api_key=api_key)
    except Exception as e:
        print(f"Error initializing system: {e}")
        sys.exit(1)

    # Determine if we should load existing index or index a new file
    index_loaded = False
    if os.path.exists(INDEX_PATH) and os.path.exists(DOC_MAP_PATH):
        print(f"Detected existing vector store index in '{DB_DIR}/'.")
        choice = input("Do you want to load the existing index? (y/n) [y]: ").strip().lower()
        if choice in ("", "y", "yes"):
            try:
                print("Loading index from disk...")
                vector_store.load(INDEX_PATH, DOC_MAP_PATH)
                index_loaded = True
                print("Index loaded successfully.")
            except Exception as e:
                print(f"Failed to load existing index: {e}. Proceeding to re-index.")

    if not index_loaded:
        print("\nPlease specify the PDF file path to ingest and index.")
        default_pdf = "FastAPI Complete Course CampusX.pdf"
        if os.path.exists(default_pdf):
            pdf_path_input = input(f"Enter PDF path [{default_pdf}]: ").strip()
            pdf_path = pdf_path_input if pdf_path_input else default_pdf
        else:
            pdf_path = input("Enter PDF path: ").strip()

        if not pdf_path:
            print("No PDF path provided. Exiting.")
            sys.exit(1)

        try:
            print(f"Reading and extracting text from '{pdf_path}'...")
            raw_text = extract_text_from_pdf(pdf_path)
            
            print("Splitting text into chunks...")
            chunks = split_text(raw_text)
            if not chunks:
                print("No chunks generated. Exiting.")
                sys.exit(1)
                
            print("Building vector search index (generating embeddings locally)...")
            vector_store.build_index(chunks)
            
            print(f"Saving vector index to '{DB_DIR}/' for future use...")
            vector_store.save(INDEX_PATH, DOC_MAP_PATH)
            print("Indexing completed and saved.")
        except Exception as e:
            print(f"Failed to ingest/index PDF: {e}")
            sys.exit(1)

    # Interactive Query Loop
    print_separator()
    print("RAG System is ready! Ask your questions below.")
    print("Type 'exit' or 'quit' to end the session.")
    print_separator()

    while True:
        try:
            query = input("\nQuery: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            print("\nSearching index for relevant contexts...")
            # Retrieve top 4 most relevant chunks
            search_results = vector_store.search(query, k=4)
            
            if not search_results:
                print("No relevant information found in the vector store.")
                continue

            # Extract chunks
            context_chunks = [chunk for chunk, dist in search_results]

            # Print sources for transparency
            print(f"\n--- Retriggered Chunks (Top {len(search_results)}) ---")
            for idx, (chunk, dist) in enumerate(search_results, 1):
                preview = chunk.replace('\n', ' ')[:100] + "..."
                print(f"[{idx}] (Distance: {dist:.4f}): {preview}")
            print("-" * 50)

            # Generate and print response from Gemini
            print("Generating answer from Gemini 2.5 Flash...")
            answer = generator.generate_answer(query, context_chunks)
            print(f"\nAnswer:\n{answer}")
            print_separator()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred during query execution: {e}")
            logger.exception("Error in query loop")
            print_separator()

if __name__ == "__main__":
    main()
