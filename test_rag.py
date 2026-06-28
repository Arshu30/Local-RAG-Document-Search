import os
import sys
import logging
from typing import List
from unittest.mock import MagicMock, patch

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Try to import reportlab to generate the PDF
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
except ImportError:
    logger.error("reportlab is not installed. Please run pip install reportlab.")
    sys.exit(1)

from core.ingestion import extract_text_from_pdf, split_text
from core.indexing import FAISSVectorStore
from core.generator import GeminiGenerator

PDF_FILENAME = "sample_paper.pdf"
DB_DIR = "test_db"
INDEX_PATH = os.path.join(DB_DIR, "index.faiss")
DOC_MAP_PATH = os.path.join(DB_DIR, "doc_map.json")

def generate_mock_pdf(filename: str):
    """
    Generates a mock PDF containing explicit facts.
    """
    logger.info(f"Generating mock PDF: {filename}")
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("<b>Security Protocol and Optimization Analysis</b>", styles['Title']))
    story.append(Spacer(1, 20))

    # Paragraph 1
    text1 = (
        "This document describes the security protocols and computational parameters "
        "designed for Project Titan. During the authentication phase, the system verifies "
        "credentials against a hardware security module. The secret protocol passcode is 98765-XYZ. "
        "All operators must use this secret passcode to bypass secondary verification gates."
    )
    story.append(Paragraph(text1, styles['BodyText']))
    story.append(Spacer(1, 15))

    # Paragraph 2
    text2 = (
        "Section 2 focuses on system throughput and tuning. Through extensive simulation, "
        "we found that gradient descent step sizing requires precise scalar calibration. "
        "Specifically, the core optimization function uses an alpha scalar of 0.45. "
        "Deviations from this alpha scalar value will cause divergence in the training loop "
        "and degrade model accuracy."
    )
    story.append(Paragraph(text2, styles['BodyText']))
    story.append(Spacer(1, 15))

    # Build the document
    doc.build(story)
    logger.info(f"Mock PDF '{filename}' built successfully.")

def execute_rag_pipeline(generator: GeminiGenerator, vector_store: FAISSVectorStore):
    """
    Executes the query pipeline against the generator.
    """
    # Query 1: Passcode
    q1 = "What is the secret protocol passcode?"
    logger.info(f"Query 1: {q1}")
    search_q1 = vector_store.search(q1, k=2)
    contexts_q1 = [c for c, dist in search_q1]
    ans1 = generator.generate_answer(q1, contexts_q1)
    logger.info(f"Answer 1: {ans1}")
    assert "98765-XYZ" in ans1, f"Gemini failed to answer Query 1 correctly. Answer was: {ans1}"

    # Query 2: Alpha scalar
    q2 = "What is the alpha scalar of the core optimization function?"
    logger.info(f"Query 2: {q2}")
    search_q2 = vector_store.search(q2, k=2)
    contexts_q2 = [c for c, dist in search_q2]
    ans2 = generator.generate_answer(q2, contexts_q2)
    logger.info(f"Answer 2: {ans2}")
    assert "0.45" in ans2, f"Gemini failed to answer Query 2 correctly. Answer was: {ans2}"

    # Query 3: Out-of-context query
    q3 = "What is the capital of France?"
    logger.info(f"Query 3: {q3}")
    search_q3 = vector_store.search(q3, k=2)
    contexts_q3 = [c for c, dist in search_q3]
    ans3 = generator.generate_answer(q3, contexts_q3)
    logger.info(f"Answer 3: {ans3}")
    expected_failure_msg = "I am sorry, but the answer to your query is not available in the provided document context."
    assert expected_failure_msg in ans3 or "not available" in ans3.lower() or "not find" in ans3.lower() or "sorry" in ans3.lower(), \
        f"Gemini failed to restrict its knowledge. Answer was: {ans3}"

def run_tests():
    # Ensure test db dir exists
    os.makedirs(DB_DIR, exist_ok=True)

    # 1. Generate the Mock PDF
    generate_mock_pdf(PDF_FILENAME)

    # 2. Extract Text
    logger.info("Extracting text from mock PDF...")
    text = extract_text_from_pdf(PDF_FILENAME)
    assert "98765-XYZ" in text, "Error: Text extraction failed to capture the secret protocol passcode."
    assert "0.45" in text, "Error: Text extraction failed to capture the alpha scalar."
    logger.info("Text extraction verified.")

    # 3. Split Text
    logger.info("Splitting text...")
    chunks = split_text(text, chunk_size=600, chunk_overlap=120)
    assert len(chunks) > 0, "Error: Splitting generated zero chunks."
    logger.info(f"Generated {len(chunks)} chunks.")

    # 4. Build Vector Store Index
    logger.info("Building FAISS vector index...")
    vector_store = FAISSVectorStore()
    vector_store.build_index(chunks)
    
    # Save the index
    vector_store.save(INDEX_PATH, DOC_MAP_PATH)
    assert os.path.exists(INDEX_PATH), "Error: FAISS index file was not saved."
    assert os.path.exists(DOC_MAP_PATH), "Error: Chunk map JSON file was not saved."
    logger.info("Vector store building and persistence verified.")

    # 5. Search verification
    logger.info("Verifying search results...")
    results_passcode = vector_store.search("secret protocol passcode", k=1)
    assert len(results_passcode) > 0, "Search returned no results."
    assert "98765-XYZ" in results_passcode[0][0], f"Search failed to retrieve passcode context: {results_passcode[0][0]}"

    results_alpha = vector_store.search("alpha scalar core optimization", k=1)
    assert len(results_alpha) > 0, "Search returned no results."
    assert "0.45" in results_alpha[0][0], f"Search failed to retrieve alpha context: {results_alpha[0][0]}"
    logger.info("Vector search retrieval correctness verified.")

    # 6. Initialize Generator and Run End-to-End RAG Queries (with mocking if no API key is present)
    api_key = os.getenv("GEMINI_API_KEY")
    is_mocked = False
    
    if not api_key:
        logger.info("GEMINI_API_KEY environment variable is not set. Mocking Gemini SDK calls for headless test.")
        os.environ["GEMINI_API_KEY"] = "mock-api-key-for-testing"
        is_mocked = True

    if is_mocked:
        class MockResponse:
            def __init__(self, text: str):
                self.text = text

        mock_client = MagicMock()
        def mock_generate_content(model, contents, config=None):
            prompt = str(contents)
            # Find the Question: block to isolate the query from context
            parts = prompt.split("Question:")
            question_part = parts[1].strip() if len(parts) > 1 else prompt
            
            if "secret protocol passcode" in question_part:
                return MockResponse("Based on the provided context, the secret protocol passcode is 98765-XYZ.")
            elif "alpha scalar" in question_part:
                return MockResponse("The core optimization function uses an alpha scalar of 0.45.")
            else:
                return MockResponse("I am sorry, but the answer to your query is not available in the provided document context.")
                
        mock_client.models.generate_content.side_effect = mock_generate_content
        
        with patch("google.genai.Client", return_value=mock_client):
            logger.info("Initializing Generator (MOCKED)...")
            generator = GeminiGenerator()
            execute_rag_pipeline(generator, vector_store)
    else:
        logger.info("Initializing Generator (REAL API)...")
        generator = GeminiGenerator()
        execute_rag_pipeline(generator, vector_store)

    logger.info("ALL TESTS COMPLETED SUCCESSFULLY! EXIT CODE 0")
    
    # Cleanup files generated during test
    try:
        os.remove(PDF_FILENAME)
        os.remove(INDEX_PATH)
        os.remove(DOC_MAP_PATH)
        os.rmdir(DB_DIR)
        logger.info("Cleaned up test artifacts.")
    except Exception as e:
        logger.warning(f"Failed to clean up test files: {e}")

if __name__ == "__main__":
    try:
        run_tests()
        sys.exit(0)
    except AssertionError as ae:
        logger.error(f"Assertion failed: {ae}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)
