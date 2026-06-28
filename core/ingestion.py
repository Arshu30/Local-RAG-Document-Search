import os
import logging
from typing import List
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts raw text content from a PDF file page by page.
    
    Args:
        pdf_path (str): The local path to the PDF file.
        
    Returns:
        str: The concatenated raw text content of the PDF.
        
    Raises:
        FileNotFoundError: If the PDF file does not exist.
        ValueError: If text extraction yields no content or fails.
    """
    if not os.path.exists(pdf_path):
        logger.error(f"PDF file not found at: {pdf_path}")
        raise FileNotFoundError(f"File not found: {pdf_path}")
        
    logger.info(f"Extracting text from PDF: {pdf_path}")
    try:
        reader = PdfReader(pdf_path)
        extracted_pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                extracted_pages.append(text)
            else:
                logger.warning(f"No text extracted from page {i + 1} of {pdf_path}")
                
        raw_text = "\n".join(extracted_pages).strip()
        if not raw_text:
            raise ValueError(f"Extracted text is empty for PDF: {pdf_path}")
            
        logger.info(f"Extracted {len(raw_text)} characters from PDF.")
        return raw_text
    except Exception as e:
        logger.exception(f"Error extracting text from PDF '{pdf_path}': {e}")
        raise

def split_text(text: str, chunk_size: int = 600, chunk_overlap: int = 120) -> List[str]:
    """
    Splits a large string into chunks using RecursiveCharacterTextSplitter.
    
    Args:
        text (str): The raw text to split.
        chunk_size (int): Max size of each chunk in characters.
        chunk_overlap (int): Character overlap between consecutive chunks.
        
    Returns:
        List[str]: A list of text chunks.
    """
    if not text:
        return []
        
    logger.info(f"Splitting text with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    chunks = splitter.split_text(text)
    logger.info(f"Generated {len(chunks)} text chunks.")
    return chunks
