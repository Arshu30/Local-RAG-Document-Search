import os
import logging
from typing import List
from google import genai
from google.genai import types

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class GeminiGenerator:
    """
    RAG Generator using Google's new genai SDK and gemini-2.5-flash.
    """
    def __init__(self, api_key: str = None):
        # Dynamically retrieve API key
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.error("GEMINI_API_KEY environment variable is not set.")
            raise ValueError(
                "GEMINI_API_KEY not found. Please set the GEMINI_API_KEY environment variable."
            )
            
        logger.info("Initializing Google GenAI client...")
        try:
            # Initialize with the official client framework
            self.client = genai.Client(api_key=self.api_key)
            self.model_name = "gemini-2.5-flash"
        except Exception as e:
            logger.exception(f"Failed to initialize Google GenAI Client: {e}")
            raise

    def generate_answer(self, query: str, context_chunks: List[str]) -> str:
        """
        Generates an answer using the provided context chunks and user query.
        
        Args:
            query (str): The user query/question.
            context_chunks (List[str]): Extracted relevant document segments.
            
        Returns:
            str: Generated answer.
        """
        if not context_chunks:
            return "No relevant context was found in the document to answer your question."

        logger.info("Preparing prompt and sending request to Gemini 2.5 Flash...")
        
        # Combine context chunks with a clear delimiter
        context_text = "\n\n---\n\n".join(context_chunks)
        
        # Format the core query prompt
        prompt = (
            f"Context:\n"
            f"{context_text}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

        # Build clean system prompt forcing model to answer ONLY from context
        system_instruction = (
            "You are a highly precise, factual RAG assistant. "
            "Your task is to answer the user's question using ONLY the provided Context snippets. "
            "Follow these strict rules:\n"
            "1. Base your answer strictly on the provided Context. Do not extrapolate, assume, or bring in external knowledge.\n"
            "2. If the Context does not contain the answer, you must state exactly: "
            "'I am sorry, but the answer to your query is not available in the provided document context.'\n"
            "3. Do not formulate speculative answers or provide generic general-knowledge responses."
        )

        try:
            config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0,  # Zero temperature for deterministic factual extraction
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            
            answer = response.text
            if not answer:
                logger.warning("Gemini returned an empty response.")
                return "No answer could be generated."
                
            return answer.strip()
        except Exception as e:
            logger.exception(f"Error during Gemini generation: {e}")
            return f"Error during Gemini generation: {str(e)}"
