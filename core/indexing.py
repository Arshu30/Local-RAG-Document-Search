import os
import json
import logging
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """
    Local Vector Store utilizing FAISS and sentence-transformers for local RAG indexing and search.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        logger.info(f"Loading local embedding model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
            self.index = None
            self.chunks: List[str] = []
        except Exception as e:
            logger.exception(f"Failed to load sentence-transformers model '{model_name}': {e}")
            raise

    def build_index(self, chunks: List[str]) -> None:
        """
        Generates local embeddings for the chunks and builds the FAISS index.
        
        Args:
            chunks (List[str]): A list of text chunks to index.
        """
        if not chunks:
            logger.warning("Empty chunks list provided. Index will not be built.")
            return

        logger.info(f"Embedding {len(chunks)} chunks and building FAISS index...")
        try:
            # Generate embeddings (list of vectors)
            embeddings = self.model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
            
            # Convert embeddings to float32 NumPy arrays
            embeddings_np = np.array(embeddings).astype("float32")
            
            # Verify shape
            num_vectors, dimension = embeddings_np.shape
            logger.info(f"Generated embeddings. Shape: {embeddings_np.shape}")
            
            # Create a flat L2 index (IndexFlatL2)
            self.index = faiss.IndexFlatL2(dimension)
            
            # Add embeddings to the index
            self.index.add(embeddings_np)
            self.chunks = list(chunks)
            logger.info("Successfully added all embeddings to the FAISS index.")
        except Exception as e:
            logger.exception(f"Error building FAISS index: {e}")
            raise

    def search(self, query: str, k: int = 4) -> List[Tuple[str, float]]:
        """
        Searches the FAISS index for the most similar chunks to the query.
        
        Args:
            query (str): The search query.
            k (int): The number of nearest neighbors to retrieve.
            
        Returns:
            List[Tuple[str, float]]: A list of tuples containing (chunk_text, distance).
        """
        if self.index is None or not self.chunks:
            logger.error("FAISS index has not been built or loaded yet.")
            raise ValueError("Vector store index is not initialized.")

        try:
            # Encode query
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            query_embedding_np = np.array(query_embedding).astype("float32")
            
            # Search FAISS index
            distances, indices = self.index.search(query_embedding_np, k)
            
            # Retrieve corresponding chunks and distances
            results = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue  # FAISS returns -1 if not enough neighbors are found
                if 0 <= idx < len(self.chunks):
                    results.append((self.chunks[idx], float(dist)))
                else:
                    logger.warning(f"FAISS index {idx} is out of bounds for chunks list of size {len(self.chunks)}")
                    
            return results
        except Exception as e:
            logger.exception(f"Error performing FAISS search for query '{query}': {e}")
            raise

    def save(self, index_path: str, doc_map_path: str) -> None:
        """
        Persists the FAISS index and the corresponding chunks mapping to disk.
        
        Args:
            index_path (str): File path to save the FAISS index.
            doc_map_path (str): File path to save the chunks JSON mapping.
        """
        if self.index is None:
            logger.error("No index built to save.")
            raise ValueError("Index is empty.")

        try:
            logger.info(f"Saving FAISS index to {index_path}")
            faiss.write_index(self.index, index_path)
            
            logger.info(f"Saving chunk mapping to {doc_map_path}")
            with open(doc_map_path, "w", encoding="utf-8") as f:
                json.dump(self.chunks, f, ensure_ascii=False, indent=2)
                
            logger.info("Vector store successfully saved to disk.")
        except Exception as e:
            logger.exception(f"Failed to save vector store: {e}")
            raise

    def load(self, index_path: str, doc_map_path: str) -> None:
        """
        Loads the FAISS index and chunks mapping from disk.
        
        Args:
            index_path (str): File path to the persisted FAISS index.
            doc_map_path (str): File path to the persisted chunks JSON mapping.
        """
        if not os.path.exists(index_path) or not os.path.exists(doc_map_path):
            logger.error(f"Cannot load index. Files not found. Index path: {index_path}, Doc map path: {doc_map_path}")
            raise FileNotFoundError("Persisted index files not found.")

        try:
            logger.info(f"Loading FAISS index from {index_path}")
            self.index = faiss.read_index(index_path)
            
            logger.info(f"Loading chunk mapping from {doc_map_path}")
            with open(doc_map_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
                
            logger.info(f"Vector store successfully loaded with {len(self.chunks)} chunks.")
        except Exception as e:
            logger.exception(f"Failed to load vector store: {e}")
            raise
