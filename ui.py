import os
import streamlit as st

# Configure page configuration first (required by streamlit)
st.set_page_config(
    page_title="RAG Document Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

from core.ingestion import extract_text_from_pdf, split_text
from core.indexing import FAISSVectorStore
from core.generator import GeminiGenerator

# Title and description
st.title("🔍 RAG Document Explorer")
st.markdown(
    "Upload a PDF, locally compile its vector embeddings with FAISS, "
    "and perform context-restricted queries using Gemini 2.5 Flash."
)

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar configurations
st.sidebar.header("📁 Document & API Configuration")

# API Key verification
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    api_key = st.sidebar.text_input(
        "Enter Gemini API Key:",
        type="password",
        help="Required to generate answers. Get one from Google AI Studio."
    )
else:
    st.sidebar.success("Found API Key in Environment Variables.")

# Initialize generator if API Key is available
if api_key:
    if "generator" not in st.session_state or st.session_state.get("api_key_used") != api_key:
        try:
            st.session_state.generator = GeminiGenerator(api_key=api_key)
            st.session_state.api_key_used = api_key
            st.sidebar.info("Gemini Generator client initialized.")
        except Exception as e:
            st.sidebar.error(f"Failed to initialize Gemini Client: {e}")
else:
    st.sidebar.warning("🔑 Gemini API key is missing. Set GEMINI_API_KEY in your env or enter it above.")

st.sidebar.divider()

# File Uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload Document (PDF):",
    type=["pdf"],
    help="Upload the document you want to search and query."
)

# Process file and build FAISS vector database
if uploaded_file is not None:
    # Check if a new file is uploaded
    if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Save temporary file for the ingestion processor
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        with st.sidebar:
            with st.spinner("Processing PDF and compiling vector index locally..."):
                try:
                    # Ingest and split text
                    raw_text = extract_text_from_pdf(temp_path)
                    chunks = split_text(raw_text)
                    
                    if chunks:
                        # Build vector store
                        vector_store = FAISSVectorStore()
                        vector_store.build_index(chunks)
                        
                        # Store in session state
                        st.session_state.vector_store = vector_store
                        st.session_state.processed_file = uploaded_file.name
                        st.sidebar.success("FAISS index compiled successfully!")
                    else:
                        st.sidebar.error("Could not extract any text chunks from the PDF.")
                except Exception as e:
                    st.sidebar.error(f"Failed to index document: {e}")

# If we have an active index, show status
if "vector_store" in st.session_state:
    st.sidebar.info(
        f"Active Document: `{st.session_state.processed_file}`\n"
        f"Total chunks indexed: {len(st.session_state.vector_store.chunks)}"
    )

# Display Conversational Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Show Retrieved Context Chunks"):
                for idx, source in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk [{idx}]** (Distance L2: {source['distance']:.4f}):")
                    st.write(source["text"])
                    st.divider()

# Input loop for new prompts
if prompt := st.chat_input("Ask a question about the document..."):
    # Check dependencies before sending prompt
    if "vector_store" not in st.session_state:
        st.error("Please upload a PDF document and wait for FAISS indexing to complete first.")
    elif not api_key:
        st.error("Please configure the Gemini API Key first.")
    else:
        # Display user input immediately
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Step 1: Perform vector search retrieval
        with st.spinner("Retrieving relevant contexts..."):
            try:
                search_results = st.session_state.vector_store.search(prompt, k=4)
                context_chunks = [chunk for chunk, dist in search_results]
                sources = [{"text": chunk, "distance": dist} for chunk, dist in search_results]
            except Exception as e:
                st.error(f"Failed during vector search retrieval: {e}")
                sources = []
                context_chunks = []

        # Step 2: Query the RAG Generator
        if context_chunks:
            with st.spinner("Querying Gemini 2.5 Flash..."):
                try:
                    answer = st.session_state.generator.generate_answer(prompt, context_chunks)
                except Exception as e:
                    answer = f"Error during answer generation: {e}"
        else:
            answer = "I am sorry, but the answer to your query is not available in the provided document context."

        # Display response
        with st.chat_message("assistant"):
            st.write(answer)
            if sources:
                with st.expander("Show Retrieved Context Chunks"):
                    for idx, source in enumerate(sources, 1):
                        st.markdown(f"**Chunk [{idx}]** (Distance L2: {source['distance']:.4f}):")
                        st.write(source["text"])
                        st.divider()

        # Save assistant message to session state
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })
