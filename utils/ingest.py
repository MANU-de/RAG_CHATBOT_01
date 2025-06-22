# utils/ingest.py
import os
from dotenv import load_dotenv
import logging

from langchain_community.document_loaders import DirectoryLoader, TextLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS 

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables (for Gemini API key)
load_dotenv() 

# --- Configuration ---
DOCUMENTS_PATH = "documents"  # Path to your documents
VECTORSTORE_PATH = "vectorstore_db" # Path to save/load the FAISS index

def create_vectorstore():
    """Loads documents, splits them, creates embeddings, and saves to FAISS."""
    logger.info(f"Loading documents from {DOCUMENTS_PATH}...")
    loader = DirectoryLoader(DOCUMENTS_PATH, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
    documents = loader.load()
    if not documents:
        logger.error("No documents found. Exiting.")
        return None
    logger.info(f"Loaded {len(documents)} documents.")

    logger.info("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(texts)} text chunks.")

    logger.info("Initializing Gemini embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # You can specify the model if needed

    logger.info("Creating FAISS vector store from documents...")
    try:
        vectorstore = FAISS.from_documents(texts, embeddings)
        logger.info("Vector store created successfully.")
    except Exception as e:
        logger.error(f"Error creating FAISS vector store: {e}", exc_info=True)
        return None
        
    logger.info(f"Saving vector store to {VECTORSTORE_PATH}...")
    vectorstore.save_local(VECTORSTORE_PATH)
    logger.info("Vector store saved.")
    return vectorstore

def load_vectorstore():
    """Loads an existing FAISS vector store."""
    if os.path.exists(VECTORSTORE_PATH):
        logger.info(f"Loading existing vector store from {VECTORSTORE_PATH}...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)
        logger.info("Vector store loaded successfully.")
        return vectorstore
    else:
        logger.warning(f"No vector store found at {VECTORSTORE_PATH}. Please run ingestion first.")
        return None

if __name__ == "__main__":
    # This part allows you to run `python utils/ingest.py` to create the vector store
    if not os.environ.get("GOOGLE_API_KEY"):
        logger.error("GOOGLE_API_KEY not found in environment variables. Please set it in .env file.")
    else:
        if os.path.exists(VECTORSTORE_PATH):
            print(f"Vector store already exists at {VECTORSTORE_PATH}. Do you want to recreate it? (yes/no)")
            choice = input().lower()
            if choice == 'yes':
                create_vectorstore()
            else:
                print("Skipping recreation.")
        else:
            create_vectorstore()