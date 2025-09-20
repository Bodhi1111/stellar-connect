# src/config.py
import os
from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# Load environment variables from .env file
load_dotenv()

class Config:
    # Database Credentials
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    QDRANT_HOST = os.getenv("QDRANT_HOST")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

    # Models
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    GENERATIVE_MODEL = os.getenv("GENERATIVE_MODEL")

    # Constants
    QDRANT_COLLECTION = "stellar_connect_transcripts"

CONFIG = Config()

# Initialize LlamaIndex Global Settings
def init_settings():
    # Set the global embedding model and LLM for LlamaIndex
    Settings.embed_model = OllamaEmbedding(model_name=CONFIG.EMBEDDING_MODEL)
    # Increased timeout (120s) is crucial for complex local tasks like KG extraction
    Settings.llm = Ollama(model=CONFIG.GENERATIVE_MODEL, request_timeout=120.0)