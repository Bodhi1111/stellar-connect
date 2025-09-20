# src/ingestion.py
import os
import qdrant_client
from unstructured.partition.auto import partition
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core import (
    Document, VectorStoreIndex, StorageContext, KnowledgeGraphIndex, Settings
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from src.config import CONFIG, init_settings

# --- Initialize Storage Connections ---

def get_storage_contexts():
    # Vector Store (Qdrant)
    q_client = qdrant_client.QdrantClient(host=CONFIG.QDRANT_HOST, port=CONFIG.QDRANT_PORT)
    vector_store = QdrantVectorStore(client=q_client, collection_name=CONFIG.QDRANT_COLLECTION)
    vector_context = StorageContext.from_defaults(vector_store=vector_store)

    # Graph Store (Neo4j)
    graph_store = Neo4jGraphStore(
        username=CONFIG.NEO4J_USER,
        password=CONFIG.NEO4J_PASSWORD,
        url=CONFIG.NEO4J_URI,
        database="neo4j",
    )
    graph_context = StorageContext.from_defaults(graph_store=graph_store)

    return vector_context, graph_context

# --- Pipeline Functions ---

def parse_document(file_path: str) -> str:
    # (Section 2.2)
    print(f"Parsing document: {file_path}")
    try:
        elements = partition(filename=file_path)
        return "\n\n".join([str(el) for el in elements])
    except Exception as e:
        print(f"Error parsing document {file_path}: {e}")
        return ""

def chunk_text(clean_text: str, file_path: str) -> list:
    # (Section 2.3 - Semantic Chunking)
    print("Starting semantic chunking...")
    splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=Settings.embed_model # Uses global setting
    )
    doc = Document(text=clean_text, metadata={"source_file": os.path.basename(file_path)})
    nodes = splitter.get_nodes_from_documents([doc])
    print(f"Chunked document into {len(nodes)} semantic nodes.")
    return nodes

def store_in_qdrant(nodes: list, vector_context: StorageContext):
    # (Section 2.4.1)
    print("Storing nodes in Qdrant...")
    VectorStoreIndex(
        nodes=nodes,
        storage_context=vector_context,
        embed_model=Settings.embed_model
    )
    print(f"Successfully stored {len(nodes)} nodes in Qdrant.")

def extract_and_store_kg(clean_text: str, file_path: str, graph_context: StorageContext):
    # (Section 2.4.2)
    print("Extracting knowledge graph triplets (this may take a few minutes)...")
    doc = Document(text=clean_text, metadata={"source_file": os.path.basename(file_path)})

    KnowledgeGraphIndex.from_documents(
        documents=[doc],
        storage_context=graph_context,
        max_triplets_per_chunk=15, # As specified in the blueprint
        llm=Settings.llm, # Uses global setting
        include_embeddings=False,
    )
    print("Successfully stored knowledge graph triplets in Neo4j.")

# --- Main Orchestrator ---

def process_new_file(file_path: str):
    print(f"\n--- Starting Ingestion for {os.path.basename(file_path)} ---")

    # Ensure models are initialized
    init_settings()
    try:
        vector_context, graph_context = get_storage_contexts()
    except Exception as e:
        print(f"Failed to initialize storage contexts (Check DB connections): {e}")
        return

    # 1. Parse
    clean_text = parse_document(file_path)
    if not clean_text: return

    # 2. Chunk and 3. Store Vector
    nodes = chunk_text(clean_text, file_path)
    if nodes:
        store_in_qdrant(nodes, vector_context)

    # 4. Store Graph
    extract_and_store_kg(clean_text, file_path, graph_context)

    print(f"--- Finished Ingestion for {os.path.basename(file_path)} ---")