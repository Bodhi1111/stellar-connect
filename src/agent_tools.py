# src/agent_tools.py
import qdrant_client
from crewai.tools import BaseTool
from llama_index.core import (
    VectorStoreIndex, StorageContext, Settings
)
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.core.program import LLMTextCompletionProgram
# from llama_index.core.bridge.pydantic import PydanticProgramMode  # Not available in current version
from src.config import CONFIG, init_settings
from src.data_models import SalesRecord

# Initialize settings (LLM and Embeddings) before initializing tools
init_settings()

# --- Initialize Indexes and Engines ---
# We initialize connections to the existing data stores

# 1. Vector Index (Qdrant)
try:
    q_client = qdrant_client.QdrantClient(host=CONFIG.QDRANT_HOST, port=CONFIG.QDRANT_PORT)
    vector_store = QdrantVectorStore(client=q_client, collection_name=CONFIG.QDRANT_COLLECTION)
    vector_index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=Settings.embed_model)
    # We use a Retriever (fetches context), which is ideal for CrewAI tools
    vector_retriever = vector_index.as_retriever(similarity_top_k=5)
except Exception as e:
    print(f"Failed to initialize Qdrant Index: {e}")
    vector_retriever = None

# 2. Knowledge Graph Engine (Neo4j)
try:
    graph_store = Neo4jGraphStore(
        username=CONFIG.NEO4J_USER,
        password=CONFIG.NEO4J_PASSWORD,
        url=CONFIG.NEO4J_URI,
    )
    graph_context = StorageContext.from_defaults(graph_store=graph_store)
    # This engine handles Text-to-Cypher translation
    kg_query_engine = KnowledgeGraphQueryEngine(
        storage_context=graph_context,
        llm=Settings.llm,
        verbose=True,
    )
except Exception as e:
    print(f"Failed to initialize Neo4j Engine: {e}")
    kg_query_engine = None

# --- Tool Definitions ---

class VectorSearchTool(BaseTool):
    name: str = "Semantic Transcript Search"
    description: str = "Searches the content of transcripts for context, discussions, or specific topics using semantic similarity."

    def _run(self, argument: str) -> str:
        if not vector_retriever: return "Vector Tool not initialized."
        print(f"\n[Tool] Executing Vector Search: {argument}")
        retrieved_nodes = vector_retriever.retrieve(argument)
        context_str = ""
        for node_with_score in retrieved_nodes:
            node = node_with_score.node
            context_str += f"Source: {node.metadata.get('source_file', 'N/A')}\nContent: {node.get_content()}\n---\n"
        return context_str or "No relevant information found in transcripts."

class KnowledgeGraphSearchTool(BaseTool):
    name: str = "Structured Relationship Search"
    description: str = "Searches the knowledge graph for explicit relationships between entities (clients, topics, outcomes)."

    def _run(self, argument: str) -> str:
        if not kg_query_engine: return "KG Tool not initialized."
        print(f"\n[Tool] Executing Knowledge Graph Search: {argument}")
        # The KG engine generates Cypher and returns a response
        response = kg_query_engine.query(argument)
        return str(response)

class PydanticExtractionTool(BaseTool):
    # (Section 4.1 implementation)
    name: str = "Structured Sales Record Extractor"
    description: str = "Extracts structured information from text and populates a SalesRecord JSON object. Input must be the text context."

    def _run(self, argument: str) -> str:
        print(f"\n[Tool] Executing Pydantic Extraction...")
        prompt_template_str = """
        Please extract the following information accurately from the provided sales meeting context.
        If information is missing, use the defaults specified in the schema.

        Context:
        ---------------------
        {context_text}
        ---------------------
        """
        # LLMTextCompletionProgram uses the LLM to fill the Pydantic model
        program = LLMTextCompletionProgram.from_defaults(
            output_cls=SalesRecord,
            llm=Settings.llm,
            prompt_template_str=prompt_template_str,
            # pydantic_program_mode=PydanticProgramMode.DEFAULT,  # Not available in current version
        )
        output = program(context_text=argument)
        return output.json()

# Instantiate Tools
vector_tool = VectorSearchTool()
kg_tool = KnowledgeGraphSearchTool()
extraction_tool = PydanticExtractionTool()