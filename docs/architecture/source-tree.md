# Stellar Connect Project Structure

This document details the complete project structure and organization for the Stellar Connect project. It is derived from the main [architecture document](../architecture.md) and serves as a focused reference for understanding the codebase organization.

## Repository Structure Overview

**Structure:** Monolithic Python application
**Monorepo Tool:** N/A - Single application structure
**Package Organization:** Modular Python packages in `/src` directory with clear separation of concerns

## Complete Project Structure

```plaintext
stellar-connect/
├── src/                        # Source code
│   ├── config.py              # Configuration management
│   ├── ingestion.py           # Document processing pipeline
│   ├── monitor.py             # File system watcher
│   ├── data_models.py         # Pydantic data models
│   ├── agent_tools.py         # CrewAI tool definitions
│   └── stellar_crew.py        # Agent orchestration
├── incoming_transcripts/       # Watch folder for new files
├── archive/                    # Processed files storage
├── docs/                       # Documentation
│   ├── architecture/           # Architecture documentation shards
│   │   ├── coding-standards.md # Development standards and practices
│   │   ├── tech-stack.md      # Technology stack details
│   │   └── source-tree.md     # This document
│   ├── prd.md                 # Product requirements
│   ├── architecture.md        # Main architecture document
│   ├── brief.md               # Project brief
│   └── *.md                   # Other documentation
├── app.py                      # Streamlit web interface
├── requirements.txt            # Python dependencies
├── deploy.sh                   # Deployment script
├── sample_transcript.txt       # Test data
├── .env                        # Environment configuration
├── .gitignore                  # Git ignore file
└── README.md                   # Project documentation
```

## Source Code Organization

### Core Application (`src/`)

The source code is organized into focused modules with clear separation of concerns:

#### `config.py` - Configuration Management
- Global configuration and settings
- Environment variable handling via CONFIG object
- Database connection parameters
- Model configuration settings

```python
# Example configuration structure
class Config:
    NEO4J_URI: str
    NEO4J_USER: str
    NEO4J_PASSWORD: str
    QDRANT_HOST: str
    QDRANT_PORT: int
    EMBEDDING_MODEL: str
    GENERATIVE_MODEL: str
```

#### `ingestion.py` - Document Processing Pipeline
- Main entry point for transcript processing
- Document parsing for various formats
- Semantic text chunking
- Vector and graph storage operations
- File archiving after processing

**Key Functions:**
- `process_new_file(file_path: str) -> None`
- `parse_document(file_path: str) -> str`
- `chunk_text(clean_text: str, file_path: str) -> list`
- `store_in_qdrant(nodes: list, vector_context: StorageContext) -> None`
- `extract_and_store_kg(clean_text: str, file_path: str, graph_context: StorageContext) -> None`

#### `monitor.py` - File System Watcher
- Automated file monitoring using watchdog
- Trigger processing pipeline for new files
- File synchronization and timing management
- Error handling for file operations

**Key Components:**
- FileSystemEventHandler implementation
- watchdog Observer configuration
- Process triggering logic

#### `data_models.py` - Pydantic Data Models
- Structured data models for extraction
- Type validation and runtime safety
- Data transformation and serialization

**Key Models:**
- `SalesRecord` - Sales meeting data structure
- `TestimonialQuote` - Marketing quote extraction
- `Document` - LlamaIndex document representation

#### `agent_tools.py` - CrewAI Tool Definitions
- Custom tool implementations for agents
- Database access abstractions
- Data extraction utilities

**Key Tools:**
- `VectorSearchTool` - Semantic query interface
- `KnowledgeGraphSearchTool` - Relationship query interface
- `PydanticExtractionTool` - Structured data extraction

#### `stellar_crew.py` - Agent Orchestration
- CrewAI agent definitions and coordination
- Task creation for different workflows
- Multi-agent workflow execution

**Key Functions:**
- `create_general_query_tasks(user_query: str) -> list`
- `create_structured_record_tasks(client_name: str) -> list`
- `create_email_recap_tasks(client_name: str) -> list`
- `run_crew(tasks: list) -> str`

## Data Directories

### `incoming_transcripts/` - Input Processing
- Watch folder for new transcript files
- Automatic processing trigger point
- Supports multiple document formats
- Files are monitored via watchdog service

### `archive/` - Processed Files Storage
- Repository for successfully processed files
- Maintains original file structure
- Enables reprocessing if needed
- Organized by processing date

## Documentation Structure

### Main Documentation (`docs/`)

#### Architecture Documentation (`docs/architecture/`)
- **`coding-standards.md`** - Development standards, naming conventions, error handling patterns
- **`tech-stack.md`** - Complete technology stack with rationale and configuration
- **`source-tree.md`** - This document, detailing project structure

#### Core Documentation
- **`architecture.md`** - Complete system architecture document
- **`prd.md`** - Product requirements and feature specifications
- **`brief.md`** - Project overview and objectives
- **Implementation and strategy documents** - Additional technical documentation

## Application Entry Points

### `app.py` - Streamlit Web Interface
- Main user interface application
- Chat interface for natural language queries
- Sidebar controls for specialized tasks
- Session state management
- Real-time interaction with agent system

**Key Components:**
- Chat interface rendering
- Message history management
- Task delegation to CrewAI system
- Error handling and user feedback

## Configuration Files

### `requirements.txt` - Python Dependencies
- Complete list of Python package dependencies
- Version specifications for reproducible builds
- AI/ML framework requirements
- Development and runtime dependencies

### `.env` - Environment Configuration
- Database connection strings
- Model configuration parameters
- Security credentials
- Local service ports and hosts

### `deploy.sh` - Deployment Script
- Automated database deployment
- Docker container orchestration
- Service initialization
- Environment setup automation

## Component Architecture

### Function Organization Pattern

```
src/
├── config.py           # Global configuration and settings
├── ingestion.py       # Document processing pipeline
├── monitor.py         # File system monitoring
├── data_models.py     # Pydantic schemas
├── agent_tools.py     # CrewAI tool implementations
└── stellar_crew.py    # Agent orchestration
```

### Service Template Pattern

```python
# Service pattern example
class IngestionService:
    def __init__(self):
        init_settings()  # Initialize LlamaIndex settings
        self.vector_context, self.graph_context = get_storage_contexts()

    def process(self, file_path: str):
        clean_text = self.parse(file_path)
        nodes = self.chunk(clean_text)
        self.store_vectors(nodes)
        self.store_graph(clean_text)
```

## Data Flow Through Structure

### File Processing Flow
1. **`incoming_transcripts/`** → New files detected
2. **`monitor.py`** → File system events trigger processing
3. **`ingestion.py`** → Document parsing and processing
4. **Vector/Graph Storage** → Dual database storage
5. **`archive/`** → Processed files moved to archive

### Query Processing Flow
1. **`app.py`** → User interface receives query
2. **`stellar_crew.py`** → Agent orchestration and task creation
3. **`agent_tools.py`** → Database queries and data retrieval
4. **AI Processing** → Local LLM inference via Ollama
5. **`app.py`** → Results displayed to user

## Development Workflow Structure

### Local Development Setup
```bash
# Repository root operations
cd stellar-connect
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Service deployment
chmod +x deploy.sh
./deploy.sh

# Application execution
python3 src/monitor.py      # Terminal 1: File monitoring
streamlit run app.py        # Terminal 2: Web interface
```

### Testing Structure (Future)
```
tests/
├── unit/
│   ├── test_ingestion.py
│   ├── test_data_models.py
│   └── test_agent_tools.py
├── integration/
│   ├── test_database_connections.py
│   └── test_crew_workflows.py
└── e2e/
    └── test_full_pipeline.py
```

## File Naming Conventions

- **Python Modules:** snake_case (e.g., `agent_tools.py`)
- **Documentation:** kebab-case (e.g., `coding-standards.md`)
- **Directories:** snake_case (e.g., `incoming_transcripts/`)
- **Configuration:** lowercase with extensions (e.g., `.env`, `requirements.txt`)

## Module Dependencies

### Import Hierarchy
```
app.py
└── src.stellar_crew
    └── src.agent_tools
        ├── src.data_models
        └── src.config

src.monitor
└── src.ingestion
    ├── src.config
    └── src.data_models
```

### External Dependencies
- **LlamaIndex:** Core RAG framework
- **CrewAI:** Agent orchestration
- **Streamlit:** Web interface
- **Qdrant/Neo4j:** Database clients
- **Ollama:** Local LLM runtime

## Scalability Considerations

### Modular Design Benefits
- **Clear Separation:** Each module has distinct responsibilities
- **Easy Testing:** Isolated components for unit testing
- **Flexible Deployment:** Individual service scaling potential
- **Maintainability:** Clear code organization and documentation

### Future Expansion Paths
- **Microservices Split:** Potential service boundary identification
- **API Layer Addition:** REST API for external integrations
- **Multi-User Support:** Session and user management layers
- **Horizontal Scaling:** Agent worker distribution capabilities

---

*This document is part of the Stellar Connect architecture documentation. For the complete system architecture, see [architecture.md](../architecture.md).*