# Stellar Connect Technology Stack

This document details the complete technology stack for the Stellar Connect project. It is derived from the main [architecture document](../architecture.md) and serves as a focused reference for technology choices and their rationale.

## Technology Stack Overview

Stellar Connect implements a local-first, privacy-preserving AI architecture using dual-database storage (Qdrant for vectors, Neo4j for knowledge graphs) with Python-based ingestion pipelines and CrewAI agent orchestration. The system employs Ollama for local LLM inference, processes sales transcripts through automated file monitoring, and provides a Streamlit-based chat interface for natural language queries.

## Core Technology Stack Table

| Category | Technology | Version | Purpose | Rationale |
|----------|------------|---------|---------|-----------|
| Backend Language | Python | 3.9+ | Core application development | Ecosystem support for AI/ML libraries |
| Backend Framework | Native Python | N/A | Service orchestration | Lightweight for local deployment |
| UI Framework | Streamlit | Latest | Web interface | Rapid prototyping with built-in components |
| Vector Database | Qdrant | Latest | Semantic search storage | High-performance vector similarity search |
| Graph Database | Neo4j | Latest | Knowledge graph storage | Relationship modeling and traversal |
| LLM Framework | LlamaIndex | Latest | RAG orchestration | Comprehensive toolkit for LLM applications |
| Agent Framework | CrewAI | Latest | Multi-agent coordination | Structured approach to agent workflows |
| LLM Runtime | Ollama | Latest | Local model inference | Optimized local LLM deployment |
| Embedding Model | nomic-embed-text | Latest | Text vectorization | Quality embeddings for semantic search |
| Generative Model | Llama3 | 8b-instruct | Text generation and analysis | Balance of capability and performance |
| File Monitoring | Watchdog | Latest | Filesystem events | Cross-platform file system monitoring |
| Data Processing | Unstructured | [all-docs] | Document parsing | Universal document format support |
| Environment Mgmt | python-dotenv | Latest | Configuration management | Secure credential handling |
| Data Validation | Pydantic | Latest | Type checking and validation | Runtime type safety |
| Container Platform | Docker | Latest | Database deployment | Consistent service deployment |
| Python Package Mgr | pip/venv | Latest | Dependency management | Standard Python tooling |

## Platform and Infrastructure

### Platform Choice
- **Platform:** Local Deployment (macOS/Linux)
- **Key Services:** Docker (Qdrant, Neo4j), Ollama (LLM inference), Python runtime
- **Deployment Host:** Single local instance on user's machine (no cloud regions)

### Infrastructure Requirements
- **Minimum RAM:** 16GB (recommended 32GB for optimal performance)
- **Storage:** 50GB free space (models + data)
- **GPU:** Optional (NVIDIA recommended for faster inference)
- **Network:** No external connectivity required

## AI and Machine Learning Stack

### Local LLM Infrastructure
- **Ollama Runtime:** Manages local model deployment and inference
- **Embedding Model:** nomic-embed-text for high-quality semantic embeddings
- **Generative Model:** Llama3:8b-instruct for text generation and analysis
- **Model Storage:** Local filesystem via Ollama management

### Vector Processing
- **Qdrant Vector Database:** High-performance similarity search
- **Embedding Dimension:** 768 (nomic-embed-text native)
- **Distance Metric:** Cosine similarity
- **Collection Management:** Single collection for transcript data

### Knowledge Graph Processing
- **Neo4j Graph Database:** Relationship modeling and traversal
- **Schema Design:** Entity-relationship model for transcript content
- **Query Language:** Cypher for complex relationship queries
- **Indexing Strategy:** Entity and concept name indexing

## Development Stack

### Python Ecosystem
- **Runtime:** Python 3.9+ for modern language features
- **Package Management:** pip with virtual environments
- **Dependency Isolation:** venv for clean development environments
- **Configuration:** python-dotenv for environment variable management

### Data Processing Pipeline
- **Document Parsing:** Unstructured library for universal format support
- **Text Processing:** LlamaIndex semantic chunking
- **Data Validation:** Pydantic models for type safety
- **File Monitoring:** Watchdog for cross-platform file system events

### Agent Framework
- **CrewAI:** Multi-agent coordination and task delegation
- **Agent Types:** Retrieval, data extraction, and content generation agents
- **Task Management:** Sequential processing with inter-agent communication
- **Tool Integration:** Custom tools for database access and data extraction

## User Interface Stack

### Frontend Framework
- **Streamlit:** Python-native web interface framework
- **Component System:** Built-in UI components for rapid development
- **State Management:** Native session state management
- **Real-time Updates:** Automatic re-rendering on state changes

### Interface Features
- **Chat Interface:** Natural language query processing
- **Sidebar Controls:** Task-specific actions and configurations
- **Progress Indicators:** Loading states and processing feedback
- **Error Handling:** User-friendly error messages and recovery

## Database Stack

### Vector Storage (Qdrant)
```json
{
  "collection": "stellar_connect_transcripts",
  "schema": {
    "vector_size": 768,
    "distance": "Cosine",
    "payload_schema": {
      "source_file": "string",
      "chunk_index": "integer",
      "text": "string",
      "metadata": "object"
    }
  }
}
```

### Graph Storage (Neo4j)
```cypher
// Node Types
(Document {source_file: String, processed_date: DateTime})
(Entity {name: String, type: String})
(Concept {name: String, category: String})

// Relationship Types
(Document)-[:CONTAINS]->(Entity)
(Entity)-[:RELATES_TO]->(Entity)
(Document)-[:DISCUSSES]->(Concept)
(Entity)-[:ASSOCIATED_WITH]->(Concept)

// Indexes
CREATE INDEX ON :Document(source_file)
CREATE INDEX ON :Entity(name)
CREATE INDEX ON :Concept(name)
```

## Deployment Stack

### Container Orchestration
- **Docker:** Service deployment for databases
- **Docker Compose:** Multi-service coordination
- **Volume Management:** Persistent data storage
- **Network Configuration:** Inter-service communication

### Service Management
- **Database Services:** Automated startup via Docker
- **Python Services:** Manual or systemd service management
- **Monitoring:** Log-based monitoring and health checks
- **Backup:** File-based backup strategies

## Security Stack

### Local Security
- **Data Privacy:** Complete local processing (no external APIs)
- **Credential Management:** Environment variables via .env files
- **Database Security:** Strong passwords and local-only access
- **File Security:** Proper file permissions and access controls

### Application Security
- **Input Validation:** Pydantic model validation
- **Error Handling:** Secure error messages (no sensitive data exposure)
- **Session Management:** Streamlit native session handling
- **Resource Limits:** Memory and processing time constraints

## Performance Stack

### Optimization Strategies
- **Caching:** In-memory caching for database clients
- **Chunking:** Semantic text splitting for optimal retrieval
- **Indexing:** Database indexes for fast query performance
- **Resource Management:** Proper connection pooling and cleanup

### Monitoring
- **Performance Metrics:** Processing time tracking
- **Memory Monitoring:** Resource usage tracking
- **Error Tracking:** Comprehensive error logging
- **Health Checks:** Service availability monitoring

## Development Tools

### Code Quality
- **Type Checking:** Pydantic models and Python type hints
- **Error Handling:** Comprehensive exception management
- **Testing:** Unit, integration, and end-to-end test frameworks
- **Documentation:** Inline documentation and architectural docs

### Development Workflow
- **Version Control:** Git with conventional commit messages
- **Environment Management:** Virtual environments for isolation
- **Dependency Tracking:** requirements.txt for reproducible builds
- **Deployment Scripts:** Automated setup and deployment

## Technology Rationale

### Local-First Architecture
- **Privacy:** Complete data control for sensitive estate planning information
- **Performance:** Eliminate network latency and external dependencies
- **Reliability:** No external service outages or API rate limits
- **Cost:** No ongoing cloud service costs

### Dual-Database Strategy
- **Vector Database:** Optimal for semantic similarity search
- **Graph Database:** Superior for relationship modeling and traversal
- **Complementary Strengths:** Combined approach provides comprehensive intelligence
- **Scalability:** Independent scaling of different data access patterns

### Python Ecosystem Choice
- **AI/ML Libraries:** Rich ecosystem for machine learning applications
- **Rapid Development:** Fast prototyping and iteration cycles
- **Community Support:** Large community and extensive documentation
- **Integration:** Seamless integration between different AI frameworks

---

*This document is part of the Stellar Connect architecture documentation. For the complete system architecture, see [architecture.md](../architecture.md).*