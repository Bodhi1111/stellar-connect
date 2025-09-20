#!/bin/bash

# Deploy script for Stellar Connect
echo "🚀 Deploying Stellar Connect Infrastructure..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Deploy Qdrant
echo "📦 Deploying Qdrant Vector Database..."
docker pull qdrant/qdrant:latest
docker volume create qdrant_data 2>/dev/null
docker stop stellar-connect-qdrant 2>/dev/null
docker rm stellar-connect-qdrant 2>/dev/null
docker run -d \
  --name stellar-connect-qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v qdrant_data:/qdrant/storage \
  qdrant/qdrant:latest

echo "✅ Qdrant deployed at http://localhost:6333/dashboard"

# Deploy Neo4j
echo "📦 Deploying Neo4j Graph Database..."
docker pull neo4j:latest
docker volume create neo4j_data 2>/dev/null
docker stop stellar-connect-neo4j 2>/dev/null
docker rm stellar-connect-neo4j 2>/dev/null
docker run -d \
    --name stellar-connect-neo4j \
    -p 7474:7474 \
    -p 7687:7687 \
    -v neo4j_data:/data \
    -e NEO4J_AUTH=neo4j/stellar_secure_2024 \
    neo4j:latest

echo "✅ Neo4j deployed at http://localhost:7474 (user: neo4j, pass: stellar_secure_2024)"

# Download Ollama models
echo "🤖 Downloading AI Models via Ollama..."
ollama pull nomic-embed-text
ollama pull llama3:8b-instruct

echo "✨ Deployment complete! Your databases are running."
echo ""
echo "Next steps:"
echo "1. Create virtual environment: python3 -m venv venv && source venv/bin/activate"
echo "2. Install dependencies: pip install -r requirements.txt"
echo "3. Run monitor in Terminal 1: python3 src/monitor.py"
echo "4. Run UI in Terminal 2: streamlit run app.py"