#!/bin/bash
# Deploy Memory Proxy v4 on the 4060 machine
set -e

echo "=== Memory Proxy v4 — Deployment ==="

# 1. Start Qdrant if not running
if ! curl -s http://localhost:6333/collections > /dev/null 2>&1; then
    echo "Starting Qdrant via Docker..."
    docker run -d --name qdrant \
        -p 6333:6333 -p 6334:6334 \
        -v /home/aque/qdrant_storage:/qdrant/storage \
        --restart unless-stopped \
        qdrant/qdrant:latest 2>/dev/null || \
    docker start qdrant 2>/dev/null || true
    sleep 3
    echo "Qdrant started"
else
    echo "Qdrant already running"
fi

# 2. Check Ollama
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "WARNING: Ollama not responding on localhost:11434"
    echo "Starting Ollama..."
    systemctl start ollama 2>/dev/null || true
    sleep 2
fi

# 3. Install Python deps
echo "Installing dependencies..."
pip install -q fastapi uvicorn httpx qdrant-client torch tokenizers sentencepiece pydantic 2>/dev/null || true

# 4. Kill old proxy
fuser -k 5556/tcp 2>/dev/null || true
sleep 1

# 5. Find best checkpoint (try step120k first, then pure.pt)
MODEL_PATH="/home/aque/trivox-memory-v3/models/trivox_pure.pt"
if [ -f "/home/aque/trivox2_step120000.pt" ]; then
    MODEL_PATH="/home/aque/trivox2_step120000.pt"
elif [ -f "/home/aque/trivox2_step084000.pt" ]; then
    MODEL_PATH="/home/aque/trivox2_step084000.pt"
fi
echo "Model: $MODEL_PATH"

export MODEL_PATH
export TOK_FR="/home/aque/trivox-memory-v3/tokenizers/fr.model"
export TOK_EN="/home/aque/trivox-memory-v3/tokenizers/en.model"
export TOK_CODE="/home/aque/trivox-memory-v3/tokenizers/code.json"
export OLLAMA_URL="http://localhost:11434"
export QDRANT_HOST="localhost"
export QDRANT_PORT="6333"

# 6. Start proxy
echo "Starting Memory Proxy v4 on port 5556..."
cd /home/aque/memory-proxy-v4
nohup python3 server.py > /tmp/memory-proxy-v4.log 2>&1 &
PROXY_PID=$!
echo "PID: $PROXY_PID"

# 7. Wait and verify
sleep 5
if kill -0 $PROXY_PID 2>/dev/null; then
    # Check health
    HEALTH=$(curl -s http://localhost:5556/health 2>/dev/null || echo "no response")
    echo "=== SUCCESS ==="
    echo "Health: $HEALTH"
    echo "Log: /tmp/memory-proxy-v4.log"
    echo ""
    echo "In Open-WebUI, set Ollama URL to:"
    echo "  http://192.168.0.203:5556"
else
    echo "=== FAILED ==="
    tail -30 /tmp/memory-proxy-v4.log
    exit 1
fi
