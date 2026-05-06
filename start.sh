#!/bin/bash
export MODEL_PATH=/home/aque/trivox2_step120000.pt
export TOK_FR=/home/aque/trivox-memory-v3/tokenizers/fr.model
export TOK_EN=/home/aque/trivox-memory-v3/tokenizers/en.model
export TOK_CODE=/home/aque/trivox-memory-v3/tokenizers/code.json
export OLLAMA_URL=http://localhost:11434
cd /home/aque/memory-proxy-v4
exec python3 server.py
