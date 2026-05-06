"""Configuration for Memory Proxy v4 — Production."""
import os

# Ollama backend
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Server
HOST = "0.0.0.0"
PORT = 5556

# Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# TriVox encoder paths
MODEL_PATH = os.getenv("MODEL_PATH", "/home/aque/trivox2_step120000.pt")
TOK_FR = os.getenv("TOK_FR", "/home/aque/trivox-memory-v3/tokenizers/fr.model")
TOK_EN = os.getenv("TOK_EN", "/home/aque/trivox-memory-v3/tokenizers/en.model")
TOK_CODE = os.getenv("TOK_CODE", "/home/aque/trivox-memory-v3/tokenizers/code.json")
MAX_SEQ_LEN = 256
EMBED_DIM = 768

# Memory settings
MAX_MEMORIES_INJECT = 5       # Max memories injected per request
MIN_SCORE = 0.12              # Min score after time-decay (low = more recall)
CHUNK_MAX_TOKENS = 100        # Max tokens per chunk
CHUNK_OVERLAP = 20            # Token overlap between chunks
MAX_MEMORIES_PER_COLLECTION = 5000  # Auto-cleanup when exceeded

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
