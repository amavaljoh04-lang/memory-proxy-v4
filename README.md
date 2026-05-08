# Memory Proxy v4.2 — Enterprise Edition

Proxy mémoire intelligent entre Open-WebUI et LLMs (Ollama, OpenAI, Grok, Anthropic, Mistral...). Intercepte les conversations, stocke les souvenirs dans Qdrant via TriVox embeddings, indexe des codebases complètes, et injecte le contexte pertinent avant chaque réponse.

## Architecture

```
Open-WebUI  -->  Memory Proxy v4.2 (port 5556)  -->  Ollama (local)
                       |                          -->  OpenAI API
                       |                          -->  Grok (xAI)
                       |                          -->  Anthropic
                       |                          -->  Mistral
                       |                          -->  Custom LLM API
                       |
                       +-- TriVox Encoder (CPU)
                       +-- Qdrant (port 6333)
                       +-- Code Indexer (repos → chunks → embeddings)
                       +-- Project Detector (auto)
                       +-- Entity Extractor (14 types)
```

## Nouvelles fonctionnalités v4.2

### Indexation de codebase
- **`/index /chemin/repo`** — indexe un repo local en quelques secondes
- **`/index https://github.com/...`** — clone et indexe depuis GitHub/GitLab
- **`/switch repo_name`** — change de codebase active
- **`/architecture`** — résumé complet (classes, fonctions, dépendances, entry points)
- **`/search query`** — cherche dans le code indexé
- Chunking par fonction/classe avec fichier + numéros de lignes
- Injection automatique de code pertinent dans chaque réponse
- Collection Qdrant dédiée par repo : `code_{repo_name}`

### Multi-backend LLM
- **`/backend`** — voir les backends disponibles
- **`/backend openai`** — utiliser GPT-4o via API OpenAI
- **`/backend grok`** — utiliser Grok via xAI
- **`/backend ollama`** — revenir au LLM local
- Configuration via variables d'environnement (OPENAI_API_KEY, GROK_API_KEY, etc.)
- Ajout de backends custom via API REST
- Conversion automatique Ollama ↔ OpenAI format

### Décisions architecturales
- **`/decision "texte"`** — stocker une décision importante dans la mémoire du projet
- **`/decision`** — lister toutes les décisions du projet actif
- Les décisions sont indexées et retrieval-able par le système de mémoire

---

## Fonctionnalités héritées de v4.1

- **Mémoire partagée** — Compatible avec TOUS les modèles Ollama
- **Isolation par projet** — Détection automatique + collections Qdrant séparées
- **14 types d'entités** — couleurs hex, money, organisations, personnes, IPs, versions, etc.
- **Double API** — Ollama (`/api/chat`) + OpenAI (`/v1/chat/completions`)
- **Streaming** — SSE pour OpenAI, ndjson pour Ollama
- **Chunking intelligent** — Code: 60 lignes max par fonction/classe avec overlap 5 lignes
- **Hybrid search** — Vector similarity + keyword boost + synonymes FR/EN
- **Sensitivity filtering** — API keys, tokens → `[REDACTED]`
- **Memory compression** — Déduplication automatique des vieilles mémoires
- **Visualisation APIs** — Timeline, graph entités-projets, scores de confiance
- **Time-decay** — Souvenirs récents scorent plus haut
- **Verrouillage projet** — `/projet X` verrouille, la détection auto ne peut pas écraser

---

## Déploiement rapide

### Prérequis

| Service | Installation |
|---------|-------------|
| Python 3.10+ | Pré-installé |
| Qdrant | `docker run -d --name qdrant -p 6333:6333 -v /home/aque/qdrant_storage:/qdrant/storage --restart unless-stopped qdrant/qdrant:latest` |
| Ollama | Déjà en service sur port 11434 |
| TriVox checkpoint | Fichier `.pt` (~1GB) |
| Tokenizers | `fr.model`, `en.model`, `code.json` |

### Installation

```bash
cd /home/aque
git clone https://github.com/amavaljoh04-lang/memory-proxy-v4
cd memory-proxy-v4
pip install -r requirements.txt
```

### Configuration

Éditer `start.sh` ou exporter les variables :

```bash
# Obligatoire
export MODEL_PATH=/home/aque/trivox2_step120000.pt
export TOK_FR=/home/aque/trivox-memory-v3/tokenizers/fr.model
export TOK_EN=/home/aque/trivox-memory-v3/tokenizers/en.model
export TOK_CODE=/home/aque/trivox-memory-v3/tokenizers/code.json
export OLLAMA_URL=http://localhost:11434

# Optionnel — Multi-backend
export OPENAI_API_KEY=sk-...           # Pour utiliser GPT
export GROK_API_KEY=xai-...           # Pour utiliser Grok
export ANTHROPIC_API_KEY=sk-ant-...   # Pour utiliser Claude
export MISTRAL_API_KEY=...            # Pour utiliser Mistral
export CUSTOM_LLM_URL=http://...      # Backend custom
export CUSTOM_LLM_KEY=...
```

### Lancer

```bash
bash start.sh
# ou en arrière-plan :
nohup bash start.sh > /tmp/memory-proxy-v4.log 2>&1 &
# ou via systemd :
sudo systemctl enable --now trivox-proxy
```

### Configurer Open-WebUI

Dans Open-WebUI > Settings > Connections, URL Ollama :
```
http://192.168.0.203:5556
```

---

## Commandes /slash

### Mémoire
| Commande | Action |
|----------|--------|
| `/projet <nom>` | Switcher + verrouiller un projet |
| `/oublie <nom>` | Effacer la mémoire d'un projet |
| `/oublie tout` | Reset complet + déverrouiller |
| `/projets` | Lister tous les projets |
| `/mémoire` | Stats actuelles |
| `/contexte` | Preview du contexte injecté |

### Codebase
| Commande | Action |
|----------|--------|
| `/index /chemin/repo` | Indexer un repo local |
| `/index https://github.com/...` | Cloner et indexer |
| `/switch <nom>` | Changer de codebase active |
| `/architecture` | Résumé architecture du repo |
| `/search <query>` | Chercher dans le code |

### Décisions
| Commande | Action |
|----------|--------|
| `/decision <texte>` | Stocker une décision |
| `/decision` | Lister les décisions |

### Backend LLM
| Commande | Action |
|----------|--------|
| `/backend` | Voir les backends disponibles |
| `/backend openai` | Utiliser OpenAI |
| `/backend ollama` | Revenir à Ollama local |
| `/backend grok` | Utiliser Grok |
| `/aide` | Aide complète |

---

## API REST

### Mémoire
- `GET /health` — Status + stats
- `POST /memory/search` — `{"query": "...", "project": "..."}` 
- `POST /memory/clear` — `{"project": "..."}` ou `{}`
- `GET /memory/timeline` — Chronologie des souvenirs
- `GET /memory/graph` — Graph entités-projets
- `GET /memory/confidence?query=...` — Recherche avec scores

### Code
- `POST /code/index` — `{"path": "/chemin/repo"}`
- `POST /code/search` — `{"query": "...", "codebase": "..."}`
- `GET /code/repos` — Liste des repos indexés
- `GET /code/architecture` — Architecture du repo actif

### Backends
- `GET /backends` — Liste des backends
- `POST /backends/switch` — `{"name": "openai"}`
- `POST /backends/add` — `{"name": "...", "base_url": "...", "api_key": "..."}`

---

## Fichiers

| Fichier | Rôle |
|---------|------|
| `server.py` | Serveur FastAPI principal (proxy + slash commands + API) |
| `memory.py` | Store Qdrant avec hybrid search + time-decay |
| `encoder.py` | TriVox encoder wrapper (CPU) |
| `chunker.py` | Chunking texte + code |
| `indexer.py` | **Nouveau** — Indexation de codebases (scan, chunk, embed, Qdrant) |
| `backends.py` | **Nouveau** — Multi-backend LLM (Ollama, OpenAI, Grok, etc.) |
| `project_detector.py` | Détection automatique de projet |
| `entity_extractor.py` | Extraction de 14 types d'entités |
| `config.py` | Configuration centralisée |
| `start.sh` | Script de lancement |
| `deploy.sh` | Script de déploiement |

---

## Scénario démo enterprise (5 min)

```bash
# 1. Lancer le proxy (déjà fait si systemd)
bash start.sh

# 2. Dans Open-WebUI, taper :
/index /chemin/vers/codebase-client

# 3. Poser des questions en français :
"Quelle fonction gère les erreurs HTTP ?"
→ retrouve http_exception_handler avec fichier + lignes

"Pourquoi le projet utilise Starlette ?"
→ cherche dans les commentaires/docs

"Génère une nouvelle route POST /users"
→ génère du code cohérent avec l'architecture

# 4. Fermer et rouvrir le chat :
"Tu te souviens de ma question précédente ?"
→ OUI — mémoire persistante cross-session

# 5. Stocker une décision :
/decision "On utilise PostgreSQL pour la persistance"

# 6. Changer de backend si besoin :
/backend openai
```
