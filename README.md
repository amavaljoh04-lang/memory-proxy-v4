# Memory Proxy v4.1

Proxy mémoire intelligent entre Open-WebUI et Ollama. Intercepte les conversations, stocke les souvenirs dans Qdrant via TriVox2 embeddings, et injecte le contexte pertinent avant chaque réponse.

## Architecture

```
Open-WebUI  -->  Memory Proxy v4 (port 5556)  -->  Ollama (port 11434)
                       |
                       +-- TriVox2 Encoder (CPU)
                       +-- Qdrant (port 6333)
                       +-- Project Detector (auto)
```

## Fonctionnalites

- **Memoire partagee** — Compatible avec TOUS les modeles Ollama
- **Isolation par projet** — Detection automatique (NEXUS, AURORA, HELIOS, etc.) + collections Qdrant separees
- **Commandes /slash** — `/projet`, `/oublie`, `/projets`, `/memoire`, `/contexte`, `/aide`
- **14 types d'entites** — couleurs hex, money, organisations, personnes, IPs, versions, etc.
- **Double API** — Ollama (`/api/chat`) + OpenAI (`/v1/chat/completions`)
- **Streaming** — SSE pour OpenAI, ndjson pour Ollama
- **Chunking intelligent** — Code: 50 lignes max par fonction/classe avec overlap 5 lignes. Texte: 100 tokens max, overlap 20 tokens
- **Hybrid search** — Vector similarity (TriVox2) + keyword boost (1.0→1.5x) + synonymes FR/EN
- **Sensitivity filtering** — API keys, tokens, passwords automatiquement rediges en `[REDACTED]` avant stockage
- **Memory compression** — Deduplication automatique des vieilles memoires (>7 jours)
- **Visualisation APIs** — Timeline, graph entites-projets, types d'entites, scores de confiance
- **Time-decay** — Souvenirs recents ont un score plus eleve (1.0 -> 0.70 sur 1 mois)
- **Limite memoire** — 5000 chunks max/collection, auto-cleanup
- **Robuste** — Try/except sur tout, mode pass-through si encoder echoue

---

## Deploiement rapide (sur le serveur 4060 = 192.168.0.203)

### Prerequis

| Service | Comment l'installer |
|---------|---------------------|
| Python 3.10+ | Pre-installe sur Pop!_OS |
| Qdrant | `docker run -d --name qdrant -p 6333:6333 -v /home/aque/qdrant_storage:/qdrant/storage --restart unless-stopped qdrant/qdrant:latest` |
| Ollama | Deja en service sur port 11434 |
| TriVox2 checkpoint | Fichier `.pt` (non inclus dans le repo, ~1GB) |
| Tokenizers | 3 fichiers: `fr.model`, `en.model`, `code.json` (non inclus) |

### Etape 1 : Cloner le repo

```bash
cd /home/aque
git clone <URL_DU_REPO> memory-proxy-v4
cd memory-proxy-v4
```

### Etape 2 : Installer les dependances

```bash
pip install -r requirements.txt
```

### Etape 3 : Configurer les chemins

Editer `start.sh` pour pointer vers le bon checkpoint et les tokenizers :

```bash
export MODEL_PATH=/home/aque/trivox2_step120000.pt
export TOK_FR=/home/aque/trivox-memory-v3/tokenizers/fr.model
export TOK_EN=/home/aque/trivox-memory-v3/tokenizers/en.model
export TOK_CODE=/home/aque/trivox-memory-v3/tokenizers/code.json
export OLLAMA_URL=http://localhost:11434
```

### Etape 4 : Lancer

```bash
# Methode 1 : Script direct
bash start.sh

# Methode 2 : En arriere-plan
nohup bash start.sh > /tmp/memory-proxy-v4.log 2>&1 &

# Methode 3 : Service systemd (auto-restart)
sudo cp trivox-proxy.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now trivox-proxy
```

### Etape 5 : Verifier

```bash
curl http://localhost:5556/health
```

### Etape 6 : Configurer Open-WebUI

Dans Open-WebUI > Settings > Connections, remplacer l'URL Ollama par :
```
http://192.168.0.203:5556
```

---

## Mise a jour du proxy

Depuis n'importe quelle machine du reseau local :

```bash
# 1. Pull les changements
cd /home/aque/memory-proxy-v4
git pull

# 2. Redemarrer
sudo systemctl restart trivox-proxy
# ou si pas de systemd :
pkill -f "server.py"; sleep 2; nohup bash start.sh > /tmp/memory-proxy-v4.log 2>&1 &
```

---

## Commandes /slash (dans le chat Open-WebUI)

| Commande | Action |
|----------|--------|
| `/projet <nom>` | Switcher sur un projet (ex: `/projet helios`) |
| `/oublie <nom>` | Effacer la memoire d'un projet (ex: `/oublie helios`) |
| `/oublie tout` | Reset complet de toute la memoire |
| `/projets` | Lister tous les projets en memoire avec nombre de souvenirs |
| `/memoire` | Voir les stats actuelles (total + par collection) |
| `/contexte` | Preview du contexte qui sera injecte pour le projet actif |
| `/aide` | Afficher l'aide des commandes |

Variantes supportees : `/project`, `/forget`, `/clear`, `/memory`, `/stats`, `/context`, `/help`

---

## Detection automatique de projet

Le proxy detecte automatiquement le projet actif dans cet ordre de priorite :

1. **Commande explicite** — "on parle du projet X", "projet: X", "switch to X"
2. **Titre/en-tete** — "DOSSIER CONFIDENTIEL -- PROJET HELIOS" en premiere ligne
3. **Pattern dans le texte** — "PROJET TITAN" en majuscules
4. **Noms recurrents** — "APOLLON-20B" apparait 3+ fois -> projet helios
5. **Mots-cles connus** — nexus, aurora, trivox, helios, orion, titan
6. **Fallback hash** — Si rien detecte et contenu > 100 chars -> cree `anon_{hash}`

Mapping modeles -> projets :
- APOLLON-20B -> helios
- PERSEUS-35B -> orion
- HELIX-7B -> aurora
- SOLARIS-12B -> aurora

---

## Endpoints API

### Chat (avec memoire)

```bash
# Ollama format
curl -X POST http://localhost:5556/api/chat \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5:7b-instruct-q4_K_M", "messages": [{"role": "user", "content": "Bonjour!"}], "stream": false}'

# OpenAI format
curl -X POST http://localhost:5556/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen2.5:7b-instruct-q4_K_M", "messages": [{"role": "user", "content": "Bonjour!"}]}'
```

### Memoire

```bash
# Stats
curl http://localhost:5556/memory/stats

# Recherche semantique
curl -X POST http://localhost:5556/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "budget du projet", "top_k": 5}'

# Lister les projets
curl http://localhost:5556/memory/projects

# Clear un projet
curl -X POST http://localhost:5556/memory/clear \
  -H "Content-Type: application/json" \
  -d '{"project": "helios"}'

# Clear tout
curl -X POST http://localhost:5556/memory/clear
```

### Hybrid Search & Visualisation (nouveau v4.1)

```bash
# Recherche avec scores de confiance (hybrid: vector + keyword boost)
curl "http://localhost:5556/memory/confidence?query=Ou+habite+Johnny&top_k=5"
# Reponse: {"results": [{"text": "...", "score": 0.95, "confidence": "high", "project": "trivox", ...}]}

# Timeline des souvenirs (chronologique)
curl "http://localhost:5556/memory/timeline?limit=20"
curl "http://localhost:5556/memory/timeline?project=helios&limit=10"

# Graph entites-projets (noeuds + liens)
curl http://localhost:5556/memory/graph
# Reponse: {"nodes": [{"id": "helios", "type": "project"}, ...], "edges": [...]}

# Types d'entites extraites (avec exemples)
curl http://localhost:5556/memory/entity_types
# Reponse: {"entity_types": {"constant": {"count": 9, "examples": ["MAX_RETRIES = 3", ...]}, ...}}

# Compression memoire (supprime les doublons >7 jours)
curl -X POST http://localhost:5556/memory/compress
curl -X POST "http://localhost:5556/memory/compress?max_age_hours=72"  # custom age
```

### Sante

```bash
curl http://localhost:5556/health
curl http://localhost:5556/api/tags   # Liste des modeles Ollama
```

---

## Configuration (config.py / variables d'environnement)

| Variable | Defaut | Description |
|----------|--------|-------------|
| `MODEL_PATH` | `/home/aque/trivox2_step120000.pt` | Checkpoint TriVox2 |
| `TOK_FR` | `.../tokenizers/fr.model` | Tokenizer francais (SentencePiece) |
| `TOK_EN` | `.../tokenizers/en.model` | Tokenizer anglais (SentencePiece) |
| `TOK_CODE` | `.../tokenizers/code.json` | Tokenizer code (HuggingFace) |
| `OLLAMA_URL` | `http://localhost:11434` | URL backend Ollama |
| `QDRANT_HOST` | `localhost` | Host Qdrant |
| `QDRANT_PORT` | `6333` | Port Qdrant |
| `MAX_MEMORIES_INJECT` | `8` | Nb de souvenirs injectes par requete |
| `MIN_SCORE` | `0.10` | Score minimum apres time-decay |
| `CHUNK_MAX_TOKENS` | `100` | Taille max d'un chunk (texte) |
| `MAX_CODE_LINES` | `50` | Taille max d'un chunk code (lignes) |
| `CODE_OVERLAP_LINES` | `5` | Overlap entre chunks code (lignes) |
| `MAX_MEMORIES_PER_COLLECTION` | `5000` | Limite par collection |

---

## Structure des fichiers

```
memory-proxy-v4/
+-- server.py             # Serveur FastAPI principal + slash commands + APIs visualisation
+-- encoder.py            # TriVox2 model + tokenizers (CPU)
+-- memory.py             # Qdrant store + hybrid recall + keyword boost + compression
+-- chunker.py            # Decoupage intelligent (code: 50L/fonction, texte: 100 tokens)
+-- entity_extractor.py   # Extraction de 14 types d'entites + constantes exactes
+-- project_detector.py   # Detection automatique de projet (6 methodes)
+-- config.py             # Configuration centralisee
+-- start.sh              # Script de demarrage
+-- deploy.sh             # Script de deploiement complet
+-- test_proxy.py         # Tests de validation
+-- trivox-proxy.service  # Service systemd
+-- requirements.txt      # Dependances Python
```

---

## Hybrid Search (nouveau v4.1)

Le recall utilise 3 signaux combines pour trouver les souvenirs pertinents :

1. **Vector similarity** — TriVox2 encode la query et compare avec les embeddings Qdrant (cosine similarity)
2. **Keyword boost** — Les mots-cles de la query sont cherches dans le texte des souvenirs. Si des mots matchent, le score est booste (x1.0 a x1.5)
3. **Synonymes FR/EN** — Les mots-cles sont etendus avec des equivalences semantiques :

| Mot-cle | Synonymes ajoutes |
|---------|-------------------|
| habite | vis, belgique, france, pays, ville, adresse, domicile, reside, localisation |
| appelle | nom, prenom, name, identity |
| travaille | projet, project, boulot, job, embedding, training, entrainement |
| carte | gpu, graphique, rtx, gtx, nvidia, vram, cuda |
| budget | million, euros, dollars, financement, cout, investisseur |
| couleur | color, #, hex, rgb, bleu, rouge, vert, jaune |
| modele | model, ia, ai, llm, embedding |

Resultat : "Ou j'habite?" retrouve maintenant "je vis en Belgique" grace au synonyme habite → vis + belgique.

---

## Sensitivity Filtering (nouveau v4.1)

Les donnees sensibles sont automatiquement filtrees **avant** stockage dans Qdrant :

| Pattern | Exemple | Resultat |
|---------|---------|----------|
| API keys | `sk-abc123...` | `[REDACTED]` |
| Bearer tokens | `Bearer eyJ...` | `[REDACTED]` |
| Passwords | `password: secret123` | `password: [REDACTED]` |
| Tokens longs | `ghp_xxxxxxxxxxxx` | `[REDACTED]` |

Les entites contenant `[REDACTED]` sont automatiquement exclues du stockage.

---

## Memory Compression (nouveau v4.1)

La compression supprime les doublons dans les vieilles memoires :

```bash
# Compresser toutes les collections (defaut: memoires >168h / 7 jours)
curl -X POST http://localhost:5556/memory/compress

# Compresser avec un age custom (ex: >72h / 3 jours)
curl -X POST "http://localhost:5556/memory/compress?max_age_hours=72"
```

Algorithme :
1. Pour chaque collection, recupere les memoires plus vieilles que `max_age_hours`
2. Compare chaque memoire avec les autres (cosine similarity)
3. Si deux memoires ont un score > 0.85, supprime la plus ancienne
4. Garde toujours la version la plus recente de chaque souvenir

---

## Modele TriVox2 (non inclus)

- **Architecture** : Transformer 6 couches, d_model=512, 8 heads, FFN 2048
- **Embeddings** : Multi-vocab (FR 50k, EN 50k, Code 32k) + language embedding
- **Sortie** : 768 dimensions, L2-normalise (cosine similarity)
- **Detection de langue** : Automatique (FR/EN/Code)
- **Inference** : CPU uniquement (GPU reserve pour Ollama)

---

## Logs

```bash
tail -f /tmp/memory-proxy-v4.log       # Logs du proxy
journalctl -u trivox-proxy -f          # Si systemd
```

---

## Depannage

| Probleme | Solution |
|----------|----------|
| Port 5556 deja utilise | `fuser -k 5556/tcp` |
| Qdrant pas demarre | `docker start qdrant` |
| Encoder ne charge pas | Verifier `MODEL_PATH` et les tokenizers |
| Ollama timeout | `systemctl status ollama` |
| Memoire vide | Normal si premiere fois. Qdrant persiste dans `/home/aque/qdrant_storage` |
| Proxy crash | `sudo systemctl restart trivox-proxy` ou `pkill -f server.py && bash start.sh` |
