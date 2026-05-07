"""Memory Proxy v4 — Production-grade, project-isolated, entity-aware.

Architecture:
  Open-WebUI → :5556 (this proxy) → Ollama :11434
  
  Each request:
    1. Detect active project from conversation
    2. Recall relevant memories (project-scoped + entities)
    3. Inject into system prompt
    4. Forward to Ollama
    5. Store conversation chunks + entities
    6. Return response

  Memory is model-agnostic — shared across ALL Ollama models.
  Project isolation prevents NEXUS/AURORA/TriVox from mixing.
"""
import json
import re
import time
import uuid
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

from config import OLLAMA_URL, HOST, PORT, MAX_MEMORIES_INJECT
from encoder import Encoder
from memory import MemoryStore
from chunker import chunk_message
from project_detector import detect_project, register_project, register_model_mapping, get_registered_projects, get_conv_name_stats
from entity_extractor import extract_entities, extract_money_and_funder

# ─── Logging ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("proxy")

# ─── Globals ───
encoder: Encoder = None
memory: MemoryStore = None

# Track active project per conversation
conv_projects: dict[str, str] = {}  # conv_id -> project
# Track manually-set projects (via /projet X) — immune to auto-detection override
manual_projects: set[str] = set()  # set of conv_ids with manual override
_global_manual: bool = False  # True if /projet X was used (global sticky)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load encoder and memory on startup."""
    global encoder, memory
    from config import MODEL_PATH, TOK_FR, TOK_EN, TOK_CODE, MAX_SEQ_LEN
    try:
        encoder = Encoder(MODEL_PATH, TOK_FR, TOK_EN, TOK_CODE, MAX_SEQ_LEN)
        memory = MemoryStore(encoder)
        stats = memory.stats()
        total = sum(stats.values()) if isinstance(stats, dict) else 0
        # Restore active project from Qdrant
        saved_project = memory.load_setting("active_project")
        if saved_project:
            conv_projects["_global"] = saved_project
            log.info(f"Restored active project: {saved_project}")
        # Restore manual lock flag
        saved_manual = memory.load_setting("manual_project_lock")
        if saved_manual == "true":
            _global_manual = True
            log.info("Restored manual project lock — auto-detect disabled")
        log.info(f"Ready — {total} memories across {len(stats)} collections")
    except Exception as e:
        log.error(f"Failed to init encoder/memory: {e}")
        log.warning("Running in pass-through mode (no memory)")
        import traceback
        traceback.print_exc()
    yield


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

http_client = httpx.AsyncClient(timeout=300.0)


# ─── Slash Commands ───
# /projet <name>   → switch project
# /oublie <name>   → clear project memory
# /oublie tout     → clear ALL memory
# /projets         → list all projects
# /mémoire         → memory stats
# /contexte        → preview injected context

def detect_slash_command(text: str, current_project: str = "general") -> dict | None:
    """Parse /slash commands from user message. Returns command dict or None."""
    if not text:
        return None
    t = text.strip()
    if not t.startswith("/"):
        return None

    parts = t.split(None, 1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/projet", "/project"):
        if not arg:
            return {"action": "show_project", "project": current_project}
        return {"action": "switch_project", "project": arg.lower().strip()}

    if cmd in ("/oublie", "/oublier", "/forget", "/clear", "/efface", "/delete"):
        if not arg or arg.lower() in ("tout", "all", "everything", "toute la mémoire"):
            return {"action": "clear_all"}
        return {"action": "clear_project", "project": arg.lower().strip()}

    if cmd in ("/projets", "/projects"):
        return {"action": "list_projects"}

    if cmd in ("/mémoire", "/memoire", "/memory", "/stats"):
        return {"action": "stats"}

    if cmd in ("/contexte", "/context"):
        return {"action": "context", "project": current_project}

    if cmd in ("/aide", "/help"):
        return {"action": "help"}

    return None


def execute_slash_command(cmd: dict, model: str, conv_id: str) -> dict:
    """Execute a slash command and return an Ollama-format response."""
    global _global_manual
    action = cmd["action"]
    text = ""

    if action == "switch_project":
        proj = cmd["project"]
        conv_projects[conv_id] = proj
        conv_projects["_global"] = proj  # global default for new conversations
        manual_projects.add(conv_id)
        _global_manual = True  # all new convs inherit manual override
        register_project(proj)
        if memory:
            memory.save_setting("active_project", proj)
            memory.save_setting("manual_project_lock", "true")
        stats = memory.stats() if memory else {}
        count = stats.get(proj, 0)
        text = f"Projet actif : **{proj.upper()}** (verrouillé — la détection auto ne changera pas ce projet)\nMémoires dans ce projet : {count}"

    elif action == "show_project":
        proj = cmd.get("project", "general")
        text = f"Projet actif : **{proj.upper()}**"

    elif action == "clear_all":
        stats_before = memory.stats() if memory else {}
        total = sum(v for v in stats_before.values() if isinstance(v, int))
        if memory:
            memory.clear_all()
            memory.save_setting("active_project", "general")
            memory.save_setting("manual_project_lock", "false")
        conv_projects.clear()
        manual_projects.clear()
        _global_manual = False
        text = f"Toute la mémoire a été effacée ({total} souvenirs supprimés).\nLe verrouillage projet est désactivé — la détection auto est réactivée."

    elif action == "clear_project":
        proj = cmd["project"]
        stats_before = memory.stats() if memory else {}
        count = stats_before.get(proj, 0)
        if memory:
            memory.clear_project(proj)
        text = f"Mémoire du projet **{proj.upper()}** effacée ({count} souvenirs supprimés)."

    elif action == "list_projects":
        stats = memory.stats() if memory else {}
        active = dict(conv_projects)
        lines = ["**Projets en mémoire :**"]
        for name, count in sorted(stats.items()):
            if isinstance(count, int) and count > 0:
                marker = " ← actif" if name in active.values() else ""
                lines.append(f"  - **{name.upper()}** : {count} souvenirs{marker}")
        if len(lines) == 1:
            lines.append("  (aucun projet)")
        text = "\n".join(lines)

    elif action == "stats":
        stats = memory.stats() if memory else {}
        total = sum(v for v in stats.values() if isinstance(v, int))
        lines = [f"**Mémoire totale : {total} souvenirs**"]
        for name, count in sorted(stats.items()):
            if isinstance(count, int) and count > 0:
                lines.append(f"  - {name} : {count}")
        text = "\n".join(lines)

    elif action == "context":
        proj = cmd.get("project", "general")
        if memory:
            try:
                mems = memory.recall("contexte actuel", project=proj, top_k=MAX_MEMORIES_INJECT)
                if mems:
                    ctx = build_memory_context(mems, proj)
                    text = f"**Preview du contexte injecté (projet {proj.upper()}) :**\n```\n{ctx}\n```"
                else:
                    text = f"Aucun souvenir trouvé pour le projet **{proj.upper()}**."
            except Exception as e:
                text = f"Erreur : {e}"
        else:
            text = "Mémoire non initialisée."

    elif action == "help":
        text = (
            "**Commandes mémoire disponibles :**\n"
            "  `/projet <nom>` → switcher sur un projet\n"
            "  `/oublie <nom>` → effacer la mémoire d'un projet\n"
            "  `/oublie tout` → reset complet de toute la mémoire\n"
            "  `/projets` → lister tous les projets en mémoire\n"
            "  `/mémoire` → voir les stats actuelles\n"
            "  `/contexte` → preview du contexte injecté\n"
            "  `/aide` → cette aide"
        )

    return {
        "model": model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "message": {"role": "assistant", "content": text},
        "done": True,
        "done_reason": "slash_command",
    }


# ─── Message Filters ───

def is_meta_message(content: str) -> bool:
    """Detect Open-WebUI internal/meta messages that should NOT be stored."""
    if not content:
        return True
    c = content[:300].lower()
    meta_patterns = [
        "### task: generate", "### task: suggest", "### guidelines:",
        "generate a concise", "summarizing the chat", "follow-up questions",
        "broad tags categorizing", "### chat history:", "```json\n{",
        "translate the following", "### instructions:",
    ]
    for pat in meta_patterns:
        if pat in c:
            return True
    if content.strip().startswith("```json"):
        return True
    return False


def is_just_question(content: str) -> bool:
    """Detect messages that are just questions without informative content."""
    c = content.strip()
    if len(c) < 60 and c.endswith("?"):
        return True
    q_patterns = [
        "comment je m", "quel est mon", "sur quoi je", "c'est quoi",
        "ou est-ce que", "qui suis-je", "tu te souviens", "qu'est-ce que",
        "dis-moi", "rappelle-moi", "tu sais", "quel model",
        "what is my", "what am i", "do you remember", "who am i",
        "where do i", "what's my",
    ]
    c_lower = c.lower()
    if len(c) < 80:
        for pat in q_patterns:
            if pat in c_lower:
                return True
    return False


def is_uninformative_response(content: str) -> bool:
    """Detect assistant responses that don't add new information."""
    c = content.strip().lower()[:300]
    no_info = [
        "je ne sais pas", "je n'ai pas cette information",
        "je ne dispose pas", "pouvez-vous me donner",
        "je n'ai pas de détail", "pourriez-vous préciser",
        "comment puis-je vous aider", "i don't know",
        "i don't have that information",
        # Identity hallucinations
        "je suis claude", "i am claude", "je suis chatgpt", "i am chatgpt",
        "créé par anthropic", "created by anthropic", "created by openai",
        "créé par openai",
        # No access to past conversations
        "je ne peux pas accéder aux informations des conversations passées",
        "je ne peux pas aux informations",
        "chaque conversation commence comme une nouvelle session",
        "i cannot access previous conversations",
        "il n'y a pas d'autre modèle",
        # Confusion / refusal
        "il semble qu'il y ait une confusion",
        "je ne suis pas en mesure",
    ]
    for pat in no_info:
        if pat in c:
            return True
    return False


# ─── Memory Injection ───

def build_memory_context(memories: list[dict], project: str) -> str:
    """Format memories into a context block for the system prompt."""
    if not memories:
        return ""

    lines = [
        f"=== MÉMOIRE [{project.upper()}] ===",
        "Voici ce que tu sais de l'utilisateur grâce aux échanges passés :"
    ]

    # Separate entities from regular memories
    regular = [m for m in memories if m.get("entity_type") == ""]
    entities = [m for m in memories if m.get("entity_type") != ""]

    for i, m in enumerate(regular, 1):
        text = m["text"].strip()
        if m["speaker"] == "user":
            lines.append(f"  {i}. L'utilisateur a dit : \"{text}\"")
        else:
            lines.append(f"  {i}. Tu as répondu : \"{text}\"")

    if entities:
        lines.append("  Faits extraits :")
        for ent in entities:
            etype = ent.get("entity_type", "")
            lines.append(f"    - [{etype}] {ent['text']}")

    lines.append("=== FIN MÉMOIRE ===")
    lines.append("Utilise ces informations pour répondre. Si on te demande quelque chose que tu sais grâce à la mémoire, réponds directement.")
    return "\n".join(lines)


def inject_memories(messages: list[dict], query: str, conv_id: str,
                    project: str = "general") -> list[dict]:
    """Inject relevant memories into the messages list."""
    if not memory or not query:
        return messages

    try:
        memories = memory.recall(query, project=project,
                                 top_k=MAX_MEMORIES_INJECT,
                                 exclude_conv_id=conv_id)
    except Exception as e:
        log.error(f"Recall error: {e}")
        return messages

    if not memories:
        return messages

    context = build_memory_context(memories, project)
    log.info(f"Injecting {len(memories)} memories [project={project}] (top: {memories[0]['score']:.3f})")

    result = list(messages)
    if result and result[0].get("role") == "system":
        result[0] = dict(result[0])
        result[0]["content"] = result[0]["content"] + "\n\n" + context
    else:
        result.insert(0, {"role": "system", "content": context})

    return result


# ─── Storage ───

def store_conversation(messages: list[dict], conv_id: str, project: str = "general"):
    """Store new messages + entities from the conversation."""
    if not memory:
        return

    try:
        to_store = []
        all_text_for_entities = []

        for msg in messages[-2:]:  # Last user + assistant
            role = msg.get("role", "")
            content = msg.get("content", "")

            if is_meta_message(content):
                continue
            if role == "user" and is_just_question(content):
                log.debug(f"Skipped question: {content[:50]}")
                continue
            if role == "assistant" and is_uninformative_response(content):
                log.debug("Skipped uninformative response")
                continue

            if role in ("user", "assistant") and content and len(content) > 10:
                chunks = chunk_message(content, speaker=role, conv_id=conv_id)
                to_store.extend(chunks)
                all_text_for_entities.append(content)

        # Store chunks
        if to_store:
            n = memory.store(to_store, project=project)
            log.info(f"Stored {n} chunks [project={project}]")

        # Extract and store entities
        for text in all_text_for_entities:
            entities = extract_entities(text)
            # Also extract money+funder pairs
            entities.extend(extract_money_and_funder(text))
            if entities:
                ne = memory.store_entities(entities, project=project)
                log.info(f"Stored {ne} entities [project={project}]")

    except Exception as e:
        log.error(f"Store error: {e}")


# ─── API Endpoints ───

@app.get("/")
async def root():
    return {"status": "ok", "service": "memory-proxy-v4", "version": "4.1.0"}


@app.get("/health")
async def health():
    stats = memory.stats() if memory else {}
    total = sum(v for v in stats.values() if isinstance(v, int))
    return {
        "status": "ok",
        "version": "4.1.0",
        "memories_total": total,
        "collections": stats,
        "encoder": "loaded" if encoder else "none",
        "ollama": OLLAMA_URL
    }


# ─── Ollama-compatible endpoints ───

@app.get("/api/tags")
@app.get("/v1/models")
async def list_models():
    """Forward model list to Ollama."""
    try:
        resp = await http_client.get(f"{OLLAMA_URL}/api/tags")
        return resp.json()
    except Exception as e:
        return JSONResponse({"models": [], "error": str(e)}, status_code=502)


@app.post("/api/chat")
async def chat(request: Request):
    """Main chat endpoint — detect project, inject memory, forward, store."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    messages = body.get("messages", [])
    stream = body.get("stream", True)
    conv_id = body.get("conv_id", str(uuid.uuid4())[:8])

    # Get last user message
    user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break

    # Check for /slash commands — intercept before Ollama
    current = conv_projects.get(conv_id, conv_projects.get("_global", "general"))
    slash_cmd = detect_slash_command(user_query, current)
    if slash_cmd:
        model = body.get("model", "unknown")
        log.info(f"Slash command: {slash_cmd['action']} [conv={conv_id}]")
        result = execute_slash_command(slash_cmd, model, conv_id)
        return JSONResponse(result)

    # Detect project — but NEVER override a manual /projet X
    is_manual = conv_id in manual_projects or _global_manual
    if is_manual:
        project = current
        log.info(f"Using manual project: {project} [conv={conv_id}] (auto-detect skipped)")
    else:
        project = detect_project(messages, current, conv_id=conv_id)
    conv_projects[conv_id] = project

    # Inject memories (skip for meta-messages)
    is_meta = is_meta_message(user_query) if user_query else True
    if user_query and memory and not is_meta:
        log.info(f"Recall for: '{user_query[:60]}' [project={project}, conv={conv_id}]")
        messages = inject_memories(messages, user_query, conv_id, project)
        body["messages"] = messages
    elif is_meta:
        log.debug(f"Skipped meta message: {user_query[:40]}")

    if stream:
        return await _stream_chat(body, messages, user_query, conv_id, project)
    else:
        return await _sync_chat(body, messages, user_query, conv_id, project)


async def _stream_chat(body: dict, messages: list, user_query: str,
                       conv_id: str, project: str):
    """Stream chat response from Ollama."""
    full_response = []

    async def generate():
        try:
            async with http_client.stream(
                "POST", f"{OLLAMA_URL}/api/chat",
                json=body, timeout=300.0
            ) as resp:
                async for line in resp.aiter_lines():
                    if line:
                        yield line + "\n"
                        try:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                full_response.append(content)
                        except json.JSONDecodeError:
                            pass
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"

        # Store after streaming completes
        assistant_text = "".join(full_response)
        if assistant_text and user_query and not is_meta_message(user_query):
            store_msgs = messages + [{"role": "assistant", "content": assistant_text}]
            store_conversation(store_msgs, conv_id, project)

    return StreamingResponse(generate(), media_type="application/x-ndjson")


async def _sync_chat(body: dict, messages: list, user_query: str,
                     conv_id: str, project: str):
    """Non-streaming chat."""
    try:
        resp = await http_client.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=300.0)
        data = resp.json()

        assistant_text = data.get("message", {}).get("content", "")
        if assistant_text and user_query and not is_meta_message(user_query):
            store_msgs = messages + [{"role": "assistant", "content": assistant_text}]
            store_conversation(store_msgs, conv_id, project)

        return JSONResponse(data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)


# ─── OpenAI-compatible endpoints ───

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat endpoint."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    messages = body.get("messages", [])
    model = body.get("model", "")
    stream = body.get("stream", False)
    conv_id = str(uuid.uuid4())[:8]

    user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break

    # Check for /slash commands
    current = conv_projects.get(conv_id, conv_projects.get("_global", "general"))
    slash_cmd = detect_slash_command(user_query, current)
    if slash_cmd:
        log.info(f"Slash command (OpenAI): {slash_cmd['action']} [conv={conv_id}]")
        result = execute_slash_command(slash_cmd, model, conv_id)
        # Convert to OpenAI format
        openai_result = {
            "id": f"chatcmpl-{conv_id}",
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "message": result["message"], "finish_reason": "stop"}],
        }
        return JSONResponse(openai_result)

    # Detect project — but NEVER override a manual /projet X
    is_manual = conv_id in manual_projects or _global_manual
    if is_manual:
        project = current
        log.info(f"Using manual project: {project} [conv={conv_id}] (auto-detect skipped)")
    else:
        project = detect_project(messages, current, conv_id=conv_id)
    conv_projects[conv_id] = project

    if user_query and memory and not is_meta_message(user_query):
        messages = inject_memories(messages, user_query, conv_id, project)

    ollama_body = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if "temperature" in body:
        ollama_body["options"] = {"temperature": body["temperature"]}
    if "max_tokens" in body:
        ollama_body.setdefault("options", {})["num_predict"] = body["max_tokens"]

    if stream:
        return await _stream_openai(ollama_body, messages, user_query, conv_id, model, project)
    else:
        return await _sync_openai(ollama_body, messages, user_query, conv_id, model, project)


async def _stream_openai(body: dict, messages: list, user_query: str,
                         conv_id: str, model: str, project: str):
    """Stream in OpenAI SSE format."""
    full_response = []
    msg_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    async def generate():
        try:
            async with http_client.stream(
                "POST", f"{OLLAMA_URL}/api/chat",
                json=body, timeout=300.0
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        content = data.get("message", {}).get("content", "")
                        if content:
                            full_response.append(content)
                            chunk = {
                                "id": msg_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": content},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                        if data.get("done"):
                            final = {
                                "id": msg_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop"
                                }]
                            }
                            yield f"data: {json.dumps(final)}\n\n"
                            yield "data: [DONE]\n\n"
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            yield f"data: {json.dumps({'error': {'message': str(e)}})}\n\n"

        assistant_text = "".join(full_response)
        if assistant_text and user_query and not is_meta_message(user_query):
            store_msgs = messages + [{"role": "assistant", "content": assistant_text}]
            store_conversation(store_msgs, conv_id, project)

    return StreamingResponse(generate(), media_type="text/event-stream")


async def _sync_openai(body: dict, messages: list, user_query: str,
                       conv_id: str, model: str, project: str):
    """Non-streaming OpenAI response."""
    try:
        resp = await http_client.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=300.0)
        data = resp.json()
        assistant_text = data.get("message", {}).get("content", "")

        if assistant_text and user_query and not is_meta_message(user_query):
            store_msgs = messages + [{"role": "assistant", "content": assistant_text}]
            store_conversation(store_msgs, conv_id, project)

        return JSONResponse({
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": assistant_text},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        })
    except Exception as e:
        return JSONResponse({"error": {"message": str(e)}}, status_code=502)


# ─── Memory management endpoints (BEFORE catch-all) ───

@app.get("/memory/stats")
async def memory_stats():
    """Get per-project memory statistics."""
    if not memory:
        return {"error": "Memory not initialized"}
    return {
        "total": memory.count(),
        "collections": memory.stats()
    }


@app.post("/memory/search")
async def memory_search(request: Request):
    """Search memories with optional project filter."""
    if not memory:
        return JSONResponse({"error": "Memory not initialized"}, status_code=503)
    body = await request.json()
    query = body.get("query", "")
    top_k = body.get("top_k", 5)
    project = body.get("project", "general")
    if not query:
        return JSONResponse({"error": "query required"}, status_code=400)
    results = memory.recall(query, project=project, top_k=top_k)
    return {"query": query, "project": project, "results": results}


@app.post("/memory/clear")
async def memory_clear(request: Request):
    """Clear memories. Pass {"project": "x"} to clear one, or {} to clear all."""
    if not memory:
        return JSONResponse({"error": "Memory not initialized"}, status_code=503)
    try:
        body = await request.json()
    except Exception:
        body = {}

    project = body.get("project")
    if project:
        count = memory.clear_project(project)
        return {"status": "cleared", "project": project, "deleted": count}
    else:
        memory.clear_all()
        return {"status": "cleared_all"}


@app.post("/memory/register_project")
async def register_project_endpoint(request: Request):
    """Register a new project name for detection."""
    body = await request.json()
    name = body.get("name", "").strip()
    if not name:
        return JSONResponse({"error": "name required"}, status_code=400)
    register_project(name)
    # Optionally register model→project mapping
    model = body.get("model", "").strip()
    if model:
        register_model_mapping(model, name)
    return {"status": "registered", "project": name.lower()}


@app.get("/memory/projects")
async def list_projects():
    """List all registered projects and active conversations."""
    return {
        "registered": get_registered_projects(),
        "active_conversations": dict(conv_projects),
    }


@app.get("/memory/project_debug/{conv_id}")
async def project_debug(conv_id: str):
    """Debug project detection for a specific conversation."""
    return {
        "conv_id": conv_id,
        "current_project": conv_projects.get(conv_id, conv_projects.get("_global", "general")),
        "recurring_names": get_conv_name_stats(conv_id),
    }


@app.post("/memory/migrate")
async def memory_migrate(request: Request):
    """Migrate old v3/v4 collections into new project-scoped format."""
    if not memory:
        return JSONResponse({"error": "Memory not initialized"}, status_code=503)
    try:
        body = await request.json()
    except Exception:
        body = {}
    
    from project_detector import detect_from_text_strict
    collections = body.get("collections", ["memory_v4", "episodic"])
    result = memory.migrate_legacy(collections, project_detector=detect_from_text_strict)
    return {"status": "migrated", "results": result}


# ─── Memory Visualization ───

@app.get("/memory/timeline")
async def memory_timeline(project: str = None, limit: int = 50):
    """Get a chronological timeline of memories, optionally filtered by project."""
    if not memory:
        return JSONResponse({"error": "Memory not initialized"}, status_code=503)
    try:
        from qdrant_client.models import OrderBy
        results = []
        collections = [c.name for c in memory.client.get_collections().collections]
        
        for col in collections:
            if not col.startswith("mem_") or col in ("mem_entities", "mem_settings"):
                continue
            proj_name = col.replace("mem_", "")
            if project and proj_name != project.lower():
                continue
            try:
                points, _ = memory.client.scroll(
                    collection_name=col, limit=limit,
                    with_payload=True, with_vectors=False
                )
                for p in points:
                    results.append({
                        "text": p.payload.get("text", "")[:200],
                        "speaker": p.payload.get("speaker", ""),
                        "project": proj_name,
                        "timestamp": p.payload.get("timestamp", 0),
                        "type": "memory",
                    })
            except Exception:
                pass
        
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        return {"timeline": results[:limit], "total": len(results)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/memory/graph")
async def memory_graph():
    """Get a graph representation of entities and their relationships."""
    if not memory:
        return JSONResponse({"error": "Memory not initialized"}, status_code=503)
    try:
        nodes = []
        edges = []
        entity_by_type = {}
        
        points, _ = memory.client.scroll(
            collection_name="mem_entities", limit=200,
            with_payload=True, with_vectors=False
        )
        
        for p in points:
            etype = p.payload.get("entity_type", "unknown")
            value = p.payload.get("entity_value", "")[:60]
            proj = p.payload.get("project", "general")
            
            node_id = f"{etype}:{value}"
            if node_id not in entity_by_type:
                entity_by_type[node_id] = True
                nodes.append({
                    "id": node_id,
                    "label": value,
                    "type": etype,
                    "project": proj,
                    "importance": p.payload.get("importance", 0.5),
                })
                # Edge: entity → project
                edges.append({
                    "source": node_id,
                    "target": f"project:{proj}",
                    "relation": "belongs_to",
                })
        
        # Add project nodes
        collections = [c.name for c in memory.client.get_collections().collections]
        for col in collections:
            if col.startswith("mem_") and col not in ("mem_entities", "mem_settings"):
                proj_name = col.replace("mem_", "")
                info = memory.client.get_collection(col)
                nodes.append({
                    "id": f"project:{proj_name}",
                    "label": proj_name.upper(),
                    "type": "project",
                    "memories": info.points_count,
                })
        
        return {"nodes": nodes, "edges": edges}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/memory/entity_types")
async def entity_types():
    """Get summary of entities grouped by type."""
    if not memory:
        return JSONResponse({"error": "Memory not initialized"}, status_code=503)
    try:
        from collections import Counter
        type_counts = Counter()
        type_examples = {}
        
        points, _ = memory.client.scroll(
            collection_name="mem_entities", limit=500,
            with_payload=True, with_vectors=False
        )
        
        for p in points:
            etype = p.payload.get("entity_type", "unknown")
            value = p.payload.get("entity_value", "")[:60]
            type_counts[etype] += 1
            if etype not in type_examples:
                type_examples[etype] = []
            if len(type_examples[etype]) < 3:
                type_examples[etype].append(value)
        
        return {
            "entity_types": {
                etype: {"count": count, "examples": type_examples.get(etype, [])}
                for etype, count in type_counts.most_common()
            }
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/memory/compress")
async def compress_memory(request: Request):
    """Compress old memories by removing duplicates."""
    if not memory:
        return JSONResponse({"error": "Memory not initialized"}, status_code=503)
    try:
        body = await request.json()
    except Exception:
        body = {}
    max_age = body.get("max_age_hours", 168)  # Default 1 week
    results = memory.compress_all(max_age_hours=max_age)
    return {"status": "compressed", "results": results}


@app.get("/memory/confidence")
async def memory_confidence(query: str, project: str = None):
    """Search memory and return results with confidence scores and source info."""
    if not memory:
        return JSONResponse({"error": "Memory not initialized"}, status_code=503)
    
    proj = project or conv_projects.get("_global", "general")
    results = memory.recall(query, project=proj, top_k=10)
    
    enriched = []
    for r in results:
        confidence = "high" if r["score"] > 0.5 else "medium" if r["score"] > 0.3 else "low"
        enriched.append({
            "text": r["text"][:200],
            "score": r["score"],
            "confidence": confidence,
            "project": r.get("project", "general"),
            "speaker": r.get("speaker", ""),
            "entity_type": r.get("entity_type", ""),
            "age_hours": round((time.time() - r.get("timestamp", time.time())) / 3600, 1),
        })
    
    return {"query": query, "project": proj, "results": enriched}


# ─── Catch-all proxy for other Ollama endpoints ───

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_catch_all(request: Request, path: str):
    """Forward any unhandled request to Ollama."""
    url = f"{OLLAMA_URL}/{path}"
    try:
        if request.method == "GET":
            resp = await http_client.get(url)
        else:
            body = await request.body()
            resp = await http_client.request(
                method=request.method, url=url,
                content=body,
                headers={"content-type": request.headers.get("content-type", "application/json")}
            )
        return Response(
            content=resp.content,
            status_code=resp.status_code,
            media_type=resp.headers.get("content-type")
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)


# ─── Run ───

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
