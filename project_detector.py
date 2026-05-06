"""Auto-detect the active project from conversation context.

Strategy (in priority order):
1. Explicit switch: "on parle du projet X", "projet: X", "switch to X"
2. Title-based: first line contains "PROJET HELIOS", "DOSSIER - PROJET X"
3. Content patterns: PROJET followed by uppercase name in message body
4. Recurring proper names: "APOLLON-20B" appears 3+ times -> likely project
5. Known project names from registry
6. Fallback: hash-based anonymous project ID to avoid polluting "general"
"""
import re
import hashlib
import logging
from collections import Counter

log = logging.getLogger("project_detector")

# Known project patterns (extend as needed, auto-populated by register_project)
KNOWN_PROJECTS: dict[str, str] = {
    "nexus": "nexus",
    "aurora": "aurora",
    "trivox": "trivox",
    "trivox2": "trivox",
    "trivoxv2": "trivox",
    "helios": "helios",
    "orion": "orion",
    "titan": "titan",
}

# Model names → associated project (for recurring name detection)
MODEL_TO_PROJECT: dict[str, str] = {
    "apollon-20b": "helios",
    "perseus-35b": "orion",
    "helix-7b": "aurora",
    "solaris-12b": "aurora",
}

# Explicit project switch patterns
SWITCH_PATTERNS = [
    r"(?:on parle|parlons)\s+(?:du|de)\s+(?:projet\s+)?(\w+)",
    r"projet\s*:\s*(\w+)",
    r"(?:switch|change)\s+(?:to|vers)\s+(?:projet\s+)?(\w+)",
    r"(?:contexte|context)\s*:\s*(\w+)",
    r"le projet\s+(\w+)",
    r"mon projet\s+(\w+)",
    r"(?:je (?:travaille|bosse) sur)\s+(\w+)",
]

# Title/header patterns (first line of message)
TITLE_PATTERNS = [
    r"(?:DOSSIER|RAPPORT|DOCUMENT|NOTE|BRIEF)\s+(?:CONFIDENTIEL|TECHNIQUE|INTERNE)?\s*(?:—|-|–|:)\s*PROJET\s+([A-Z][A-Z0-9_-]{2,})",
    r"^PROJET\s+([A-Z][A-Z0-9_-]{2,})",
    r"^---+\s*PROJET\s+([A-Z][A-Z0-9_-]{2,})",
]

# Content detection patterns (uppercase project names in body)
CONTENT_PATTERNS = [
    r"PROJET\s+([A-Z][A-Z0-9_-]{2,})",
    r"PROJECT\s+([A-Z][A-Z0-9_-]{2,})",
    r"[Pp]rojet\s+(\d+)\s*:\s*([A-Z][A-Z0-9_-]{2,})",  # "Projet 1 : TITAN"
]

# Proper name pattern (ALLCAPS-DIGITS like model/project codenames)
PROPER_NAME_RE = re.compile(r"\b([A-Z][A-Z0-9]+-[A-Z0-9]+[A-Z0-9]*)\b")  # e.g. APOLLON-20B

# Words to ignore as project names (common French/English words in CAPS)
IGNORE_NAMES = {
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "ou",
    "the", "and", "for", "with", "from", "that", "this", "not",
    "json", "html", "http", "https", "api", "url", "ip", "tcp",
    "udp", "ssh", "ssl", "tls", "gpu", "cpu", "ram", "vram",
    "ssd", "hdd", "usb", "rtx", "gtx", "cuda", "rom",
    "python", "java", "ruby", "php", "css", "sql",
    "fastapi", "flask", "django", "react", "vue", "node",
    "linux", "ubuntu", "windows", "macos", "docker",
    "git", "github", "gitlab", "npm", "pip",
    "gpt", "llm", "nlp", "bert", "llama",
    "pdf", "csv", "xml", "yaml", "toml",
    "qdrant", "redis", "postgres", "mongo", "sqlite",
    "ollama", "webui", "claude",
    "utf", "ascii", "cors", "rest", "grpc",
}

# Track recurring names per conversation for method 4
_conv_name_counts: dict[str, Counter] = {}


def detect_project(messages: list[dict], current_project: str = "general",
                   conv_id: str = None) -> str:
    """Detect the active project from conversation messages.
    
    Tries 6 methods in priority order:
    1. Explicit switch commands ("on parle du projet X")
    2. Title/header detection (first line of message)
    3. Content patterns (PROJET X in uppercase)
    4. Recurring proper names (APOLLON-20B 3+ times)
    5. Known project keywords
    6. Fallback: hash-based anonymous ID
    
    Args:
        messages: List of {role, content} dicts
        current_project: Previously detected project (sticky)
        conv_id: Conversation ID for tracking recurring names
    
    Returns:
        Project name (lowercase, alphanumeric + hyphens)
    """
    if not messages:
        return current_project

    # Check messages from newest to oldest
    for msg in reversed(messages):
        content = msg.get("content", "")
        if not content or len(content) < 3:
            continue
        
        # 1. Explicit switch command
        detected = _detect_explicit_switch(content)
        if detected:
            log.info(f"Project detected [method=explicit_switch]: {detected}")
            return detected
    
    # 2. Title/header detection (check first message and last user message)
    for msg in messages:
        content = msg.get("content", "")
        if content:
            detected = _detect_from_title(content)
            if detected:
                log.info(f"Project detected [method=title]: {detected}")
                return detected
    
    # 3. Content patterns (PROJET X in uppercase)
    for msg in reversed(messages):
        content = msg.get("content", "")
        if content:
            detected = _detect_from_content(content)
            if detected:
                log.info(f"Project detected [method=content_pattern]: {detected}")
                return detected
    
    # 4. Recurring proper names
    if conv_id:
        all_text = " ".join(m.get("content", "") for m in messages if m.get("content"))
        detected = _detect_from_recurring_names(all_text, conv_id)
        if detected:
            log.info(f"Project detected [method=recurring_names]: {detected}")
            return detected
    
    # 5. Known project keywords
    for msg in reversed(messages):
        content = msg.get("content", "")
        if content:
            detected = _detect_known_project(content)
            if detected:
                log.info(f"Project detected [method=known_keyword]: {detected}")
                return detected
    
    # 6. Fallback: if there's substantial content and current is "general",
    #    create a hash-based anonymous project
    if current_project == "general":
        first_content = _get_first_user_content(messages)
        if first_content and len(first_content) > 100:
            anon = _hash_project_id(first_content)
            log.info(f"Project detected [method=hash_fallback]: {anon}")
            return anon
    
    return current_project


def _detect_explicit_switch(text: str) -> str | None:
    """Method 1: Explicit switch commands."""
    text_lower = text.lower()
    for pattern in SWITCH_PATTERNS:
        m = re.search(pattern, text_lower)
        if m:
            name = m.group(1).lower().strip()
            if name in KNOWN_PROJECTS:
                return KNOWN_PROJECTS[name]
            if _is_valid_project_name(name):
                register_project(name)
                return name
    return None


def _detect_from_title(text: str) -> str | None:
    """Method 2: Detect from title/header (first line of message)."""
    first_line = text.strip().split("\n")[0][:500]
    
    for pattern in TITLE_PATTERNS:
        m = re.search(pattern, first_line)
        if m:
            name = m.group(1).lower().strip("-_ ")
            if _is_valid_project_name(name):
                register_project(name)
                return name
    return None


def _detect_from_content(text: str) -> str | None:
    """Method 3: Detect from content patterns (PROJET X in uppercase)."""
    for pattern in CONTENT_PATTERNS:
        matches = re.findall(pattern, text)
        for match in matches:
            # Handle tuple from "Projet 1 : TITAN" pattern
            if isinstance(match, tuple):
                name = match[-1].lower().strip("-_ ")
            else:
                name = match.lower().strip("-_ ")
            
            if _is_valid_project_name(name):
                register_project(name)
                return name
    return None


def _detect_from_recurring_names(text: str, conv_id: str) -> str | None:
    """Method 4: Detect from recurring proper names (e.g. APOLLON-20B 3+ times)."""
    # Find all proper names (ALLCAPS-DIGITS pattern)
    names = PROPER_NAME_RE.findall(text)
    
    if conv_id not in _conv_name_counts:
        _conv_name_counts[conv_id] = Counter()
    
    counter = _conv_name_counts[conv_id]
    for name in names:
        name_lower = name.lower()
        if name_lower not in IGNORE_NAMES and len(name) >= 4:
            counter[name_lower] += 1
    
    # Check if any name appears 3+ times
    for name_lower, count in counter.most_common(5):
        if count >= 3:
            # Check if it maps to a known project
            if name_lower in MODEL_TO_PROJECT:
                return MODEL_TO_PROJECT[name_lower]
            # Use the name itself as project
            base = name_lower.split("-")[0]  # APOLLON-20B → apollon
            if _is_valid_project_name(base):
                register_project(base)
                return base
    
    return None


def _detect_known_project(text: str) -> str | None:
    """Method 5: Match known project keywords."""
    text_lower = text.lower()
    for keyword, project in KNOWN_PROJECTS.items():
        if keyword in text_lower:
            return project
    return None


def _hash_project_id(text: str) -> str:
    """Method 6: Create anonymous project ID from content hash."""
    h = hashlib.sha256(text[:500].encode("utf-8", errors="ignore")).hexdigest()[:8]
    return f"anon_{h}"


def _get_first_user_content(messages: list[dict]) -> str | None:
    """Get the first user message content."""
    for msg in messages:
        if msg.get("role") == "user" and msg.get("content"):
            return msg["content"]
    return None


def _is_valid_project_name(name: str) -> bool:
    """Check if a string is a valid project name."""
    if not name or len(name) < 3 or len(name) > 30:
        return False
    if name in IGNORE_NAMES:
        return False
    if not re.match(r"^[a-z][a-z0-9_-]*$", name):
        return False
    return True


def _detect_from_text(text: str) -> str | None:
    """Detect project from a single text string.
    
    Used by recall and general detection. Tries methods 1-3, 5.
    """
    # 1. Explicit switch
    detected = _detect_explicit_switch(text)
    if detected:
        return detected
    
    # 2. Title
    detected = _detect_from_title(text)
    if detected:
        return detected
    
    # 3. Content patterns
    detected = _detect_from_content(text)
    if detected:
        return detected
    
    # 5. Known keywords
    detected = _detect_known_project(text)
    if detected:
        return detected
    
    return None


def detect_from_text_strict(text: str) -> str | None:
    """Strict detection — only match KNOWN_PROJECTS. Used for migration."""
    text_lower = text.lower()
    for keyword, project in KNOWN_PROJECTS.items():
        if keyword in text_lower:
            return project
    return None


def register_project(name: str):
    """Register a new project name for future detection."""
    key = name.lower().strip()
    if key and len(key) >= 2 and key not in IGNORE_NAMES:
        KNOWN_PROJECTS[key] = key


def register_model_mapping(model_name: str, project: str):
    """Register a model name → project mapping for recurring name detection."""
    MODEL_TO_PROJECT[model_name.lower()] = project.lower()


def get_registered_projects() -> dict:
    """Return all known projects."""
    return dict(KNOWN_PROJECTS)


def get_conv_name_stats(conv_id: str) -> dict:
    """Return recurring name stats for a conversation."""
    if conv_id in _conv_name_counts:
        return dict(_conv_name_counts[conv_id].most_common(20))
    return {}
