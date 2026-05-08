"""Codebase indexer — indexes a Git repo into Qdrant via TriVox embeddings.

Usage via chat:
  /index /path/to/repo           → index local repo
  /index https://github.com/...  → clone + index
  /switch repo_name              → switch active codebase

Each repo gets its own Qdrant collection: "code_{repo_name}"
Chunks by function/class with file path + line numbers for citation.
Extracts architecture: classes, functions, imports, dependencies.
"""
import os
import re
import time
import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("indexer")

# File extensions to index
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".rb",
    ".php", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift", ".kt",
    ".scala", ".lua", ".sh", ".bash", ".yml", ".yaml", ".toml",
    ".json", ".xml", ".sql", ".md", ".rst", ".txt", ".cfg", ".ini",
    ".dockerfile", ".tf", ".hcl",
}

# Directories to skip
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", "env",
    ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
    ".next", ".nuxt", "target", "vendor", ".cargo",
    ".idea", ".vscode", "coverage", ".eggs",
}

# Max file size (500KB)
MAX_FILE_SIZE = 500_000

# Chunk settings
MAX_CHUNK_LINES = 60
OVERLAP_LINES = 5

# Function/class detection
FUNC_RE = re.compile(r'^([ \t]*(?:async\s+)?(?:def|function)\s+(\w+))', re.MULTILINE)
CLASS_RE = re.compile(r'^([ \t]*class\s+(\w+))', re.MULTILINE)
METHOD_RE = re.compile(r'^([ \t]+(?:async\s+)?(?:def|function)\s+(\w+))', re.MULTILINE)
IMPORT_RE = re.compile(r'^(?:import|from|require|use|include|using)\s+(.+)', re.MULTILINE)


@dataclass
class CodeChunk:
    text: str
    file_path: str       # relative to repo root
    start_line: int
    end_line: int
    chunk_type: str      # "function", "class", "module", "config", "doc"
    name: str            # function/class name or ""
    language: str        # "python", "javascript", etc.


@dataclass
class RepoArchitecture:
    """Extracted architecture summary."""
    repo_name: str
    total_files: int = 0
    total_lines: int = 0
    languages: dict = field(default_factory=dict)     # lang -> file count
    classes: list = field(default_factory=list)        # [(name, file, line)]
    functions: list = field(default_factory=list)      # [(name, file, line)]
    imports: dict = field(default_factory=dict)        # package -> count
    entry_points: list = field(default_factory=list)   # main files
    config_files: list = field(default_factory=list)   # config/setup files


def _detect_language(ext: str) -> str:
    mapping = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".tsx": "typescript", ".jsx": "javascript", ".java": "java",
        ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
        ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp",
        ".cs": "csharp", ".swift": "swift", ".kt": "kotlin",
        ".scala": "scala", ".lua": "lua", ".sh": "bash",
        ".yml": "yaml", ".yaml": "yaml", ".toml": "toml",
        ".json": "json", ".sql": "sql", ".md": "markdown",
    }
    return mapping.get(ext, "text")


def _is_entry_point(filename: str) -> bool:
    return filename.lower() in {
        "main.py", "app.py", "server.py", "index.js", "index.ts",
        "main.go", "main.rs", "main.java", "manage.py", "wsgi.py",
        "asgi.py", "cli.py", "__main__.py",
    }


def _is_config_file(filename: str) -> bool:
    return filename.lower() in {
        "package.json", "pyproject.toml", "setup.py", "setup.cfg",
        "cargo.toml", "go.mod", "pom.xml", "build.gradle",
        "docker-compose.yml", "dockerfile", "makefile",
        "requirements.txt", "pipfile", ".env.example",
        "tsconfig.json", "webpack.config.js", "vite.config.ts",
    }


def scan_repo(repo_path: str) -> tuple[list[tuple[str, str]], RepoArchitecture]:
    """Scan a repo and return (file_path, content) pairs + architecture.
    
    Returns:
        files: list of (relative_path, content) tuples
        arch: RepoArchitecture with extracted metadata
    """
    repo_path = os.path.abspath(repo_path)
    repo_name = os.path.basename(repo_path)
    arch = RepoArchitecture(repo_name=repo_name)
    files = []

    for root, dirs, filenames in os.walk(repo_path):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for fname in filenames:
            fpath = os.path.join(root, fname)
            rel_path = os.path.relpath(fpath, repo_path)
            ext = os.path.splitext(fname)[1].lower()

            # Skip non-code files
            if ext not in CODE_EXTENSIONS and not _is_config_file(fname):
                continue

            # Skip large files
            try:
                size = os.path.getsize(fpath)
                if size > MAX_FILE_SIZE or size == 0:
                    continue
            except OSError:
                continue

            # Read file
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except Exception:
                continue

            files.append((rel_path, content))
            lang = _detect_language(ext)
            arch.total_files += 1
            arch.total_lines += content.count("\n") + 1
            arch.languages[lang] = arch.languages.get(lang, 0) + 1

            # Entry points
            if _is_entry_point(fname):
                arch.entry_points.append(rel_path)
            if _is_config_file(fname):
                arch.config_files.append(rel_path)

            # Extract classes
            for m in CLASS_RE.finditer(content):
                line_num = content[:m.start()].count("\n") + 1
                arch.classes.append((m.group(2), rel_path, line_num))

            # Extract functions (top-level only for architecture)
            for m in FUNC_RE.finditer(content):
                indent = len(m.group(1)) - len(m.group(1).lstrip())
                if indent == 0:  # top-level only
                    line_num = content[:m.start()].count("\n") + 1
                    arch.functions.append((m.group(2), rel_path, line_num))

            # Extract imports
            for m in IMPORT_RE.finditer(content):
                imp = m.group(1).strip().split()[0].split(".")[0].strip("'\"")
                if imp and len(imp) > 1:
                    arch.imports[imp] = arch.imports.get(imp, 0) + 1

    return files, arch


def chunk_file(rel_path: str, content: str) -> list[CodeChunk]:
    """Chunk a single file by function/class boundaries."""
    ext = os.path.splitext(rel_path)[1].lower()
    lang = _detect_language(ext)
    lines = content.split("\n")
    total_lines = len(lines)

    # Find all function/class boundaries
    boundaries = []  # (line_idx, name, type)
    for m in CLASS_RE.finditer(content):
        line_idx = content[:m.start()].count("\n")
        boundaries.append((line_idx, m.group(2), "class"))
    for m in FUNC_RE.finditer(content):
        line_idx = content[:m.start()].count("\n")
        boundaries.append((line_idx, m.group(2), "function"))

    boundaries.sort(key=lambda x: x[0])

    # If no boundaries or small file, chunk as one block
    if not boundaries or total_lines <= MAX_CHUNK_LINES:
        chunk_type = "config" if _is_config_file(os.path.basename(rel_path)) else "module"
        return [CodeChunk(
            text=f"# File: {rel_path}\n{content}",
            file_path=rel_path,
            start_line=1,
            end_line=total_lines,
            chunk_type=chunk_type,
            name=os.path.basename(rel_path),
            language=lang,
        )]

    chunks = []

    # Header chunk (imports, module-level code before first boundary)
    if boundaries[0][0] > 0:
        header_end = boundaries[0][0]
        header_text = "\n".join(lines[:header_end]).strip()
        if header_text and len(header_text) > 20:
            chunks.append(CodeChunk(
                text=f"# File: {rel_path} (imports/header)\n{header_text}",
                file_path=rel_path,
                start_line=1,
                end_line=header_end,
                chunk_type="module",
                name=f"{os.path.basename(rel_path)}:header",
                language=lang,
            ))

    # Each function/class = 1 chunk
    for i, (start_line, name, btype) in enumerate(boundaries):
        end_line = boundaries[i + 1][0] if i + 1 < len(boundaries) else total_lines
        block = "\n".join(lines[start_line:end_line]).rstrip()

        if not block.strip():
            continue

        # Sub-chunk if too large
        block_lines = block.split("\n")
        if len(block_lines) > MAX_CHUNK_LINES:
            for j in range(0, len(block_lines), MAX_CHUNK_LINES - OVERLAP_LINES):
                sub = "\n".join(block_lines[j:j + MAX_CHUNK_LINES]).strip()
                if sub:
                    actual_start = start_line + j + 1
                    actual_end = min(start_line + j + MAX_CHUNK_LINES, end_line)
                    chunks.append(CodeChunk(
                        text=f"# File: {rel_path}:{actual_start}-{actual_end}\n# {btype}: {name}\n{sub}",
                        file_path=rel_path,
                        start_line=actual_start,
                        end_line=actual_end,
                        chunk_type=btype,
                        name=name,
                        language=lang,
                    ))
        else:
            chunks.append(CodeChunk(
                text=f"# File: {rel_path}:{start_line + 1}-{end_line}\n# {btype}: {name}\n{block}",
                file_path=rel_path,
                start_line=start_line + 1,
                end_line=end_line,
                chunk_type=btype,
                name=name,
                language=lang,
            ))

    return chunks


def index_repo(repo_path: str, encoder, qdrant_client, embed_dim: int = 768,
               progress_callback=None) -> dict:
    """Index an entire repo into Qdrant.
    
    Args:
        repo_path: path to the repo
        encoder: TriVox encoder instance
        qdrant_client: Qdrant client
        embed_dim: embedding dimension
        progress_callback: optional function(current, total, message) for progress
    
    Returns:
        dict with indexing stats
    """
    from qdrant_client.models import VectorParams, Distance, PointStruct
    import uuid

    repo_path = os.path.abspath(repo_path)
    repo_name = os.path.basename(repo_path)
    collection = f"code_{repo_name.lower().replace(' ', '_').replace('-', '_')}"

    t0 = time.time()

    # Scan repo
    if progress_callback:
        progress_callback(0, 0, f"Scanning {repo_name}...")
    files, arch = scan_repo(repo_path)
    log.info(f"Scanned {repo_name}: {arch.total_files} files, {arch.total_lines} lines")

    # Create/recreate collection
    try:
        collections = [c.name for c in qdrant_client.get_collections().collections]
        if collection in collections:
            qdrant_client.delete_collection(collection)
    except Exception:
        pass

    qdrant_client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE)
    )

    # Chunk all files
    all_chunks = []
    for rel_path, content in files:
        chunks = chunk_file(rel_path, content)
        all_chunks.extend(chunks)

    total_chunks = len(all_chunks)
    log.info(f"Created {total_chunks} chunks from {arch.total_files} files")

    # Encode and store in batches
    batch_size = 32
    stored = 0
    for i in range(0, total_chunks, batch_size):
        batch = all_chunks[i:i + batch_size]
        points = []
        for chunk in batch:
            embedding = encoder.encode(chunk.text)
            point_id = str(uuid.uuid4())
            payload = {
                "text": chunk.text,
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "chunk_type": chunk.chunk_type,
                "name": chunk.name,
                "language": chunk.language,
                "repo": repo_name,
                "timestamp": time.time(),
            }
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

        qdrant_client.upsert(collection_name=collection, points=points)
        stored += len(points)

        if progress_callback:
            progress_callback(stored, total_chunks,
                              f"Indexed {stored}/{total_chunks} chunks...")

    elapsed = time.time() - t0

    # Store architecture as a special chunk
    arch_text = format_architecture(arch)
    arch_emb = encoder.encode(arch_text)
    from qdrant_client.models import PointStruct
    qdrant_client.upsert(
        collection_name=collection,
        points=[PointStruct(
            id=str(uuid.uuid4()),
            vector=arch_emb,
            payload={
                "text": arch_text,
                "file_path": "_architecture",
                "start_line": 0,
                "end_line": 0,
                "chunk_type": "architecture",
                "name": "architecture",
                "language": "text",
                "repo": repo_name,
                "timestamp": time.time(),
            }
        )]
    )

    return {
        "repo": repo_name,
        "collection": collection,
        "files": arch.total_files,
        "lines": arch.total_lines,
        "chunks": total_chunks,
        "languages": arch.languages,
        "classes": len(arch.classes),
        "functions": len(arch.functions),
        "elapsed_seconds": round(elapsed, 1),
        "entry_points": arch.entry_points,
    }


def format_architecture(arch: RepoArchitecture) -> str:
    """Format architecture as readable text for storage and display."""
    lines = [
        f"# Architecture: {arch.repo_name}",
        f"Files: {arch.total_files} | Lines: {arch.total_lines}",
        "",
        "## Languages:",
    ]
    for lang, count in sorted(arch.languages.items(), key=lambda x: -x[1]):
        lines.append(f"  - {lang}: {count} files")

    if arch.entry_points:
        lines.append("\n## Entry Points:")
        for ep in arch.entry_points[:10]:
            lines.append(f"  - {ep}")

    if arch.config_files:
        lines.append("\n## Config Files:")
        for cf in arch.config_files[:10]:
            lines.append(f"  - {cf}")

    if arch.classes:
        lines.append(f"\n## Classes ({len(arch.classes)}):")
        for name, fpath, line in arch.classes[:30]:
            lines.append(f"  - {name} ({fpath}:{line})")

    if arch.functions:
        lines.append(f"\n## Top-level Functions ({len(arch.functions)}):")
        for name, fpath, line in arch.functions[:30]:
            lines.append(f"  - {name}() ({fpath}:{line})")

    if arch.imports:
        lines.append(f"\n## Dependencies (top {min(20, len(arch.imports))}):")
        for pkg, count in sorted(arch.imports.items(), key=lambda x: -x[1])[:20]:
            lines.append(f"  - {pkg} (used {count}x)")

    return "\n".join(lines)


def clone_repo(url: str, target_dir: str = "/tmp/trivox_repos") -> str:
    """Clone a Git repo and return the local path."""
    os.makedirs(target_dir, exist_ok=True)
    # Extract repo name from URL
    name = url.rstrip("/").split("/")[-1].replace(".git", "")
    local_path = os.path.join(target_dir, name)

    if os.path.exists(local_path):
        # Pull latest
        subprocess.run(["git", "-C", local_path, "pull"], capture_output=True, timeout=120)
    else:
        subprocess.run(["git", "clone", "--depth=1", url, local_path],
                        capture_output=True, timeout=300)

    if not os.path.isdir(local_path):
        raise FileNotFoundError(f"Clone failed: {url}")

    return local_path


def search_code(query: str, encoder, qdrant_client, collection: str,
                top_k: int = 5) -> list[dict]:
    """Search indexed code with TriVox embeddings."""
    query_emb = encoder.encode(query)
    hits = qdrant_client.search(
        collection_name=collection,
        query_vector=query_emb,
        limit=top_k,
        with_payload=True,
    )
    results = []
    for h in hits:
        results.append({
            "text": h.payload.get("text", ""),
            "file_path": h.payload.get("file_path", ""),
            "start_line": h.payload.get("start_line", 0),
            "end_line": h.payload.get("end_line", 0),
            "chunk_type": h.payload.get("chunk_type", ""),
            "name": h.payload.get("name", ""),
            "language": h.payload.get("language", ""),
            "score": h.score,
        })
    return results
