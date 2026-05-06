"""Text and code chunker for conversations.

Strategy:
- Plain text: split by sentences, group into ~100 token chunks with overlap
- Code: split by function/class boundaries, sub-chunk if > 100 lines
- Each chunk stored with metadata (speaker, timestamp, conv_id)
"""
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    speaker: str  # "user" or "assistant"
    chunk_idx: int
    metadata: dict


# ─── Code detection ───

_CODE_FENCE_RE = re.compile(r'```(\w*)\n([\s\S]*?)```')
_FUNC_RE = re.compile(r'^([ \t]*(?:async\s+)?def\s+\w+)', re.MULTILINE)
_CLASS_RE = re.compile(r'^([ \t]*class\s+\w+)', re.MULTILINE)
_BLOCK_START_RE = re.compile(r'^([ \t]*(?:async\s+)?(?:def|class)\s+\w+[^\n]*)', re.MULTILINE)

MAX_CODE_LINES = 100


def _is_code(text: str) -> bool:
    """Heuristic: does this text look like code?"""
    lines = text.strip().split('\n')
    if len(lines) < 3:
        return False
    code_signals = 0
    for line in lines[:30]:
        stripped = line.strip()
        if stripped.startswith(('def ', 'class ', 'import ', 'from ', 'async def ')):
            code_signals += 2
        elif stripped.startswith(('#', '//', '/*')):
            code_signals += 1
        elif '=' in stripped and not stripped.startswith(('http', 'url')):
            code_signals += 0.5
        elif stripped.endswith((':',  '{', '}', ');')):
            code_signals += 0.5
    return code_signals >= 3


def _chunk_code_block(code: str, speaker: str, chunk_idx_start: int,
                      meta: dict) -> list[Chunk]:
    """Chunk a code block by function/class boundaries."""
    lines = code.split('\n')

    # Find all function/class boundaries
    boundaries = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r'(?:async\s+)?def\s+\w+', stripped) or re.match(r'class\s+\w+', stripped):
            boundaries.append(i)

    if not boundaries:
        # No functions/classes found — chunk by line groups
        return _chunk_lines(lines, speaker, chunk_idx_start, meta)

    chunks = []
    idx = chunk_idx_start

    # Chunk before first boundary (module-level code, imports, constants)
    if boundaries[0] > 0:
        header = '\n'.join(lines[:boundaries[0]]).strip()
        if header:
            for sub in _sub_chunk_lines(header, MAX_CODE_LINES):
                chunks.append(Chunk(text=sub, speaker=speaker, chunk_idx=idx, metadata=meta.copy()))
                idx += 1

    # Each function/class = 1 chunk
    for i, start in enumerate(boundaries):
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(lines)
        block = '\n'.join(lines[start:end]).rstrip()
        if not block.strip():
            continue

        # Sub-chunk if > MAX_CODE_LINES
        block_lines = block.split('\n')
        if len(block_lines) > MAX_CODE_LINES:
            for sub in _sub_chunk_lines(block, MAX_CODE_LINES):
                chunks.append(Chunk(text=sub, speaker=speaker, chunk_idx=idx, metadata=meta.copy()))
                idx += 1
        else:
            chunks.append(Chunk(text=block, speaker=speaker, chunk_idx=idx, metadata=meta.copy()))
            idx += 1

    return chunks


def _chunk_lines(lines: list[str], speaker: str, chunk_idx_start: int,
                 meta: dict) -> list[Chunk]:
    """Chunk raw lines into groups of MAX_CODE_LINES."""
    chunks = []
    idx = chunk_idx_start
    for i in range(0, len(lines), MAX_CODE_LINES):
        block = '\n'.join(lines[i:i + MAX_CODE_LINES]).strip()
        if block:
            chunks.append(Chunk(text=block, speaker=speaker, chunk_idx=idx, metadata=meta.copy()))
            idx += 1
    return chunks


def _sub_chunk_lines(text: str, max_lines: int) -> list[str]:
    """Split text into sub-chunks of max_lines with some overlap."""
    lines = text.split('\n')
    overlap = 5
    result = []
    i = 0
    while i < len(lines):
        end = min(i + max_lines, len(lines))
        block = '\n'.join(lines[i:end]).strip()
        if block:
            result.append(block)
        i = end - overlap if end < len(lines) else len(lines)
    return result


# ─── Text chunking (original logic) ───

def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Handles code blocks specially."""
    code_blocks = []
    def replace_code(m):
        code_blocks.append(m.group(0))
        return f"__CODE_BLOCK_{len(code_blocks)-1}__"

    text = re.sub(r'```[\s\S]*?```', replace_code, text)
    text = re.sub(r'`[^`]+`', replace_code, text)

    parts = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*(?=\S)', text)

    result = []
    for part in parts:
        for i, block in enumerate(code_blocks):
            part = part.replace(f"__CODE_BLOCK_{i}__", block)
        part = part.strip()
        if part:
            result.append(part)
    return result


def chunk_message(text: str, speaker: str, max_tokens: int = 100,
                  overlap: int = 20, conv_id: str = "", metadata: dict = None) -> list[Chunk]:
    """Chunk a message into overlapping segments for memory storage.

    Detects code blocks and uses function/class-aware chunking for code.
    Uses sentence-based chunking for plain text.
    """
    if not text or not text.strip():
        return []

    meta = metadata or {}
    meta["conv_id"] = conv_id
    meta["speaker"] = speaker

    all_chunks = []
    idx = 0

    # Extract fenced code blocks and chunk them separately
    parts = _CODE_FENCE_RE.split(text)
    # parts: [text, lang, code, text, lang, code, text, ...]

    i = 0
    while i < len(parts):
        if i + 2 < len(parts) and (i % 3) == 0:
            # Check if next part is a code block
            pre_text = parts[i].strip()
            lang = parts[i + 1] if i + 1 < len(parts) else ""
            code = parts[i + 2] if i + 2 < len(parts) else ""

            # Chunk pre-text as prose
            if pre_text:
                prose_chunks = _chunk_prose(pre_text, speaker, idx, max_tokens, overlap, meta)
                all_chunks.extend(prose_chunks)
                idx += len(prose_chunks)

            # Chunk code block
            if code.strip():
                code_meta = meta.copy()
                code_meta["language"] = lang
                code_meta["is_code"] = True
                code_chunks = _chunk_code_block(code, speaker, idx, code_meta)
                all_chunks.extend(code_chunks)
                idx += len(code_chunks)

            i += 3
        else:
            # Plain text remainder
            plain = parts[i].strip() if i < len(parts) else ""
            if plain:
                # Check if it looks like raw code (no fences)
                if _is_code(plain):
                    code_meta = meta.copy()
                    code_meta["is_code"] = True
                    code_chunks = _chunk_code_block(plain, speaker, idx, code_meta)
                    all_chunks.extend(code_chunks)
                    idx += len(code_chunks)
                else:
                    prose_chunks = _chunk_prose(plain, speaker, idx, max_tokens, overlap, meta)
                    all_chunks.extend(prose_chunks)
                    idx += len(prose_chunks)
            i += 1

    return all_chunks


def _chunk_prose(text: str, speaker: str, chunk_idx_start: int,
                 max_tokens: int, overlap: int, meta: dict) -> list[Chunk]:
    """Chunk plain text by sentences with token-based grouping."""
    sentences = split_sentences(text)
    if not sentences:
        return []

    word_count = len(text.split())
    if word_count <= max_tokens:
        return [Chunk(text=text.strip(), speaker=speaker,
                      chunk_idx=chunk_idx_start, metadata=meta.copy())]

    chunks = []
    current_words = []
    current_count = 0
    idx = chunk_idx_start

    for sentence in sentences:
        s_words = sentence.split()
        s_count = len(s_words)

        if current_count + s_count > max_tokens and current_words:
            chunk_text = " ".join(current_words)
            chunks.append(Chunk(text=chunk_text, speaker=speaker,
                                chunk_idx=idx, metadata=meta.copy()))
            idx += 1
            overlap_words = current_words[-overlap:] if overlap > 0 else []
            current_words = overlap_words + s_words
            current_count = len(current_words)
        else:
            current_words.extend(s_words)
            current_count += s_count

    if current_words:
        chunk_text = " ".join(current_words)
        chunks.append(Chunk(text=chunk_text, speaker=speaker,
                                chunk_idx=idx, metadata=meta.copy()))

    return chunks


def chunk_conversation(messages: list[dict], conv_id: str = "") -> list[Chunk]:
    """Chunk an entire conversation (list of {role, content} dicts)."""
    all_chunks = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("user", "assistant") and content:
            chunks = chunk_message(content, speaker=role, conv_id=conv_id)
            all_chunks.extend(chunks)
    return all_chunks
