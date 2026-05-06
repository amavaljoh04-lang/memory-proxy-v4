"""Simple, robust text chunker for conversations.

Strategy:
1. Split by sentences (or code blocks)
2. Group into chunks of ~100 tokens with 20 token overlap
3. Each chunk gets stored with metadata (speaker, timestamp, conv_id)
"""
import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    speaker: str  # "user" or "assistant"
    chunk_idx: int
    metadata: dict


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Handles code blocks specially."""
    # Protect code blocks
    code_blocks = []
    def replace_code(m):
        code_blocks.append(m.group(0))
        return f"__CODE_BLOCK_{len(code_blocks)-1}__"

    text = re.sub(r'```[\s\S]*?```', replace_code, text)
    text = re.sub(r'`[^`]+`', replace_code, text)

    # Split on sentence boundaries
    parts = re.split(r'(?<=[.!?])\s+|(?<=\n)\s*(?=\S)', text)

    # Restore code blocks
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

    Args:
        text: The message text
        speaker: "user" or "assistant"
        max_tokens: Approx max tokens per chunk (using word count as proxy)
        overlap: Number of tokens overlap between chunks
        conv_id: Conversation ID
        metadata: Additional metadata
    """
    if not text or not text.strip():
        return []

    meta = metadata or {}
    meta["conv_id"] = conv_id
    meta["speaker"] = speaker

    sentences = split_sentences(text)
    if not sentences:
        return []

    # If text is short enough, single chunk
    word_count = len(text.split())
    if word_count <= max_tokens:
        return [Chunk(text=text.strip(), speaker=speaker, chunk_idx=0, metadata=meta)]

    # Group sentences into chunks
    chunks = []
    current_words = []
    current_count = 0

    for sentence in sentences:
        s_words = sentence.split()
        s_count = len(s_words)

        if current_count + s_count > max_tokens and current_words:
            # Emit chunk
            chunk_text = " ".join(current_words)
            chunks.append(Chunk(
                text=chunk_text,
                speaker=speaker,
                chunk_idx=len(chunks),
                metadata=meta.copy()
            ))
            # Keep overlap
            overlap_words = current_words[-overlap:] if overlap > 0 else []
            current_words = overlap_words + s_words
            current_count = len(current_words)
        else:
            current_words.extend(s_words)
            current_count += s_count

    # Final chunk
    if current_words:
        chunk_text = " ".join(current_words)
        chunks.append(Chunk(
            text=chunk_text,
            speaker=speaker,
            chunk_idx=len(chunks),
            metadata=meta.copy()
        ))

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
