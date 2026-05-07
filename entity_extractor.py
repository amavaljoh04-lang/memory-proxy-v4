"""Extract structured entities from text.

Each entity = a micro-chunk stored separately in Qdrant for precise recall.
Entities include: colors, money, organizations, persons, IPs, versions,
variables, paths, URLs, metrics, functions, classes, commands.
"""
import re
from dataclasses import dataclass, field


@dataclass
class Entity:
    entity_type: str      # "color", "money", "organization", etc.
    value: str            # "#00CED1", "8.3 millions d'euros"
    context: str          # ~50 tokens of surrounding text
    importance: float     # 0.0-1.0 for ranking

    def to_chunk_text(self) -> str:
        """Format for storage — value + context for semantic search."""
        return f"[{self.entity_type}: {self.value}] {self.context}"


# ─── Extraction patterns ───

PATTERNS: list[tuple[str, str, float]] = [
    # (regex, entity_type, importance)
    
    # Colors (hex)
    (r'#[0-9A-Fa-f]{6}\b', "color", 0.80),
    (r'#[0-9A-Fa-f]{3}\b', "color", 0.70),
    (r'rgb\(\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}\s*\)', "color", 0.75),
    (r'rgba\(\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*[\d.]+\s*\)', "color", 0.75),
    
    # Money / budget
    (r'\d[\d\s,.]*\s*(?:millions?|milliards?)\s*(?:d[\'e]\s*euros?|€|\$|dollars?)?', "money", 0.90),
    (r'\d[\d\s,.]*\s*(?:€|euros?|\$|dollars?|USD|EUR|CHF|GBP)', "money", 0.90),
    (r'(?:budget|coût|prix|financement|investissement|levée)\s*(?:de\s+)?\d[\d\s,.]*\s*\w+', "money", 0.85),
    
    # Organizations (after "par", "chez", "de", "avec", + capitalized)
    (r'(?:financ[ée]s?\s+par|par|chez|avec|de)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,4})', "organization", 0.85),
    
    # Persons (capitalized names, 2+ words)
    (r'\b([A-Z][a-zéèêëàâäùûüôöïîç]+\s+[A-Z][a-zéèêëàâäùûüôöïîç]+(?:\s+[A-Z][a-zéèêëàâäùûüôöïîç]+)?)\b', "person", 0.90),
    
    # IPs
    (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?\b', "ip", 0.85),
    
    # Constants (UPPER_CASE = value) — high importance, exact value preserved
    (r'\b[A-Z][A-Z0-9_]{2,}\s*=\s*(?:"[^"]*"|\'[^\']*\'|\d[\d.eE+\-]*|\[[^\]]*\]|\{[^}]*\}|True|False|None)[^\n]{0,30}', "constant", 0.95),
    # Fallback: any UPPER_CASE = something
    (r'\b[A-Z][A-Z0-9_]{2,}\s*=\s*[^\n]{1,80}', "constant", 0.90),
    
    # Variables / parameters (key=value)
    (r'\b[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*[\d.e\-+]+(?:\s*[A-Z]+)?', "variable", 0.80),
    
    # Versions
    (r'\bv?\d+\.\d+(?:\.\d+)?(?:-[a-zA-Z0-9]+)?\b', "version", 0.75),
    
    # URLs
    (r'https?://[^\s<>"\']+', "url", 0.80),
    
    # Paths
    (r'(?:/[a-zA-Z0-9_.~-]+){2,}', "path", 0.75),
    
    # Metrics (nDCG@10=0.547, loss=0.019, score: 0.95)
    (r'\b[a-zA-Z_@]+\s*[=:]\s*\d+\.?\d*(?:\s*%)?', "metric", 0.80),
    
    # Functions
    (r'\bdef\s+([a-zA-Z_][a-zA-Z0-9_]*)', "function", 0.75),
    (r'\bfunction\s+([a-zA-Z_][a-zA-Z0-9_]*)', "function", 0.75),
    
    # Classes
    (r'\bclass\s+([A-Z][a-zA-Z0-9_]*)', "class_def", 0.75),
    
    # Commands (shell)
    (r'(?:pip|npm|apt|docker|git|curl|wget|ssh|scp)\s+[^\n]{5,60}', "command", 0.70),
    
    # Model names (common patterns)
    (r'\b(?:GPT-?[0-9.]+|BERT|MiniLM|BAAI/[a-zA-Z0-9-]+|[a-z]+/[a-z0-9_.-]+:[a-z0-9_.-]+)\b', "model_name", 0.80),
    
    # Dates
    (r'\b\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4}\b', "date", 0.75),
    (r'\b\d{4}-\d{2}-\d{2}\b', "date", 0.75),
]

# Words that should NOT be treated as person names
NOT_PERSON = {
    "je suis", "je vis", "il est", "elle est", "nous avons",
    "il faut", "on parle", "je vais", "tu veux",
    # Common false positive beginnings of sentences
    "le projet", "la solution", "le modèle", "le système",
    "en belgique", "en france", "en europe",
}


def extract_entities(text: str, max_context_words: int = 50) -> list[Entity]:
    """Extract all entities from text with surrounding context."""
    entities = []
    seen_values = set()
    words = text.split()
    
    for pattern, entity_type, importance in PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE if entity_type == "money" else 0):
            value = m.group(0).strip()
            
            # For organization/function/class patterns that capture a group
            if entity_type in ("organization", "function", "class_def") and m.lastindex:
                value = m.group(1).strip()
            
            # Skip duplicates
            norm_val = value.lower()
            if norm_val in seen_values:
                continue
            
            # Skip false positive persons
            if entity_type == "person":
                if any(fp in value.lower() for fp in NOT_PERSON):
                    continue
                # Skip if it's just 2 short words
                parts = value.split()
                if len(parts) < 2 or all(len(p) <= 2 for p in parts):
                    continue
            
            seen_values.add(norm_val)
            
            # Get surrounding context
            start = m.start()
            end = m.end()
            context = _get_context(text, start, end, max_context_words)
            
            entities.append(Entity(
                entity_type=entity_type,
                value=value,
                context=context,
                importance=importance
            ))
    
    return entities


def _get_context(text: str, match_start: int, match_end: int, max_words: int = 50) -> str:
    """Get surrounding context around a match position."""
    # Get ~25 words before and ~25 words after
    half = max_words // 2
    
    # Find word boundaries before the match
    before_text = text[:match_start]
    before_words = before_text.split()
    before = " ".join(before_words[-half:]) if len(before_words) > half else before_text.strip()
    
    # Find word boundaries after the match  
    after_text = text[match_end:]
    after_words = after_text.split()
    after = " ".join(after_words[:half]) if len(after_words) > half else after_text.strip()
    
    # Combine: context before + match + context after
    matched = text[match_start:match_end]
    result = f"{before} {matched} {after}".strip()
    return result


def extract_money_and_funder(text: str) -> list[Entity]:
    """Special handling: extract money AND funder from same phrase.
    
    Example: "8.3 millions financé par Nordic Ventures" 
    → [money: 8.3 millions], [organization: Nordic Ventures]
    """
    entities = []
    
    # Pattern: amount + "financé par" + organization
    pattern = r'(\d[\d\s,.]*\s*(?:millions?|milliards?|k|€|euros?|\$))\s+(?:financ[ée]s?\s+par|par)\s+([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})'
    
    for m in re.finditer(pattern, text):
        amount = m.group(1).strip()
        org = m.group(2).strip()
        context = _get_context(text, m.start(), m.end(), 50)
        
        entities.append(Entity(
            entity_type="money",
            value=amount,
            context=context,
            importance=0.90
        ))
        entities.append(Entity(
            entity_type="organization",
            value=org,
            context=context,
            importance=0.85
        ))
    
    return entities
