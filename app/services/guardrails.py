#guardrails.py
"""
Security guardrails for the OpenMultiRAG API.

FIX applied to validate_citations():
  Previously checked whether str(page_number) appeared anywhere in the
  response text. Since page_number=1 means searching for the string "1",
  and virtually every sentence contains the digit "1", this produced a
  citation for page 1 on almost every single response — regardless of
  whether the LLM actually referenced page 1.

  Fixed by matching the more specific pattern "page {N}" or "Page {N}"
  (case-insensitive), which is the format the LLM uses in its Sources section.
"""

import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. FILE VALIDATION
# ---------------------------------------------------------------------------

_MAGIC = {
    b"%PDF": "application/pdf",
}

MAX_FILE_SIZE_DEFAULT = 50 * 1024 * 1024  # 50 MB


def validate_file(content: bytes, filename: str, max_bytes: int = MAX_FILE_SIZE_DEFAULT) -> Tuple[bool, str]:
    if len(content) > max_bytes:
        mb = max_bytes // (1024 * 1024)
        return False, f"File exceeds maximum allowed size of {mb} MB."

    allowed_exts = {".pdf"}
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in allowed_exts:
        return False, f"File type '{ext}' is not allowed. Only PDF files are accepted."

    header = content[:4]
    matched = any(header.startswith(magic) for magic in _MAGIC)
    if not matched:
        return False, "File content does not match a valid PDF signature."

    return True, ""


# ---------------------------------------------------------------------------
# 2. INPUT SANITIZATION
# ---------------------------------------------------------------------------

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

MAX_QUERY_LENGTH = 2_000


def sanitize_query(text: str) -> str:
    cleaned = _CONTROL_CHARS.sub("", text)
    return cleaned[:MAX_QUERY_LENGTH].strip()


# ---------------------------------------------------------------------------
# 3. PROMPT INJECTION DETECTION
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions?",
    r"forget\s+(your|all)\s+(previous\s+)?instructions?",
    r"disregard\s+(all\s+)?previous",
    r"new\s+instructions?:",
    r"you\s+are\s+now\s+(a|an|the)\s+\w+",
    r"act\s+as\s+(a|an|the)\s+\w+",
    r"pretend\s+(you\s+are|to\s+be)",
    r"roleplay\s+as",
    r"do\s+anything\s+now",
    r"\bDAN\b",
    r"jailbreak",
    r"system\s*prompt",
    r"override\s+(the\s+)?instructions?",
    r"reveal\s+(your\s+)?(system\s+)?prompt",
    r"print\s+(your\s+)?(system\s+)?prompt",
    r"leak\s+(your|the)\s+(instructions?|prompt)",
    r"<!--.*?-->",
    r"<\s*script",
    r"\{\{.*?\}\}",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _INJECTION_PATTERNS]


def detect_prompt_injection(text: str) -> Tuple[bool, str]:
    for pattern in _COMPILED_PATTERNS:
        match = pattern.search(text)
        if match:
            logger.warning(
                f"[GUARDRAILS] Prompt injection detected. "
                f"Pattern='{pattern.pattern}' | Input='{text[:100]}'"
            )
            return True, pattern.pattern
    return False, ""


# ---------------------------------------------------------------------------
# 4. PII SCRUBBING
# ---------------------------------------------------------------------------

_PII_PATTERNS = [
    (re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"), "[EMAIL]"),
    (re.compile(r"(\+?\d[\d\s\-\.\(\)]{6,14}\d)"), "[PHONE]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[CARD]"),
    (re.compile(r"\b[A-Za-z0-9_\-]{32,}\b"), "[TOKEN]"),
]


def scrub_pii(text: str) -> str:
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# 5. OUTPUT VALIDATION
# ---------------------------------------------------------------------------

# FIX: Match "page N" / "Page N" instead of bare str(page_number).
# The old check `page_num in response_text` with page_num="1" matched
# every response (the digit "1" appears in virtually any sentence),
# producing spurious citations for page 1 on every answer.
_PAGE_PATTERN = re.compile(r"\bpage\s+(\d+)\b", re.IGNORECASE)


def validate_citations(response_text: str, retrieved_chunks: list) -> list:
    """
    Filter citations: only keep those whose source_file AND page reference
    ("page N") both appear verbatim in the LLM response text.

    This prevents hallucinated citations from reaching the user.
    """
    from app.models.schemas import SourceCitation

    # Collect all page numbers the LLM actually mentioned
    mentioned_pages = {int(m.group(1)) for m in _PAGE_PATTERN.finditer(response_text)}

    valid = {}
    for chunk in retrieved_chunks:
        file_name = chunk.get("source_file", "")
        page_num = chunk.get("page_number")

        if file_name in response_text and page_num in mentioned_pages:
            key = f"{file_name}_{page_num}"
            if key not in valid:
                valid[key] = SourceCitation(
                    page_number=page_num,
                    source_file=file_name,
                    image_path=chunk.get("image_path"),
                )

    return list(valid.values())