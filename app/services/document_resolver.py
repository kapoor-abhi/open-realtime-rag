#document_resolver.py
"""
Document name resolver.

When a user says "what does page 5 of the annual report say?" or
"compare page 3 of contract_A.pdf with contract_B", the intent node
extracts a hint string like "annual report" or "contract_A".

This module resolves that hint to an exact file_hash by fuzzy-matching
against the active_documents list (which contains both hash and filename).

Matching strategy (in priority order):
  1. Exact filename match (case-insensitive)
  2. Filename-without-extension match
  3. Subsequence / contains match (hint is substring of filename)
  4. difflib SequenceMatcher ratio >= 0.6
  5. No match → return None (retrieval falls back to all active docs)
"""

import difflib
import logging
import os
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


def resolve_document(
    hint: Optional[str],
    active_documents: List[dict],  # List of {file_hash, filename}
    threshold: float = 0.55,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Match a natural-language document hint to one of the active documents.

    Args:
        hint: The string the LLM extracted, e.g. "annual report" or "Q3.pdf"
        active_documents: List of dicts with keys 'file_hash' and 'filename'
        threshold: Minimum SequenceMatcher ratio to accept a fuzzy match

    Returns:
        (file_hash, filename) of the best match, or (None, None) if not found.
    """
    if not hint or not active_documents:
        return None, None

    hint_clean = hint.strip().lower()
    hint_no_ext = os.path.splitext(hint_clean)[0]

    best_score = 0.0
    best_doc = None

    for doc in active_documents:
        filename = doc.get("filename", "")
        filename_lower = filename.lower()
        filename_no_ext = os.path.splitext(filename_lower)[0]

        # ---- 1. Exact match ----
        if hint_clean == filename_lower or hint_no_ext == filename_no_ext:
            logger.info(
                f"[DOC RESOLVER] Exact match: hint='{hint}' → '{filename}'"
            )
            return doc["file_hash"], filename

        # ---- 2. Contains match ----
        if hint_clean in filename_lower or hint_no_ext in filename_no_ext:
            score = 0.9  # high but not 1.0 so exact can beat it
        elif filename_no_ext in hint_no_ext:
            score = 0.8
        else:
            # ---- 3. Fuzzy ratio ----
            score = difflib.SequenceMatcher(
                None, hint_no_ext, filename_no_ext
            ).ratio()

        if score > best_score:
            best_score = score
            best_doc = doc

    if best_doc and best_score >= threshold:
        logger.info(
            f"[DOC RESOLVER] Fuzzy match (score={best_score:.2f}): "
            f"hint='{hint}' → '{best_doc['filename']}'"
        )
        return best_doc["file_hash"], best_doc["filename"]

    logger.info(
        f"[DOC RESOLVER] No match for hint='{hint}' "
        f"(best_score={best_score:.2f} < threshold={threshold}). "
        f"Will search across all active documents."
    )
    return None, None
