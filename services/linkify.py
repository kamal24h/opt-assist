# src/rag_service/services/linkify.py
"""Convert bare URLs in text to <a> tags for HTML rendering.

Pre-processing:
  - Normalize URLs that contain internal spaces (e.g. LLM output like
    "https://portal2 .ariansystem.net/" → "https://portal2.ariansystem.net/").
  - This is a best-effort fix for LLM tokenization artifacts.

Usage:
    from services.linkify import linkify_html
    answer = linkify_html(answer)

Security:
    The transformation only replaces bare URLs with <a> tags. It does not
    interpret any other HTML. To avoid XSS when the client renders HTML,
    the upstream text must be trusted (LLM output is not HTML).
"""
from __future__ import annotations

import logging
import re

from linkify_it import LinkifyIt

logger = logging.getLogger(__name__)

_linkify = LinkifyIt()

# Match an http(s) URL that may contain spaces between its parts.
# The pattern is intentionally greedy up to the first whitespace that is
# NOT followed by a domain-like segment (letter/digit).
_URL_WITH_SPACES = re.compile(
    r"https?://[^\s<>\"']+(?:[ \t]+[^\s<>\"']+)*",
    re.IGNORECASE,
)
# A space is "inside a URL" if the next token starts with a letter/digit
# and contains a dot (domain-like).
_URL_SPACE_INTERNAL = re.compile(
    r"[ \t]+(?=[A-Za-z0-9\u0600-\u06FF])",
)


def _normalize_url_spaces(text: str) -> str:
    """Remove internal spaces from http(s) URLs.

    Example:
        "https://portal2 .ariansystem.net/ ادامه" → "https://portal2.ariansystem.net/ ادامه"
        "به https://portal2 .ariansystem.net/ مراجعه کنید" → "به https://portal2.ariansystem.net/ مراجعه کنید"
    """
    def _fix(m: re.Match) -> str:
        raw = m.group(0)
        # Only collapse spaces that are clearly inside a URL:
        # the segment after the space must look like a domain part.
        # We collapse all internal spaces that are followed by a domain-like
        # token (letter/digit + contains a dot or slash).
        def _sub(sm: re.Match) -> str:
            start = sm.end()
            following = raw[start:start + 50]
            if re.match(r"[A-Za-z0-9\u0600-\u06FF][A-Za-z0-9\u0600-\u06FF.-]*[./]", following):
                return ""
            return sm.group(0)
        return _URL_SPACE_INTERNAL.sub(_sub, raw)

    try:
        return _URL_WITH_SPACES.sub(_fix, text)
    except Exception:
        logger.warning("URL space normalization failed; returning original text")
        return text


def linkify_html(text: str) -> str:
    """Convert bare URLs in text to HTML anchors.

    - Normalizes internal spaces inside URLs first.
    - Preserves existing <a> tags (no double-wrapping).
    - Escapes nothing (caller must ensure text is safe or already escaped).
    """
    if not text:
        return text

    text = _normalize_url_spaces(text)

    try:
        matches = _linkify.match(text) or []
    except Exception:
        logger.warning("linkify failed; returning original text")
        return text
    if not matches:
        return text

    # Reverse to avoid index shifting.
    for match in reversed(matches):
        raw = match.raw
        url = match.url
        # Skip if already inside an <a> tag (basic check).
        before = text[: match.index]
        if before.rfind("<a ") > before.rfind("</a>"):
            continue
        replacement = f'<a href="{url}" target="_blank" rel="noopener noreferrer">{raw}</a>'
        text = before + replacement + text[match.last_index:]
    return text