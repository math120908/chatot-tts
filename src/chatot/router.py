"""Language detection and TTS backend routing for multilingual text."""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Script detection
# ---------------------------------------------------------------------------

def detect_scripts(text: str) -> set[str]:
    """Detect which writing scripts are present in *text*.

    Returns a set of script tags:
    - "zh" — CJK Unified Ideographs (and extensions / compatibility blocks)
    - "ko" — Hangul Syllables and Jamo
    - "en" — Basic Latin letters and Latin Extended (a–z, A–Z, U+00C0–U+024F)

    Punctuation, digits, and whitespace are ignored and contribute nothing.
    """
    scripts: set[str] = set()
    for ch in text:
        cp = ord(ch)
        if _is_cjk(cp):
            scripts.add("zh")
        elif _is_hangul(cp):
            scripts.add("ko")
        elif _is_latin(cp):
            scripts.add("en")
    return scripts


def _is_cjk(cp: int) -> bool:
    return (
        0x4E00 <= cp <= 0x9FFF    # CJK Unified Ideographs
        or 0x3400 <= cp <= 0x4DBF  # CJK Extension A
        or 0x20000 <= cp <= 0x2A6DF  # CJK Extension B
        or 0x2A700 <= cp <= 0x2B73F  # CJK Extension C
        or 0x2B740 <= cp <= 0x2B81F  # CJK Extension D
        or 0xF900 <= cp <= 0xFAFF    # CJK Compatibility Ideographs
        or 0x2F800 <= cp <= 0x2FA1F  # CJK Compatibility Ideographs Supplement
    )


def _is_hangul(cp: int) -> bool:
    return (
        0xAC00 <= cp <= 0xD7AF    # Hangul Syllables
        or 0x1100 <= cp <= 0x11FF  # Hangul Jamo
        or 0x3130 <= cp <= 0x318F  # Hangul Compatibility Jamo
        or 0xA960 <= cp <= 0xA97F  # Hangul Jamo Extended-A
        or 0xD7B0 <= cp <= 0xD7FF  # Hangul Jamo Extended-B
    )


def _is_latin(cp: int) -> bool:
    return (
        0x41 <= cp <= 0x5A        # A–Z
        or 0x61 <= cp <= 0x7A     # a–z
        or 0xC0 <= cp <= 0x24F    # Latin Extended
    )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RoutingDecision:
    backend: str    # "edge" or "qwen3"
    voice: str      # voice ID
    lang_code: str  # language code


def route(
    text: str,
    lang_override: str | None,
    voices: dict[str, str],
    primary: str,
    secondary: str | None,
) -> RoutingDecision:
    """Determine which backend, voice, and language code to use.

    Priority order:
    1. lang_override == "auto" or "mixed"  → secondary backend (or primary
       fallback), mixed voice, lang="auto".
    2. lang_override is a specific tag (e.g. "zh")  → primary backend,
       voices[lang_override] (or default voice if unknown), lang=lang_override.
    3. No override  → detect scripts in text:
       - 1 script   → primary + corresponding voice + that lang
       - 2+ scripts + secondary available → secondary + mixed voice + "auto"
       - 0 scripts or mixed without secondary → primary + default voice + "en"
    """
    default_voice = voices.get("en", next(iter(voices.values())))

    # --- Case 1: explicit "auto" / "mixed" override --------------------
    if lang_override in ("auto", "mixed"):
        if secondary is not None:
            return RoutingDecision(
                backend=secondary,
                voice=voices.get("mixed", default_voice),
                lang_code="auto",
            )
        # No secondary — fall through to primary with mixed voice
        return RoutingDecision(
            backend=primary,
            voice=voices.get("mixed", default_voice),
            lang_code="auto",
        )

    # --- Case 2: specific language override ----------------------------
    if lang_override is not None:
        return RoutingDecision(
            backend=primary,
            voice=voices.get(lang_override, default_voice),
            lang_code=lang_override,
        )

    # --- Case 3: auto-detect -------------------------------------------
    scripts = detect_scripts(text)

    if len(scripts) == 1:
        lang = next(iter(scripts))
        return RoutingDecision(
            backend=primary,
            voice=voices.get(lang, default_voice),
            lang_code=lang,
        )

    if len(scripts) >= 2 and secondary is not None:
        return RoutingDecision(
            backend=secondary,
            voice=voices.get("mixed", default_voice),
            lang_code="auto",
        )

    # 0 scripts (punctuation-only) or mixed without secondary
    return RoutingDecision(
        backend=primary,
        voice=default_voice,
        lang_code="en",
    )
