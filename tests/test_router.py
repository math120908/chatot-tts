"""Tests for chatot.router module."""

from __future__ import annotations

from chatot.router import RoutingDecision, detect_scripts, route


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

VOICES: dict[str, str] = {
    "en": "en-US-AvaNeural",
    "zh": "zh-TW-HsiaoChenNeural",
    "ko": "ko-KR-SunHiNeural",
    "mixed": "Ryan",
}

PRIMARY = "edge"
SECONDARY = "qwen3"


# ---------------------------------------------------------------------------
# detect_scripts
# ---------------------------------------------------------------------------


class TestDetectScripts:
    """Test detect_scripts character classification."""

    def test_english_only(self) -> None:
        assert detect_scripts("Hello world") == {"en"}

    def test_chinese_only(self) -> None:
        assert detect_scripts("你好世界") == {"zh"}

    def test_korean_only(self) -> None:
        assert detect_scripts("안녕하세요") == {"ko"}

    def test_mixed_zh_en(self) -> None:
        assert detect_scripts("Hello 你好") == {"zh", "en"}

    def test_mixed_zh_ko_en(self) -> None:
        assert detect_scripts("Hello 你好 안녕") == {"zh", "en", "ko"}

    def test_mixed_ko_en(self) -> None:
        assert detect_scripts("Hello 안녕") == {"ko", "en"}

    def test_numbers_and_punctuation_only(self) -> None:
        assert detect_scripts("123!@# .,") == set()

    def test_empty_string(self) -> None:
        assert detect_scripts("") == set()

    def test_chinese_with_punctuation(self) -> None:
        assert detect_scripts("你好，世界！") == {"zh"}

    def test_english_with_digits(self) -> None:
        assert detect_scripts("abc 123") == {"en"}

    def test_hangul_jamo(self) -> None:
        # U+1100 is in the Hangul Jamo block
        assert detect_scripts("\u1100") == {"ko"}

    def test_cjk_extension_b(self) -> None:
        # U+20000 is the first character in CJK Extension B
        assert detect_scripts("\U00020000") == {"zh"}


# ---------------------------------------------------------------------------
# route
# ---------------------------------------------------------------------------


class TestRoute:
    """Test route() backend/voice/lang decisions."""

    def test_english_text_uses_primary_en_voice(self) -> None:
        decision = route("Hello world", None, VOICES, PRIMARY, SECONDARY)
        assert decision == RoutingDecision(
            backend=PRIMARY,
            voice=VOICES["en"],
            lang_code="en",
        )

    def test_chinese_text_uses_primary_zh_voice(self) -> None:
        decision = route("你好世界", None, VOICES, PRIMARY, SECONDARY)
        assert decision == RoutingDecision(
            backend=PRIMARY,
            voice=VOICES["zh"],
            lang_code="zh",
        )

    def test_korean_text_uses_primary_ko_voice(self) -> None:
        decision = route("안녕하세요", None, VOICES, PRIMARY, SECONDARY)
        assert decision == RoutingDecision(
            backend=PRIMARY,
            voice=VOICES["ko"],
            lang_code="ko",
        )

    def test_mixed_text_uses_secondary_mixed_voice(self) -> None:
        decision = route("Hello 你好", None, VOICES, PRIMARY, SECONDARY)
        assert decision == RoutingDecision(
            backend=SECONDARY,
            voice=VOICES["mixed"],
            lang_code="auto",
        )

    def test_lang_override_zh_uses_primary_zh_voice(self) -> None:
        decision = route("some text", "zh", VOICES, PRIMARY, SECONDARY)
        assert decision == RoutingDecision(
            backend=PRIMARY,
            voice=VOICES["zh"],
            lang_code="zh",
        )

    def test_lang_override_auto_uses_secondary_mixed_voice(self) -> None:
        decision = route("some text", "auto", VOICES, PRIMARY, SECONDARY)
        assert decision == RoutingDecision(
            backend=SECONDARY,
            voice=VOICES["mixed"],
            lang_code="auto",
        )

    def test_lang_override_mixed_uses_secondary_mixed_voice(self) -> None:
        decision = route("some text", "mixed", VOICES, PRIMARY, SECONDARY)
        assert decision == RoutingDecision(
            backend=SECONDARY,
            voice=VOICES["mixed"],
            lang_code="auto",
        )

    def test_punctuation_only_falls_back_to_primary_default(self) -> None:
        decision = route("123!.,", None, VOICES, PRIMARY, SECONDARY)
        assert decision.backend == PRIMARY
        assert decision.voice == VOICES["en"]  # default_voice = voices["en"]

    def test_unknown_lang_override_uses_primary_default_voice(self) -> None:
        decision = route("some text", "ja", VOICES, PRIMARY, SECONDARY)
        assert decision == RoutingDecision(
            backend=PRIMARY,
            voice=VOICES["en"],  # "ja" not in VOICES → default_voice
            lang_code="ja",
        )

    def test_mixed_text_no_secondary_falls_back_to_primary(self) -> None:
        decision = route("Hello 你好", None, VOICES, PRIMARY, secondary=None)
        assert decision.backend == PRIMARY
        assert decision.lang_code == "en"

    def test_lang_override_auto_no_secondary_uses_primary(self) -> None:
        decision = route("some text", "auto", VOICES, PRIMARY, secondary=None)
        assert decision.backend == PRIMARY
        assert decision.lang_code == "auto"

    # --- voice_override routing ----------------------------------------

    def test_bare_voice_routes_to_secondary(self) -> None:
        decision = route("Hello world", None, VOICES, PRIMARY, SECONDARY, voice_override="vivian")
        assert decision == RoutingDecision(
            backend=SECONDARY,
            voice="vivian",
            lang_code="en",
        )

    def test_edge_voice_routes_to_primary(self) -> None:
        decision = route("Hello world", None, VOICES, PRIMARY, SECONDARY, voice_override="en-US-BrianNeural")
        assert decision == RoutingDecision(
            backend=PRIMARY,
            voice="en-US-BrianNeural",
            lang_code="en",
        )

    def test_bare_voice_no_secondary_falls_back_to_primary(self) -> None:
        decision = route("Hello world", None, VOICES, PRIMARY, secondary=None, voice_override="vivian")
        assert decision == RoutingDecision(
            backend=PRIMARY,
            voice="vivian",
            lang_code="en",
        )

    def test_bare_voice_preserves_lang_from_text(self) -> None:
        decision = route("你好世界", None, VOICES, PRIMARY, SECONDARY, voice_override="uncle_fu")
        assert decision == RoutingDecision(
            backend=SECONDARY,
            voice="uncle_fu",
            lang_code="zh",
        )

    def test_bare_voice_with_lang_override(self) -> None:
        decision = route("Hello", "zh", VOICES, PRIMARY, SECONDARY, voice_override="vivian")
        assert decision == RoutingDecision(
            backend=SECONDARY,
            voice="vivian",
            lang_code="zh",
        )

    def test_voice_override_mixed_text_lang_auto(self) -> None:
        decision = route("Hello 你好", None, VOICES, PRIMARY, SECONDARY, voice_override="ryan")
        assert decision == RoutingDecision(
            backend=SECONDARY,
            voice="ryan",
            lang_code="auto",
        )
