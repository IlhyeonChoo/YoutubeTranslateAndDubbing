"""LangChain agent가 호출하는 품질 판단 보조 도구."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from langchain.tools import tool
from yt_dlp import YoutubeDL


@tool
def inspect_subtitle_availability(
    source_url: str,
    preferred_language: str | None = None,
) -> dict[str, Any]:
    """YouTube 영상 metadata에서 수동 자막과 자동 자막 제공 여부를 확인한다."""
    try:
        with YoutubeDL({"quiet": True, "skip_download": True, "noplaylist": True}) as ydl:
            info = ydl.extract_info(source_url, download=False)
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": str(exc),
            "has_manual_subtitles": False,
            "has_auto_captions": False,
            "manual_languages": [],
            "automatic_languages": [],
            "preferred_language": preferred_language or "",
            "preferred_language_available": False,
        }

    manual_languages = sorted((info.get("subtitles") or {}).keys())
    automatic_languages = sorted((info.get("automatic_captions") or {}).keys())
    preferred = preferred_language or ""
    return {
        "ok": True,
        "title": info.get("title", ""),
        "duration": info.get("duration"),
        "has_manual_subtitles": bool(manual_languages),
        "has_auto_captions": bool(automatic_languages),
        "manual_languages": manual_languages,
        "automatic_languages": automatic_languages,
        "preferred_language": preferred,
        "preferred_language_available": preferred in manual_languages
        or preferred in automatic_languages,
    }


@tool
def analyze_translation_quality(
    source_text: str,
    translated_text: str,
    duration_seconds: float,
    target_language: str,
) -> dict[str, Any]:
    """번역 후보의 기본 품질, 원문 복사 가능성, 발화 길이 부담을 계산한다."""
    source = source_text.strip()
    translated = translated_text.strip()
    source_units = _speech_units(source)
    translated_units = _speech_units(translated)
    overlap_ratio = _similarity(source.lower(), translated.lower())
    length_ratio = translated_units / max(1.0, source_units)
    target_units_per_second = translated_units / max(0.1, duration_seconds)

    issues: list[str] = []
    if not translated:
        issues.append("번역문이 비어 있습니다.")
    if source and translated and overlap_ratio >= 0.82:
        issues.append("번역문이 원문을 거의 그대로 복사한 것으로 보입니다.")
    if source_units >= 8.0 and length_ratio <= 0.2:
        issues.append("번역문이 원문에 비해 지나치게 짧아 의미가 누락됐을 수 있습니다.")
    if length_ratio >= 2.4:
        issues.append("번역문이 원문에 비해 지나치게 깁니다.")
    if target_units_per_second >= _max_units_per_second(target_language):
        issues.append("현재 세그먼트 길이에 비해 발화량이 많습니다.")

    heuristic_score = 1.0
    heuristic_score -= 0.35 if not translated else 0.0
    heuristic_score -= 0.25 if overlap_ratio >= 0.82 else 0.0
    heuristic_score -= min(0.35, max(0.0, 0.24 - length_ratio) * 1.5)
    heuristic_score -= min(0.25, max(0.0, length_ratio - 1.8) * 0.15)
    heuristic_score -= min(
        0.25,
        max(0.0, target_units_per_second - _max_units_per_second(target_language))
        * 0.08,
    )
    heuristic_score = max(0.0, min(1.0, heuristic_score))

    return {
        "source_units": source_units,
        "translated_units": translated_units,
        "overlap_ratio": round(overlap_ratio, 3),
        "length_ratio": round(length_ratio, 3),
        "target_units_per_second": round(target_units_per_second, 3),
        "heuristic_score": round(heuristic_score, 3),
        "heuristic_passed": heuristic_score >= 0.72 and not issues,
        "issues": issues,
    }


@tool
def analyze_dubbing_quality(
    source_seconds: float,
    tts_seconds: float,
    tolerance: float,
    text: str = "",
    target_language: str = "",
) -> dict[str, Any]:
    """생성된 TTS 길이와 발화 속도가 원본 시간에 적합한지 계산한다."""
    source = max(0.1, float(source_seconds))
    tts = max(0.0, float(tts_seconds))
    allowed_max = source * (1.0 + max(0.0, float(tolerance)))
    ratio = tts / source
    exceeded_seconds = max(0.0, tts - allowed_max)
    passed = exceeded_seconds <= 0.0
    speech_units = _speech_units(text)
    actual_units_per_second = speech_units / max(0.1, tts)
    required_units_per_second = speech_units / max(0.1, allowed_max)
    units_per_second = max(actual_units_per_second, required_units_per_second)
    max_units_per_second = _max_units_per_second(target_language)
    speed_passed = units_per_second <= max_units_per_second

    issues: list[str] = []
    if not passed:
        issues.append(
            f"TTS가 허용 길이를 {exceeded_seconds:.2f}초 초과했습니다."
        )
    if tts <= source * 0.45:
        issues.append("TTS가 원본 구간에 비해 지나치게 짧을 수 있습니다.")
    if text and actual_units_per_second > max_units_per_second:
        issues.append("생성된 TTS의 실제 말하기 속도가 지나치게 빠릅니다.")
    if text and required_units_per_second > max_units_per_second:
        issues.append(
            "정해진 문장 길이 안에 맞추려면 말하기 속도가 지나치게 빠릅니다."
        )

    score = 1.0
    if not passed:
        score -= min(0.6, exceeded_seconds / source)
    if tts <= source * 0.45:
        score -= 0.2
    if text and not speed_passed:
        score -= min(0.35, max(0.0, units_per_second - max_units_per_second) * 0.08)
    score = max(0.0, min(1.0, score))

    return {
        "source_seconds": round(source, 3),
        "tts_seconds": round(tts, 3),
        "tolerance": tolerance,
        "allowed_max_seconds": round(allowed_max, 3),
        "ratio": round(ratio, 3),
        "exceeded_seconds": round(exceeded_seconds, 3),
        "speech_units": round(speech_units, 3),
        "units_per_second": round(units_per_second, 3),
        "actual_units_per_second": round(actual_units_per_second, 3),
        "required_units_per_second": round(required_units_per_second, 3),
        "max_units_per_second": round(max_units_per_second, 3),
        "speed_passed": speed_passed,
        "heuristic_passed": passed and speed_passed,
        "heuristic_score": round(score, 3),
        "issues": issues,
        "action_hint": "pass" if passed and speed_passed else "rewrite",
    }


def _similarity(left: str, right: str) -> float:
    """두 문자열의 대략적인 유사도를 계산한다."""
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _speech_units(text: str) -> float:
    """언어와 무관하게 발화량을 비교하기 위한 단순 단위 수를 계산한다."""
    compact = "".join(ch for ch in text if not ch.isspace())
    if not compact:
        return 0.0
    whitespace_units = len(text.split())
    character_units = len(compact) / 2.6
    return float(max(whitespace_units, character_units))


def _max_units_per_second(language: str) -> float:
    """대상 언어별 대략적인 초당 발화 단위 상한을 반환한다."""
    normalized = language.lower()
    if normalized.startswith(("ko", "ja", "zh")):
        return 4.5
    return 3.8
