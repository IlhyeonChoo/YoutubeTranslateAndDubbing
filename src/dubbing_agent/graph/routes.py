"""LangGraph 워크플로우 라우팅 함수."""

from __future__ import annotations

from typing import Literal

from dubbing_agent.state import DubbingState


def route_has_segments(
    state: DubbingState,
) -> Literal["translate_segment", "final_quality_gate"]:
    """원본 세그먼트가 있을 때만 세그먼트 처리 루프로 보낸다."""
    if state.get("segments"):
        return "translate_segment"
    return "final_quality_gate"


def route_source_text(
    state: DubbingState,
) -> Literal["load_source_subtitle", "extract_audio"]:
    """agent가 기존 자막 사용을 권장하면 자막 로드 경로로 보낸다."""
    decision = state.get("metadata", {}).get("subtitle_inspection", {})
    if (
        decision.get("action") == "use_subtitle"
        and decision.get("selected_language")
        and decision.get("subtitle_source") in {"manual", "automatic"}
    ):
        return "load_source_subtitle"
    return "extract_audio"


def route_loaded_source_subtitle(
    state: DubbingState,
) -> Literal["build_context", "extract_audio"]:
    """자막 세그먼트 로드에 성공했으면 STT를 건너뛴다."""
    if state.get("segments"):
        return "build_context"
    return "extract_audio"


def route_translation_qc(
    state: DubbingState,
) -> Literal["translate_segment", "duration_adjust"]:
    """QC가 실패했고 재시도 횟수가 남아 있으면 번역을 다시 시도한다."""
    segment = _current_segment(state)
    if segment.get("qc_passed", False):
        return "duration_adjust"

    attempts = _translation_attempt_count(segment)
    max_attempts = state.get("max_translation_attempts", 3)
    if attempts < max_attempts:
        return "translate_segment"

    return "duration_adjust"


def route_duration_adjust(
    state: DubbingState,
) -> Literal["translate_segment", "tts", "next_segment"]:
    """길이 재작성 결과가 번역 재시도를 요구하면 TTS 전에 되돌린다."""
    segment = _current_segment(state)
    if segment.get("qc_passed") is not False:
        return "tts"

    attempts = _translation_attempt_count(segment)
    max_attempts = state.get("max_translation_attempts", 3)
    if attempts < max_attempts:
        return "translate_segment"
    return "next_segment"


def route_tts_duration(
    state: DubbingState,
) -> Literal["duration_adjust", "translate_segment", "next_segment"]:
    """TTS가 부적합하면 재작성하거나 번역 단계로 되돌린다."""
    segment = _current_segment(state)
    if segment.get("duration_passed", False):
        return "next_segment"

    translation_attempts = _translation_attempt_count(segment)
    max_translation_attempts = state.get("max_translation_attempts", 3)
    if segment.get("dubbing_action") == "retry_translation":
        if translation_attempts < max_translation_attempts:
            return "translate_segment"
        return "next_segment"

    attempts = segment.get("duration_rewrite_attempts", 0)
    max_attempts = state.get("max_duration_rewrite_attempts", 3)
    if state.get("duration_rewrite_enabled", True) and attempts < max_attempts:
        return "duration_adjust"

    return "next_segment"


def route_next_segment(
    state: DubbingState,
) -> Literal["translate_segment", "translation_qc", "global_translation_review"]:
    """세그먼트 루프를 계속 진행하거나 출력 생성 단계로 이동한다."""
    if state.get("global_translation_reprocess_segment_ids"):
        return "translation_qc"
    if state.get("current_segment_index", 0) < len(state.get("segments", [])):
        return "translate_segment"
    return "global_translation_review"


def route_global_translation_review(
    state: DubbingState,
) -> Literal["translation_qc", "final_quality_gate"]:
    """전체 번역 리뷰가 수정 제안을 적용했으면 해당 세그먼트 QC부터 재처리한다."""
    review = state.get("metadata", {}).get("global_translation_review", {})
    if (
        review.get("action") == "reprocess"
        and state.get("global_translation_reprocess_segment_ids")
    ):
        return "translation_qc"
    return "final_quality_gate"


def _current_segment(state: DubbingState):
    """라우팅 판단에 사용할 현재 세그먼트를 반환한다."""
    idx = state.get("current_segment_index", 0)
    return state["segments"][idx]


def _translation_attempt_count(segment) -> int:
    """Return the highest known translation attempt count for routing."""
    return max(
        int(segment.get("translation_attempts", 0) or 0),
        int(segment.get("total_translation_attempts", 0) or 0),
    )
