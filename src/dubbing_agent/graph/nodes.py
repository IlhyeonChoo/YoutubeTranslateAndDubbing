"""LangGraph 노드 구현."""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from dubbing_agent.langchain_layer.runtime import build_agent_chains
from dubbing_agent.langchain_layer.tools import (
    analyze_dubbing_quality,
    analyze_translation_quality,
)
from dubbing_agent.services.media_services import (
    compose_dubbed_audio,
    download_youtube_subtitle_segments,
    extract_source_audio,
    get_video_duration,
    mux_dubbed_video,
    prepare_video_from_url,
    synthesize_segment_tts,
    transcribe_and_normalize,
    write_subtitle_file,
)
from dubbing_agent.state import DubbingState, SegmentState
from dubbing_agent.utils.duration import get_audio_duration_seconds
from dubbing_agent.utils.paths import ensure_dir


DEFAULT_RESOURCE_EXHAUSTED_RETRIES = 8
DEFAULT_RESOURCE_EXHAUSTED_DELAY_SECONDS = 60.0
DEFAULT_RESOURCE_EXHAUSTED_MAX_DELAY_SECONDS = 300.0
DEFAULT_OPTIONAL_LLM_RETRIES = 2
DEFAULT_OPTIONAL_LLM_DELAY_SECONDS = 5.0
DEFAULT_OPTIONAL_LLM_MAX_DELAY_SECONDS = 15.0
DEFAULT_TRANSLATION_QC_MIN_SCORE = 0.85
DEFAULT_DUBBING_QC_MIN_SCORE = 0.85
DEFAULT_GLOBAL_TRANSLATION_REVIEW_MIN_SCORE = 0.85
SENTENCE_GAP_SECONDS = 0.9
SENTENCE_MAX_SECONDS = 12.0
SENTENCE_MAX_CHARS = 240
SEGMENT_CONTEXT_RADIUS = 2
GLOBAL_CONTEXT_MAX_CHARS = 12000
REVISION_FEEDBACK_MAX_ITEMS = 8
REVISION_FEEDBACK_ITEM_MAX_CHARS = 360
REVISION_FEEDBACK_TOTAL_MAX_CHARS = 2400
LLM_TRACE_TEXT_MAX_CHARS = 20000
PROGRESS_ENV_VAR = "DUBBING_AGENT_PROGRESS"
CONTEXT_SUMMARY_JSON = "context_summary.json"
CONTEXT_SUMMARY_MARKDOWN = "context_summary.md"
LLM_TRACE_JSONL = "llm_calls.jsonl"
LLM_TRACE_MARKDOWN = "llm_calls.md"
TTS_SLOT_GUARD_SECONDS = 0.15
TTS_ABNORMAL_GAP_GRACE_SECONDS = 0.35
TTS_ABNORMAL_GAP_MIN_SECONDS = 0.55
PROGRESS_NODE_LABELS = {
    "inspect_source": "원본 검사",
    "prepare_video": "영상 준비",
    "load_source_subtitle": "원본 자막 로드",
    "extract_audio": "오디오 추출",
    "stt": "STT",
    "build_context": "문맥 구성",
    "translate_segment": "번역",
    "translation_qc": "번역 QC",
    "duration_adjust": "길이 조정",
    "tts": "TTS",
    "tts_duration_check": "더빙 QC",
    "global_translation_review": "전체 번역 리뷰",
    "final_quality_gate": "최종 품질 게이트",
    "generate_subtitle": "자막 생성",
    "compose_audio": "오디오 합성",
    "write_quality_report": "품질 리포트",
    "mux_video": "영상 합성",
}
PROGRESS_RESULT_LABELS = {
    "pass": "통과",
    "retry": "재시도",
    "retry_translation": "번역 재시도",
    "rewrite": "재작성",
    "continue": "계속",
    "fail": "실패",
}


def _log_progress(
    state: DubbingState,
    node: str,
    message: str = "",
    segment: SegmentState | None = None,
) -> None:
    """사용자가 긴 작업의 현재 노드와 세그먼트를 볼 수 있게 진행 로그를 출력한다."""
    if not _progress_enabled():
        return

    parts = [f"[진행 {time.strftime('%H:%M:%S')}]", f"노드={_progress_node_label(node)}"]
    if segment is not None:
        parts.append(_segment_progress_label(state, segment))
    if message:
        parts.append(f"- {message}")
    print(" ".join(part for part in parts if part), flush=True)


def _progress_enabled() -> bool:
    """환경 변수로 진행 로그를 끌 수 있는지 확인한다."""
    value = os.getenv(PROGRESS_ENV_VAR, "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _progress_node_label(node: str) -> str:
    """진행 로그에 표시할 노드명을 한국어로 변환한다."""
    return PROGRESS_NODE_LABELS.get(node, node)


def _progress_result_label(value: str) -> str:
    """진행 로그에 표시할 결과/action 값을 한국어로 변환한다."""
    return PROGRESS_RESULT_LABELS.get(value, value)


class _LLMTraceCallback(BaseCallbackHandler):
    """Collect prompts that LangChain sends to the chat model."""

    def __init__(self) -> None:
        self.chat_messages: list[list[dict[str, Any]]] = []
        self.text_prompts: list[str] = []

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        **kwargs: Any,
    ) -> None:
        del serialized, kwargs
        for message_group in messages:
            self.chat_messages.append(
                [_message_to_plain(message) for message in message_group]
            )

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        del serialized, kwargs
        self.text_prompts.extend(str(prompt) for prompt in prompts)

    def records(self) -> dict[str, Any]:
        """Return collected prompt records."""
        return {
            "chat_messages": self.chat_messages,
            "text_prompts": self.text_prompts,
        }


def _segment_progress_label(state: DubbingState, segment: SegmentState) -> str:
    """진행 로그에 넣을 세그먼트 위치와 문장 정보를 만든다."""
    segments = state.get("segments", [])
    segment_id = segment.get("id", "?")
    index = next(
        (idx for idx, item in enumerate(segments) if item.get("id") == segment_id),
        state.get("current_segment_index", 0),
    )
    return (
        f"세그먼트={index + 1}/{len(segments) or '?'}"
        f" id={segment_id}"
        f" 문장={segment.get('sentence_id', '?')}"
    )


def inspect_source_node(state: DubbingState) -> DubbingState:
    """입력 영상 metadata를 검사해 기존 자막 활용 가능성을 판단한다."""
    next_state = deepcopy(state)
    _log_progress(next_state, "inspect_source", "YouTube 자막 메타데이터 확인 중")
    source_url = next_state.get("source_url")
    if not source_url:
        return next_state

    chains = _chains(next_state)
    result = _invoke_llm(
        chains.subtitle_inspection_agent,
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "다음 영상의 기존 자막 제공 여부를 확인하세요.\n"
                        f"영상 URL: {source_url}\n"
                        f"희망 자막 언어: {next_state.get('source_language') or 'auto'}\n"
                        f"번역 대상 언어: {next_state.get('target_language', 'ko')}"
                    ),
                }
            ]
        },
        "inspect_source",
        next_state,
    )
    decision = _structured_response(result)
    next_state.setdefault("metadata", {})["subtitle_inspection"] = _to_plain_data(
        decision
    )
    return next_state


def prepare_video_node(state: DubbingState) -> DubbingState:
    """YouTube URL에서 원본 영상을 준비한다."""
    next_state = deepcopy(state)
    _log_progress(next_state, "prepare_video", "원본 영상 준비 중")
    output_dir = ensure_dir(next_state["output_dir"])

    if next_state.get("input_video_path") and not next_state.get("source_url"):
        raise NotImplementedError(
            "로컬 영상 입력은 현재 범위에서 제외되어 있습니다. source_url을 입력하세요."
        )

    source_url = next_state.get("source_url")
    if not source_url:
        raise ValueError("현재 워크플로우에는 source_url이 필요합니다.")

    video_path = prepare_video_from_url(source_url, output_dir)
    next_state["input_video_path"] = video_path
    next_state.setdefault("metadata", {})["source_url"] = source_url
    return next_state


def load_source_subtitle_node(state: DubbingState) -> DubbingState:
    """agent가 선택한 기존 YouTube 자막을 세그먼트로 로드한다."""
    next_state = deepcopy(state)
    _log_progress(next_state, "load_source_subtitle", "선택한 YouTube 자막 로드 중")
    decision = next_state.get("metadata", {}).get("subtitle_inspection", {})
    language = str(decision.get("selected_language") or "").strip()
    subtitle_source = str(decision.get("subtitle_source") or "").strip()

    try:
        raw_segments, subtitle_path = download_youtube_subtitle_segments(
            next_state["source_url"],
            next_state["output_dir"],
            language,
            subtitle_source,
        )
    except Exception as exc:  # noqa: BLE001
        next_state.setdefault("metadata", {})["subtitle_load_error"] = str(exc)
        next_state.setdefault("metadata", {})["subtitle_fallback"] = "stt"
        next_state["segments"] = []
        _log_progress(
            next_state,
            "load_source_subtitle",
            "자막 로드 실패, STT로 전환",
        )
        return next_state

    next_state["detected_language"] = language
    next_state["segments"] = _build_segment_states(raw_segments)
    next_state["current_segment_index"] = 0
    next_state.setdefault("metadata", {})["source_subtitle_path"] = subtitle_path
    next_state.setdefault("metadata", {})["segment_count"] = len(next_state["segments"])
    _log_progress(
        next_state,
        "load_source_subtitle",
        f"자막 세그먼트 {len(next_state['segments'])}개 로드 완료",
    )
    return next_state


def extract_audio_node(state: DubbingState) -> DubbingState:
    """준비된 영상에서 원본 오디오를 추출한다."""
    next_state = deepcopy(state)
    _log_progress(next_state, "extract_audio", "원본 오디오 추출 중")
    next_state["audio_path"] = extract_source_audio(
        next_state["input_video_path"],
        next_state["output_dir"],
    )
    return next_state


def stt_node(state: DubbingState) -> DubbingState:
    """Whisper STT를 실행하고 세그먼트를 정규화한다."""
    next_state = deepcopy(state)
    _log_progress(
        next_state,
        "stt",
        f"Whisper 실행 중 model={next_state.get('whisper_model', 'large')}",
    )
    raw_segments, detected_language = transcribe_and_normalize(
        next_state["audio_path"],
        next_state.get("whisper_model", "large"),
        next_state.get("source_language"),
        next_state.get("whisper_device", "auto"),
    )
    next_state["detected_language"] = detected_language
    next_state["segments"] = _build_segment_states(raw_segments)
    next_state["current_segment_index"] = 0
    next_state.setdefault("metadata", {})["segment_count"] = len(next_state["segments"])
    _log_progress(
        next_state,
        "stt",
        f"감지 언어={detected_language}, 세그먼트={len(next_state['segments'])}개",
    )
    return next_state


def build_context_node(state: DubbingState) -> DubbingState:
    """LangChain 번역에 사용할 전체 transcript 문맥을 만든다."""
    next_state = deepcopy(state)
    _log_progress(next_state, "build_context", "전체 transcript 문맥 구성 중")
    transcript = "\n".join(
        f"[{seg['id']}] {seg['source_text']}"
        for seg in next_state.get("segments", [])
    )

    if not transcript:
        next_state.setdefault("summary", "")
        next_state.setdefault("domain", "")
        next_state.setdefault("glossary", {})
        next_state.setdefault("translation_style", "")
        _write_context_artifacts(next_state, transcript)
        return next_state

    chains = _chains(next_state)
    result = _invoke_llm(
        chains.context_chain,
        {
            "target_language": next_state.get("target_language", "ko"),
            "transcript": transcript,
        },
        "build_context",
        next_state,
    )
    next_state["summary"] = _field(result, "summary", "")
    next_state["domain"] = _field(result, "domain", "")
    next_state["glossary"] = _normalize_glossary(_field(result, "glossary", []))
    next_state["translation_style"] = _field(result, "translation_style", "")
    _write_context_artifacts(next_state, transcript)
    return next_state


def translate_segment_node(state: DubbingState) -> DubbingState:
    """현재 세그먼트를 번역한다."""
    next_state = deepcopy(state)
    segment = _current_segment(next_state)
    revision_feedback = _revision_feedback_text(segment)
    if _total_translation_attempts(segment) >= _max_translation_attempts(next_state):
        segment["translation_budget_exhausted"] = True
        segment["qc_passed"] = False
        segment["qc_action"] = "retry_budget_exhausted"
        segment["qc_issues"] = _unique_strings(
            list(segment.get("qc_issues", []))
            + ["번역 재시도 예산을 모두 사용했습니다."]
        )
        segment["adjusted_text"] = segment.get("adjusted_text") or segment.get(
            "translated_text",
            "",
        )
        _log_progress(
            next_state,
            "translate_segment",
            "재시도 예산 소진, 기존 번역 유지",
            segment,
        )
        return next_state

    segment["translation_attempts"] = segment.get("translation_attempts", 0) + 1
    segment["total_translation_attempts"] = (
        segment.get("total_translation_attempts", 0) + 1
    )
    _log_progress(
        next_state,
        "translate_segment",
        "시도="
        + str(segment.get("translation_attempts", 0))
        + " 총="
        + str(segment.get("total_translation_attempts", 0)),
        segment,
    )

    chains = _chains(next_state)
    result = _invoke_llm(
        chains.translation_chain,
        {
            "source_language": _source_language(next_state),
            "target_language": next_state.get("target_language", "ko"),
            "duration_seconds": _translation_timing_duration(next_state, segment),
            "sentence_duration_seconds": _dubbing_slot_duration(next_state, segment),
            "summary": next_state.get("summary", ""),
            "domain": next_state.get("domain", ""),
            "glossary": next_state.get("glossary", {}),
            "translation_style": next_state.get("translation_style", ""),
            "global_transcript_context": _global_transcript_context(next_state),
            "sentence_source_text": _sentence_source_text(next_state, segment),
            "sentence_translation_draft": _sentence_translation_draft(
                next_state,
                segment,
                segment.get("translated_text", ""),
            ),
            "segment_context": _segment_context(next_state),
            "neighbor_translations": _neighbor_translations(next_state),
            "revision_feedback": revision_feedback,
            "source_text": segment["source_text"],
        },
        f"translate_segment[{segment.get('id', '?')}]",
        next_state,
    )
    translated = _field(result, "translated_text", "")
    _reset_segment_after_new_translation(segment, revision_feedback)
    _clear_sentence_tts_artifacts(next_state, segment)
    segment["translated_text"] = translated
    segment["adjusted_text"] = translated
    _log_progress(
        next_state,
        "translate_segment",
        "번역 글자수=" + str(len(translated)),
        segment,
    )
    return next_state


def translation_qc_node(state: DubbingState) -> DubbingState:
    """현재 세그먼트 번역 품질을 agent와 tool로 검증한다."""
    next_state = deepcopy(state)
    segment = _current_segment(next_state)
    _log_progress(next_state, "translation_qc", "번역 품질 검사 중", segment)
    if segment.get("translation_budget_exhausted"):
        _log_progress(
            next_state,
            "translation_qc",
            "재시도 예산 소진, LLM QC 생략",
            segment,
        )
        return next_state

    translated_text = segment.get("translated_text", "")
    sentence_source_text = _sentence_source_text(next_state, segment)
    sentence_translation_draft = _sentence_translation_draft(
        next_state,
        segment,
        translated_text,
    )

    chains = _chains(next_state)
    operation = f"translation_qc[{segment.get('id', '?')}]"
    try:
        result = _invoke_optional_llm(
            chains.translation_qc_agent,
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "다음 세그먼트 번역 후보를 검증하세요.\n"
                            f"원본 언어: {_source_language(next_state)}\n"
                            f"대상 언어: {next_state.get('target_language', 'ko')}\n"
                            f"참고용 세그먼트 길이: {_segment_duration(segment):.2f}초\n"
                            f"문장 단위 길이: {_sentence_duration(segment):.2f}초\n"
                            "판단 기준: 개별 세그먼트 timestamp보다 문장 단위 흐름과 "
                            "자연스러운 말하기 속도를 우선하세요.\n"
                            f"영상 요약: {next_state.get('summary', '')}\n"
                            f"용어집: {next_state.get('glossary', {})}\n\n"
                            f"전체 원문 흐름:\n{_global_transcript_context(next_state)}\n\n"
                            f"같은 문장과 주변 문맥:\n{_segment_context(next_state)}\n\n"
                            f"이미 확정된 주변 번역:\n{_neighbor_translations(next_state)}\n\n"
                            f"문장 전체 원문:\n{sentence_source_text}\n\n"
                            f"문장 전체 번역 후보:\n{sentence_translation_draft}\n\n"
                            f"이전 품질/더빙 피드백:\n{_revision_feedback_text(segment)}\n\n"
                            f"원문:\n{segment['source_text']}\n\n"
                            f"후보 번역:\n{translated_text}"
                        ),
                    }
                ]
            },
            operation,
            next_state,
        )
    except Exception as exc:  # noqa: BLE001
        _record_llm_fallback(next_state, operation, exc)
        result = {
            "action": "pass",
            "score": 1.0,
            "passed": True,
            "issues": [],
        }
    result = _structured_response(result)
    action = _field(result, "action", "retry")
    suggested = _field(result, "suggested_translation", "")

    if suggested and action == "rewrite":
        translated_text = suggested
        segment["translated_text"] = suggested
        segment["adjusted_text"] = suggested
        _reset_translation_qc_artifacts(segment)
        _clear_sentence_tts_artifacts(next_state, segment)

    target_language = next_state.get("target_language", "ko")
    segment_metrics = _translation_quality_metrics(
        segment["source_text"],
        translated_text,
        _translation_timing_duration(next_state, segment),
        target_language,
    )
    sentence_metrics = _sentence_translation_quality_metrics(
        next_state,
        segment,
        target_language,
    )
    metrics = _combined_translation_quality_metrics(segment_metrics, sentence_metrics)
    agent_score = float(_field(result, "score", 0.0))
    agent_passed = bool(_field(result, "passed", False)) or bool(
        suggested and action == "rewrite"
    )
    min_agent_score = _env_float(
        "DUBBING_AGENT_TRANSLATION_QC_MIN_SCORE",
        DEFAULT_TRANSLATION_QC_MIN_SCORE,
    )
    agent_score_passed = agent_score >= min_agent_score
    heuristic_passed = bool(metrics.get("heuristic_passed", False))
    issues = _unique_strings(
        list(_field(result, "issues", [])) + list(metrics.get("issues", []))
    )
    if not agent_score_passed:
        issues = _unique_strings(
            issues
            + [
                "LLM 번역 QC 점수가 기준보다 낮습니다 "
                f"({agent_score:.2f} < {min_agent_score:.2f})."
            ]
        )

    segment["translation_metrics"] = metrics
    segment["qc_passed"] = agent_passed and agent_score_passed and heuristic_passed
    segment["qc_score"] = min(agent_score, float(metrics.get("heuristic_score", 0.0)))
    segment["qc_issues"] = issues
    segment["qc_action"] = action if segment["qc_passed"] else "retry"
    group = _sentence_group(next_state, segment)
    sentence_density_failed = bool(
        sentence_metrics and not sentence_metrics.get("heuristic_passed", False)
    )
    sentence_final_review_failed = (
        len(group) > 1
        and _is_sentence_group_final(next_state, segment)
        and segment["qc_passed"] is not True
    )
    if sentence_density_failed or sentence_final_review_failed:
        _mark_sentence_translation_for_retry(
            next_state,
            group,
            issues,
        )
    _log_progress(
        next_state,
        "translation_qc",
        (
            "결과="
            f"{_progress_result_label('pass' if segment['qc_passed'] else 'retry')} "
            f"점수={segment['qc_score']:.2f} 이슈={len(issues)}개"
        ),
        segment,
    )
    return next_state


def duration_adjust_node(state: DubbingState) -> DubbingState:
    """활성화된 경우 번역문을 발화 시간에 맞게 재작성한다."""
    next_state = deepcopy(state)
    original_index = next_state.get("current_segment_index", 0)
    segment = _current_segment(next_state)
    target_indices = _duration_adjust_target_indices(next_state, segment)
    _log_progress(next_state, "duration_adjust", "발화 길이 조정 중", segment)

    if not next_state.get("duration_rewrite_enabled", True):
        for idx in target_indices:
            item = next_state["segments"][idx]
            item["adjusted_text"] = item.get("adjusted_text") or item.get(
                "translated_text",
                "",
            )
        _log_progress(
            next_state,
            "duration_adjust",
            "비활성화됨, 번역문 그대로 사용",
            segment,
        )
        return next_state

    chains = _chains(next_state)
    adjusted_failures: list[str] = []
    for idx in target_indices:
        next_state["current_segment_index"] = idx
        target = next_state["segments"][idx]
        current_text = target.get("adjusted_text") or target.get("translated_text", "")
        target["duration_rewrite_attempts"] = (
            target.get("duration_rewrite_attempts", 0) + 1
        )
        result = _invoke_optional_llm(
            chains.duration_rewrite_chain,
            {
                "target_language": next_state.get("target_language", "ko"),
                "duration_seconds": _translation_timing_duration(next_state, target),
                "sentence_duration_seconds": _dubbing_slot_duration(next_state, target),
                "issue": _duration_issue(target),
                "segment_context": _segment_context(next_state),
                "neighbor_translations": _neighbor_translations(next_state),
                "source_text": target["source_text"],
                "translated_text": current_text,
            },
            f"duration_adjust[{target.get('id', '?')}]",
            next_state,
        )
        adjusted = _field(result, "adjusted_text", current_text)
        target["adjusted_text"] = adjusted or current_text
        target["adjusted_translation_metrics"] = _translation_quality_metrics(
            target["source_text"],
            target["adjusted_text"],
            _translation_timing_duration(next_state, target),
            next_state.get("target_language", "ko"),
        )
        if target["adjusted_translation_metrics"].get("heuristic_passed") is not True:
            adjusted_failures.extend(
                f"Adjusted text QC segment {target.get('id', '?')}: {issue}"
                for issue in target["adjusted_translation_metrics"].get("issues", [])
            )

    next_state["current_segment_index"] = original_index
    _clear_sentence_tts_artifacts(next_state, segment)
    if adjusted_failures:
        _mark_sentence_translation_for_retry(
            next_state,
            _sentence_group(next_state, segment),
            adjusted_failures,
        )
    if adjusted_failures:
        _log_progress(
            next_state,
            "duration_adjust",
            f"결과={_progress_result_label('retry_translation')} 이슈={len(adjusted_failures)}개",
            segment,
        )
    else:
        _log_progress(
            next_state,
            "duration_adjust",
            f"완료, 조정 세그먼트={len(target_indices)}개",
            segment,
        )
    return next_state


def tts_node(state: DubbingState) -> DubbingState:
    """현재 문장 그룹의 TTS를 생성한다."""
    next_state = deepcopy(state)
    segment = _current_segment(next_state)
    _log_progress(next_state, "tts", "문장 단위 TTS 준비 중", segment)
    if not _is_sentence_group_final(next_state, segment):
        segment["duration_passed"] = True
        segment["dubbing_action"] = "wait_sentence"
        _log_progress(next_state, "tts", "문장 그룹 종료 대기 중", segment)
        return next_state

    group = _sentence_group(next_state, segment)
    text = _sentence_tts_text(group)
    if not text.strip():
        _mark_sentence_for_retry(
            next_state,
            group,
            ["TTS로 생성할 번역문이 비어 있습니다."],
            "retry_translation",
        )
        _log_progress(next_state, "tts", "번역 재시도 필요, TTS 텍스트가 비어 있음", segment)
        return next_state

    tts_dir = ensure_dir(os.path.join(next_state["output_dir"], "tts_sentences"))
    output_path = os.path.join(tts_dir, f"sentence_{segment['sentence_id']:04d}.mp3")
    slot = _dubbing_slot(next_state, segment)
    slot_duration_ms = max(200, round(slot["duration"] * 1000))
    sentence_tts_path = synthesize_segment_tts(
        text,
        next_state.get("target_language", "ko"),
        output_path,
        next_state.get("tts_voice"),
        next_state.get("tts_rate", "+0%"),
        slot_duration_ms,
        allow_trimming=False,
    )
    for item in group:
        item["sentence_tts_path"] = sentence_tts_path
        item["sentence_tts_slot_start"] = slot["start"]
        item["sentence_tts_slot_end"] = slot["end"]
        item["sentence_tts_slot_duration"] = slot["duration"]
        item.pop("duration_passed", None)
    segment["tts_path"] = sentence_tts_path
    _log_progress(
        next_state,
        "tts",
        f"문장 TTS 생성 완료, 슬롯={slot['duration']:.2f}초",
        segment,
    )
    return next_state


def tts_duration_check_node(state: DubbingState) -> DubbingState:
    """TTS 길이가 원본 문장 구간에 맞는지 agent와 tool로 확인한다."""
    next_state = deepcopy(state)
    segment = _current_segment(next_state)
    _log_progress(
        next_state,
        "tts_duration_check",
        "문장 TTS 길이와 발화 속도 검사 중",
        segment,
    )
    if not _is_sentence_group_final(next_state, segment):
        segment["duration_passed"] = True
        segment["dubbing_action"] = "wait_sentence"
        _log_progress(
            next_state,
            "tts_duration_check",
            "문장 그룹 종료 대기 중",
            segment,
        )
        return next_state

    group = _sentence_group(next_state, segment)
    tts_path = segment.get("sentence_tts_path") or segment.get("tts_path")
    if not tts_path:
        _mark_sentence_for_retry(
            next_state,
            group,
            ["문장 단위 TTS 파일이 생성되지 않았습니다."],
            "retry_translation",
        )
        _log_progress(
            next_state,
            "tts_duration_check",
            "번역 재시도 필요, TTS 파일 없음",
            segment,
        )
        return next_state

    tts_duration = get_audio_duration_seconds(tts_path)
    original_duration = _sentence_duration(segment)
    slot = _dubbing_slot(next_state, segment)
    source_duration = slot["duration"]
    tolerance = next_state.get("tts_duration_tolerance", 0.12)
    target_language = next_state.get("target_language", "ko")
    text = _sentence_tts_text(group)
    for item in group:
        item["sentence_tts_duration"] = tts_duration
    segment["tts_duration"] = tts_duration
    local_metrics = _dubbing_quality_metrics(
        source_duration,
        tts_duration,
        tolerance,
        text,
        target_language,
    )
    local_metrics["original_source_seconds"] = round(original_duration, 3)
    local_metrics["slot_start_seconds"] = round(slot["start"], 3)
    local_metrics["slot_end_seconds"] = round(slot["end"], 3)
    local_metrics["slot_extension_seconds"] = round(slot["extension"], 3)
    local_metrics = _merge_dubbing_alignment_metrics(
        local_metrics,
        _tts_alignment_metrics(next_state, segment, tts_duration, slot),
    )
    sentence_metrics = local_metrics

    chains = _chains(next_state)
    operation = f"tts_duration_check[{segment.get('id', '?')}]"
    try:
        result = _invoke_optional_llm(
            chains.dubbing_qc_agent,
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "다음 문장 단위 TTS의 더빙 적합성을 검증하세요.\n"
                            f"원본 문장 길이: {original_duration:.2f}초\n"
                            f"사용 가능한 더빙 슬롯: {source_duration:.2f}초\n"
                            f"슬롯 확장: {slot['extension']:.2f}초\n"
                            f"문장 TTS 길이: {tts_duration:.2f}초\n"
                            f"허용 초과 비율: {tolerance:.3f}\n"
                            f"대상 언어: {next_state.get('target_language', 'ko')}\n"
                            f"같은 문장과 주변 문맥:\n{_segment_context(next_state)}\n\n"
                            f"문장 단위 길이/속도 분석:\n{local_metrics}\n\n"
                            f"대사:\n{text}"
                        ),
                    }
                ]
            },
            operation,
            next_state,
        )
    except Exception as exc:  # noqa: BLE001
        _record_llm_fallback(next_state, operation, exc)
        result = {
            "action": "continue",
            "score": 1.0,
            "passed": True,
            "issues": [],
        }
    result = _structured_response(result)
    agent_action = _field(result, "action", "rewrite")
    agent_score = float(_field(result, "score", 0.0))
    agent_passed = bool(_field(result, "passed", False)) or agent_action == "continue"
    min_agent_score = _env_float(
        "DUBBING_AGENT_DUBBING_QC_MIN_SCORE",
        DEFAULT_DUBBING_QC_MIN_SCORE,
    )
    agent_score_passed = agent_score >= min_agent_score
    local_passed = bool(local_metrics.get("heuristic_passed", False))
    issues = _unique_strings(
        list(_field(result, "issues", []))
        + list(local_metrics.get("issues", []))
    )
    if not agent_score_passed:
        issues = _unique_strings(
            issues
            + [
                "LLM 더빙 QC 점수가 기준보다 낮습니다 "
                f"({agent_score:.2f} < {min_agent_score:.2f})."
            ]
        )
    score_candidates = [
        agent_score,
        float(local_metrics.get("heuristic_score", 0.0)),
    ]

    passed = agent_passed and agent_score_passed and local_passed
    action = _dubbing_action(
        agent_action,
        local_metrics,
        sentence_metrics,
    )
    for item in group:
        item["duration_passed"] = passed
        item["dubbing_score"] = min(score_candidates)
        item["dubbing_issues"] = issues
        item["dubbing_metrics"] = {
            "sentence": sentence_metrics,
        }
        item["dubbing_action"] = action
    if not passed:
        _append_sentence_revision_feedback(group, issues)
        retry_index = _sentence_retry_index(next_state, group, action)
        next_state["current_segment_index"] = retry_index
    _log_progress(
        next_state,
        "tts_duration_check",
        (
            "결과="
            f"{_progress_result_label('pass' if passed else action)} "
            f"원본={original_duration:.2f}초 슬롯={source_duration:.2f}초 "
            f"TTS={tts_duration:.2f}초 "
            f"이슈={len(issues)}개"
        ),
        segment,
    )
    return next_state


def next_segment_node(state: DubbingState) -> DubbingState:
    """현재 세그먼트 인덱스를 다음으로 이동한다."""
    next_state = deepcopy(state)
    if next_state.get("global_translation_reprocess_segment_ids"):
        _advance_global_translation_reprocess(next_state)
        return next_state

    next_state["current_segment_index"] = next_state.get("current_segment_index", 0) + 1
    return next_state


def global_translation_review_node(state: DubbingState) -> DubbingState:
    """전체 transcript 기준으로 최종 번역 흐름을 검토하고 필요한 수정안을 적용한다."""
    next_state = deepcopy(state)
    segments = next_state.get("segments", [])
    metadata = next_state.setdefault("metadata", {})
    _log_progress(
        next_state,
        "global_translation_review",
        "전체 번역 흐름 검토 중",
    )
    if not segments:
        review_metadata = {
            "passed": False,
            "score": 0.0,
            "issues": ["원본 transcript 세그먼트가 없습니다."],
            "action": "fail",
        }
        metadata["global_translation_review"] = review_metadata
        _append_global_translation_review_history(metadata, review_metadata)
        return next_state

    attempts = int(next_state.get("global_translation_review_attempts", 0)) + 1
    next_state["global_translation_review_attempts"] = attempts
    chains = _chains(next_state)
    try:
        result = _invoke_optional_llm(
            chains.global_translation_review_chain,
            _global_translation_review_payload(next_state),
            "global_translation_review",
            next_state,
        )
    except Exception as exc:  # noqa: BLE001
        _record_llm_fallback(next_state, "global_translation_review", exc)
        result = {
            "passed": True,
            "score": 1.0,
            "summary": "LLM 전체 번역 리뷰 실패로 로컬 품질 게이트만 적용했습니다.",
            "issues": [],
            "action": "pass",
        }
    review = _structured_response(result)
    issues = _global_review_issues(review)
    score = float(_field(review, "score", 0.0))
    min_review_score = _env_float(
        "DUBBING_AGENT_GLOBAL_TRANSLATION_REVIEW_MIN_SCORE",
        DEFAULT_GLOBAL_TRANSLATION_REVIEW_MIN_SCORE,
    )
    if score < min_review_score:
        issues.append(
            {
                "segment_id": -1,
                "issue": (
                    "전체 번역 리뷰 점수가 기준보다 낮습니다 "
                    f"({score:.2f} < {min_review_score:.2f})."
                ),
                "suggested_translation": "",
            }
        )
    passed = bool(_field(review, "passed", False)) and not issues
    summary = _field(review, "summary", "")

    review_metadata = {
        "passed": passed,
        "score": score,
        "summary": summary,
        "issues": issues,
        "attempts": attempts,
        "action": "pass" if passed else "fail",
    }
    metadata["global_translation_review"] = review_metadata

    if passed:
        _append_global_translation_review_history(metadata, review_metadata)
        return next_state

    max_attempts = int(next_state.get("max_global_translation_review_attempts", 2))
    if attempts > max_attempts:
        _append_global_translation_review_history(metadata, review_metadata)
        return next_state

    changed_indices = _apply_global_translation_suggestions(next_state, issues)
    if not changed_indices:
        _append_global_translation_review_history(metadata, review_metadata)
        return next_state

    review_metadata["action"] = "reprocess"
    review_metadata["changed_segment_ids"] = [
        next_state["segments"][idx].get("id") for idx in changed_indices
    ]
    reprocess_ids = _global_review_reprocess_segment_ids(next_state, changed_indices)
    review_metadata["reprocess_segment_ids"] = reprocess_ids
    retry_budget_reset_ids = _global_review_retry_budget_segment_ids(
        next_state,
        changed_indices,
    )
    _reset_global_review_retry_budgets(next_state, retry_budget_reset_ids)
    review_metadata["retry_budget_reset_segment_ids"] = retry_budget_reset_ids
    _append_global_translation_review_history(metadata, review_metadata)
    if not reprocess_ids:
        return next_state
    next_state["global_translation_reprocess_segment_ids"] = reprocess_ids
    next_state["current_segment_index"] = _segment_index_by_id(
        next_state,
        reprocess_ids[0],
    )
    return next_state


def final_quality_gate_node(state: DubbingState) -> DubbingState:
    """Record unresolved quality failures before final artifact generation."""
    next_state = deepcopy(state)
    failures = _quality_gate_failures(next_state)
    _log_progress(next_state, "final_quality_gate", "해결되지 않은 품질 실패 확인 중")
    metadata = next_state.setdefault("metadata", {})
    metadata["quality_gate"] = {
        "enforced": bool(next_state.get("enforce_quality_gates", True)),
        "passed": not failures,
        "failures": failures,
    }

    if failures:
        report_path = _try_write_quality_report(next_state)
        if report_path:
            metadata["quality_gate"]["failure_report_path"] = report_path
            metadata["quality_gate"]["failure_summary_path"] = _quality_summary_path(
                next_state
            )
            next_state["quality_report_path"] = report_path
            next_state["quality_summary_path"] = _quality_summary_path(next_state)
        metadata["quality_gate"]["continued_with_failures"] = True
        _log_progress(
            next_state,
            "final_quality_gate",
            f"품질 실패 {len(failures)}개 기록 후 산출물 생성을 계속",
        )

    return next_state


def generate_subtitle_node(state: DubbingState) -> DubbingState:
    """번역된 세그먼트로 자막을 생성한다."""
    next_state = deepcopy(state)
    _log_progress(next_state, "generate_subtitle", "자막 파일 작성 중")
    video_path = next_state.get("input_video_path", "output")
    base_name = os.path.splitext(os.path.basename(video_path))[0] or next_state["job_id"]
    subtitle_path = os.path.join(next_state["output_dir"], f"{base_name}_sub.srt")
    subtitle_segments = _subtitle_segments(next_state)
    next_state["subtitle_path"] = write_subtitle_file(
        subtitle_segments,
        _source_language(next_state),
        subtitle_path,
        next_state.get("subtitle_mode", "full"),
        "srt",
    )
    return next_state


def compose_audio_node(state: DubbingState) -> DubbingState:
    """생성된 모든 TTS 세그먼트를 하나의 더빙 오디오 트랙으로 합성한다."""
    next_state = deepcopy(state)
    _log_progress(next_state, "compose_audio", "문장 TTS 오디오 합성 중")
    video_path = next_state["input_video_path"]
    base_name = os.path.splitext(os.path.basename(video_path))[0] or next_state["job_id"]
    final_audio_path = os.path.join(next_state["output_dir"], f"{base_name}_tts_merged.wav")
    tts_segments = _dubbing_audio_units(next_state)
    total_duration = get_video_duration(video_path)
    next_state["final_audio_path"] = compose_dubbed_audio(
        tts_segments,
        total_duration,
        final_audio_path,
        fit_overlong_segments=bool(
            next_state.get("metadata", {})
            .get("quality_gate", {})
            .get("continued_with_failures", False)
        ),
    )
    return next_state


def write_quality_report_node(state: DubbingState) -> DubbingState:
    """번역/더빙 QC 결과를 작업 디렉터리에 JSON 리포트로 저장한다."""
    next_state = deepcopy(state)
    _log_progress(next_state, "write_quality_report", "품질 리포트 작성 중")
    report_path = _write_quality_report(next_state)
    next_state["quality_report_path"] = report_path
    next_state["quality_summary_path"] = _quality_summary_path(next_state)
    return next_state


def mux_video_node(state: DubbingState) -> DubbingState:
    """원본 영상에 더빙 오디오를 합성한다."""
    next_state = deepcopy(state)
    _log_progress(next_state, "mux_video", "더빙 오디오를 영상에 합성 중")
    video_path = next_state["input_video_path"]
    base_name = os.path.splitext(os.path.basename(video_path))[0] or next_state["job_id"]
    output_path = os.path.join(next_state["output_dir"], f"{base_name}_dubbed.mp4")
    next_state["output_video_path"] = mux_dubbed_video(
        video_path,
        next_state["final_audio_path"],
        output_path,
    )
    return next_state


def _build_segment_states(raw_segments: list[dict[str, Any]]) -> list[SegmentState]:
    """원본 세그먼트를 내부 상태로 변환하고 문장 그룹을 태깅한다."""
    segments: list[SegmentState] = [
        {
            "id": idx,
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "source_text": str(seg["text"]).strip(),
            "translation_attempts": 0,
            "sentence_translation_retry_attempts": 0,
            "duration_rewrite_attempts": 0,
            "qc_issues": [],
            "dubbing_issues": [],
            "revision_feedback": [],
        }
        for idx, seg in enumerate(raw_segments)
        if str(seg.get("text", "")).strip()
    ]
    _annotate_sentence_groups(segments)
    return segments


def _annotate_sentence_groups(segments: list[SegmentState]) -> None:
    """구두점, gap, 안전 상한을 기준으로 세그먼트를 문장 단위로 묶는다."""
    if not segments:
        return

    group_start = 0
    sentence_id = 0
    for idx, segment in enumerate(segments):
        next_segment = segments[idx + 1] if idx + 1 < len(segments) else None
        large_gap = (
            next_segment is not None
            and float(next_segment["start"]) - float(segment["end"]) > SENTENCE_GAP_SECONDS
        )
        exceeds_limit = (
            next_segment is not None
            and _would_exceed_sentence_limits(segments, group_start, idx, next_segment)
        )
        if (
            next_segment is not None
            and not _is_sentence_boundary(segment["source_text"])
            and not large_gap
            and not exceeds_limit
        ):
            continue

        group = segments[group_start : idx + 1]
        sentence_start = float(group[0]["start"])
        sentence_end = float(group[-1]["end"])
        sentence_text = " ".join(item["source_text"] for item in group)
        for item in group:
            item["sentence_id"] = sentence_id
            item["sentence_start"] = sentence_start
            item["sentence_end"] = sentence_end
            item["sentence_text"] = sentence_text

        sentence_id += 1
        group_start = idx + 1


def _is_sentence_boundary(text: str) -> bool:
    """텍스트가 문장 종료로 보이는지 판단한다."""
    return text.strip().endswith((".", "?", "!", "…", "。", "？", "！"))


def _would_exceed_sentence_limits(
    segments: list[SegmentState],
    group_start: int,
    current_idx: int,
    next_segment: SegmentState,
) -> bool:
    """다음 세그먼트를 붙였을 때 문장 그룹 안전 상한을 넘는지 판단한다."""
    group = segments[group_start : current_idx + 1] + [next_segment]
    duration = float(group[-1]["end"]) - float(group[0]["start"])
    text_length = len(" ".join(item["source_text"] for item in group))
    return duration > SENTENCE_MAX_SECONDS or text_length > SENTENCE_MAX_CHARS


def _is_sentence_group_final(state: DubbingState, segment: SegmentState) -> bool:
    """현재 세그먼트가 문장 그룹의 마지막인지 반환한다."""
    group = _sentence_group(state, segment)
    return bool(group) and group[-1].get("id") == segment.get("id")


def _sentence_tts_text(group: list[SegmentState]) -> str:
    """문장 그룹 전체를 하나의 TTS 입력 텍스트로 결합한다."""
    parts = [
        item.get("adjusted_text") or item.get("translated_text", "")
        for item in group
    ]
    return " ".join(part.strip() for part in parts if part.strip())


def _sentence_source_text(state: DubbingState, segment: SegmentState) -> str:
    """현재 세그먼트가 속한 문장 전체 원문을 반환한다."""
    group = _sentence_group(state, segment)
    if not group:
        return segment.get("source_text", "")
    return str(
        segment.get("sentence_text")
        or " ".join(item.get("source_text", "") for item in group)
    ).strip()


def _sentence_translation_draft(
    state: DubbingState,
    segment: SegmentState,
    current_candidate: str,
) -> str:
    """현재 후보를 포함한 문장 전체 번역 초안을 반환한다."""
    group = _sentence_group(state, segment)
    if not group:
        return str(current_candidate).strip()

    current_id = segment.get("id")
    parts = []
    for item in group:
        if item.get("id") == current_id:
            text = current_candidate
        else:
            text = item.get("adjusted_text") or item.get("translated_text", "")
        parts.append(str(text).strip())
    draft = " ".join(part for part in parts if part)
    return draft or "없음"


def _duration_adjust_target_indices(
    state: DubbingState,
    segment: SegmentState,
) -> list[int]:
    """길이 재작성 대상 세그먼트 인덱스를 반환한다."""
    current_idx = state.get("current_segment_index", 0)
    group = _sentence_group(state, segment)
    if (
        group
        and _is_sentence_group_final(state, segment)
        and any(item.get("dubbing_action") == "rewrite" for item in group)
        and any(item.get("duration_passed") is False for item in group)
    ):
        group_ids = {item.get("id") for item in group}
        return [
            idx
            for idx, item in enumerate(state.get("segments", []))
            if item.get("id") in group_ids
        ]
    return [current_idx]


def _clear_sentence_tts_artifacts(
    state: DubbingState,
    segment: SegmentState,
) -> None:
    """문장 내 번역이 바뀔 때 기존 문장 TTS/QC 결과를 지운다."""
    for item in _sentence_group(state, segment):
        _preserve_fallback_tts_artifacts(item)
        for key in (
            "sentence_tts_path",
            "sentence_tts_duration",
            "sentence_tts_slot_start",
            "sentence_tts_slot_end",
            "sentence_tts_slot_duration",
            "tts_path",
            "tts_duration",
            "duration_passed",
            "dubbing_score",
            "dubbing_action",
            "dubbing_metrics",
        ):
            item.pop(key, None)
        item["dubbing_issues"] = []


def _preserve_fallback_tts_artifacts(segment: SegmentState) -> None:
    """Keep the last generated TTS so failed reprocessing can still render audio."""
    path = segment.get("sentence_tts_path") or segment.get("tts_path")
    if not path:
        return

    segment["fallback_sentence_tts_path"] = path
    if "sentence_tts_duration" in segment:
        segment["fallback_sentence_tts_duration"] = segment["sentence_tts_duration"]
    if "sentence_tts_slot_start" in segment:
        segment["fallback_sentence_tts_slot_start"] = segment["sentence_tts_slot_start"]
    if "sentence_tts_slot_end" in segment:
        segment["fallback_sentence_tts_slot_end"] = segment["sentence_tts_slot_end"]
    if "sentence_tts_slot_duration" in segment:
        segment["fallback_sentence_tts_slot_duration"] = segment[
            "sentence_tts_slot_duration"
        ]


def _mark_sentence_for_retry(
    state: DubbingState,
    group: list[SegmentState],
    issues: list[str],
    action: str,
) -> None:
    """문장 그룹 전체에 더빙 실패 상태를 표시한다."""
    unique_issues = _unique_strings(issues)
    for item in group:
        item["duration_passed"] = False
        item["dubbing_action"] = action
        item["dubbing_issues"] = unique_issues
    _append_sentence_revision_feedback(group, unique_issues)
    if group:
        state["current_segment_index"] = _segment_index_by_id(state, group[0]["id"])


def _mark_sentence_translation_for_retry(
    state: DubbingState,
    group: list[SegmentState],
    issues: list[str],
) -> None:
    """문장 단위 번역 QC 실패 시 같은 문장 전체를 재번역 대상으로 되돌린다."""
    if not group:
        return

    retry_issues = _unique_strings(
        f"문장 번역 QC: {issue}"
        for issue in issues
        if str(issue).strip()
    )
    for item in group:
        item["qc_passed"] = False
        item["qc_action"] = "retry"
        item["qc_issues"] = _unique_strings(
            list(item.get("qc_issues", [])) + retry_issues
        )
        item["revision_feedback"] = _bounded_revision_feedback(
            list(item.get("revision_feedback", [])) + retry_issues
        )
        item["adjusted_text"] = item.get("translated_text", "")
        item.pop("adjusted_translation_metrics", None)
        item["duration_rewrite_attempts"] = 0
        _clear_sentence_tts_artifacts(state, item)

    _reserve_sentence_translation_retry(state, group)

    state["current_segment_index"] = _segment_index_by_id(state, group[0]["id"])


def _reserve_sentence_translation_retry(
    state: DubbingState,
    group: list[SegmentState],
) -> bool:
    """문장 단위 재번역 라운드 예산을 예약하고 남은 예산 여부를 반환한다."""
    if not group:
        return False

    max_attempts = _max_translation_attempts(state)
    next_attempt = max(
        int(item.get("sentence_translation_retry_attempts", 0)) for item in group
    ) + 1
    for item in group:
        item["sentence_translation_retry_attempts"] = next_attempt

    if next_attempt <= max_attempts and any(
        _total_translation_attempts(item) < max_attempts for item in group
    ):
        return True

    for item in group:
        item["translation_attempts"] = max_attempts
        item["total_translation_attempts"] = max_attempts
    return False


def _append_sentence_revision_feedback(
    group: list[SegmentState],
    issues: list[str],
) -> None:
    """문장 단위 실패 이유를 이후 재번역 프롬프트용 피드백으로 보존한다."""
    feedback = _unique_strings(f"Dubbing QC: {issue}" for issue in issues)
    if not feedback:
        return
    for item in group:
        item["revision_feedback"] = _bounded_revision_feedback(
            list(item.get("revision_feedback", [])) + feedback
        )


def _sentence_retry_index(
    state: DubbingState,
    group: list[SegmentState],
    action: str,
) -> int:
    """더빙 실패 후 다시 처리할 세그먼트 인덱스를 선택한다."""
    if not group:
        return state.get("current_segment_index", 0)
    if action == "retry_translation":
        return _segment_index_by_id(state, group[0]["id"])
    return _segment_index_by_id(state, group[-1]["id"])


def _segment_index_by_id(state: DubbingState, segment_id: int) -> int:
    """세그먼트 id로 현재 state 내 인덱스를 찾는다."""
    for idx, item in enumerate(state.get("segments", [])):
        if item.get("id") == segment_id:
            return idx
    return state.get("current_segment_index", 0)


def _advance_global_translation_reprocess(state: DubbingState) -> None:
    """전체 번역 리뷰 수정 후 재처리할 다음 세그먼트로 이동한다."""
    current_idx = state.get("current_segment_index", 0)
    current_id = None
    if 0 <= current_idx < len(state.get("segments", [])):
        current_id = state["segments"][current_idx].get("id")

    pending = [
        int(segment_id)
        for segment_id in state.get("global_translation_reprocess_segment_ids", [])
        if segment_id != current_id
    ]
    state["global_translation_reprocess_segment_ids"] = pending

    if not pending:
        state["current_segment_index"] = len(state.get("segments", []))
        return

    state["current_segment_index"] = _segment_index_by_id(state, pending[0])


def _dubbing_audio_units(state: DubbingState) -> list[dict[str, Any]]:
    """문장 단위 TTS 파일 목록을 오디오 합성 입력으로 변환한다."""
    units: list[dict[str, Any]] = []
    seen_sentence_ids: set[Any] = set()

    for segment in state.get("segments", []):
        sentence_id = segment.get("sentence_id", segment.get("id"))
        if sentence_id in seen_sentence_ids:
            continue

        group = _sentence_group(state, segment)
        if not group:
            continue
        final_segment = group[-1]
        audio_path = (
            final_segment.get("sentence_tts_path")
            or final_segment.get("tts_path")
            or final_segment.get("fallback_sentence_tts_path")
        )
        if not audio_path:
            continue

        slot = _dubbing_audio_slot(state, final_segment)
        seen_sentence_ids.add(sentence_id)
        unit = {
            "start": slot["start"],
            "end": slot["end"],
            "audio_path": audio_path,
        }
        if not final_segment.get("sentence_tts_path") and final_segment.get(
            "fallback_sentence_tts_path"
        ):
            unit["fallback"] = True
        units.append(unit)

    return units


def _try_write_quality_report(state: DubbingState) -> str:
    """가능한 경우 품질 리포트를 저장하고, 불가능하면 빈 문자열을 반환한다."""
    if not state.get("output_dir"):
        return ""
    try:
        report_path = _quality_report_path(state)
        state.setdefault("metadata", {}).setdefault("quality_gate", {})[
            "failure_report_path"
        ] = report_path
        state.setdefault("metadata", {}).setdefault("quality_gate", {})[
            "failure_summary_path"
        ] = _quality_summary_path(state)
        return _write_quality_report(state)
    except Exception as exc:  # noqa: BLE001
        state.setdefault("metadata", {})["quality_report_write_error"] = str(exc)
        return ""


def _quality_report_path(state: DubbingState) -> str:
    """품질 리포트 파일 경로를 계산한다."""
    video_path = state.get("input_video_path") or state.get("job_id", "output")
    base_name = os.path.splitext(os.path.basename(video_path))[0] or state.get(
        "job_id",
        "output",
    )
    return os.path.join(state["output_dir"], f"{base_name}_quality_report.json")


def _quality_summary_path(state: DubbingState) -> str:
    """사람이 읽는 품질 요약 파일 경로를 계산한다."""
    video_path = state.get("input_video_path") or state.get("job_id", "output")
    base_name = os.path.splitext(os.path.basename(video_path))[0] or state.get(
        "job_id",
        "output",
    )
    return os.path.join(state["output_dir"], f"{base_name}_quality_summary.md")


def _write_quality_report(state: DubbingState) -> str:
    """품질 리포트 JSON을 쓰고 경로를 반환한다."""
    report_path = _quality_report_path(state)
    report_data = _quality_report_data(state)
    ensure_dir(os.path.dirname(report_path))
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    summary_path = _quality_summary_path(state)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(_quality_summary_markdown(report_data))
    return report_path


def _quality_summary_markdown(report: dict[str, Any]) -> str:
    """품질 리포트 데이터를 사람이 읽기 쉬운 Markdown으로 변환한다."""
    summary = report.get("summary", {})
    metadata = report.get("metadata", {})
    quality_gate = metadata.get("quality_gate", {})
    gate_status = "통과" if quality_gate.get("passed", False) else "실패"
    lines = [
        "# 더빙 품질 요약",
        "",
        f"- 작업 ID: {report.get('job_id', '')}",
        f"- 번역 대상 언어: {report.get('target_language', '')}",
        f"- 원본 언어: {report.get('source_language', '')}",
        f"- 품질 게이트: {gate_status}",
        f"- 품질 게이트 강제 적용: {_display_bool(quality_gate.get('enforced'))}",
        f"- 세그먼트 수: {summary.get('segment_count', 0)}",
        f"- 문장 수: {summary.get('sentence_count', 0)}",
        f"- 최소 번역 점수: {_display_value(summary.get('min_translation_score'))}",
        f"- 최소 더빙 점수: {_display_value(summary.get('min_dubbing_score'))}",
        "",
        "## 산출물",
        "",
        f"- 자막: {report.get('subtitle_path', '')}",
        f"- 더빙 오디오: {report.get('final_audio_path', '')}",
        f"- 더빙 영상: {report.get('output_video_path', '')}",
        "",
        "## 품질 게이트 실패",
        "",
    ]

    failures = quality_gate.get("failures", [])
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- 없음")

    global_review = metadata.get("global_translation_review", {})
    lines.extend(["", "## 전체 번역 리뷰", ""])
    if global_review:
        lines.extend(
            [
                f"- 통과 여부: {_display_bool(global_review.get('passed'))}",
                f"- 점수: {_display_value(global_review.get('score'))}",
                f"- 시도 횟수: {global_review.get('attempts', 0)}",
                f"- 조치: {_display_action(global_review.get('action', ''))}",
            ]
        )
        changed_segment_ids = global_review.get("changed_segment_ids", [])
        if changed_segment_ids:
            lines.append(f"- 변경된 세그먼트: {changed_segment_ids}")
        issues = global_review.get("issues", [])
        if issues:
            lines.append("")
            for issue in issues:
                lines.append(f"- {_global_review_issue_text(issue)}")
        else:
            lines.append("- 이슈: 없음")
    else:
        lines.append("- 실행되지 않음")

    history = metadata.get("global_translation_review_history", [])
    if history:
        lines.extend(["", "### 리뷰 이력", ""])
        for item in history:
            lines.append(f"- {_global_review_history_text(item)}")

    lines.extend(["", "## 재시도한 세그먼트", ""])
    retried_segments = [
        segment
        for segment in report.get("segments", [])
        if segment.get("translation_attempts", 0) > 1
        or segment.get("sentence_translation_retry_attempts", 0) > 0
        or segment.get("duration_rewrite_attempts", 0) > 0
        or segment.get("revision_feedback")
    ]
    if retried_segments:
        for segment in retried_segments:
            lines.append(
                "- 세그먼트 "
                f"{segment.get('id')}: 번역 시도="
                f"{segment.get('translation_attempts', 0)}, "
                "문장 번역 재시도="
                f"{segment.get('sentence_translation_retry_attempts', 0)}, "
                f"길이 재작성 시도={segment.get('duration_rewrite_attempts', 0)}, "
                f"QC 점수={_display_value(segment.get('qc_score'))}"
            )
    else:
        lines.append("- 없음")

    lines.extend(["", "## 문장 더빙 QC", ""])
    for sentence in report.get("sentences", []):
        issues = "; ".join(sentence.get("dubbing_issues", [])) or "없음"
        lines.append(
            "- 문장 "
            f"{sentence.get('sentence_id')}: 세그먼트={sentence.get('segment_ids')}, "
            f"통과={_display_bool(sentence.get('duration_passed'))}, "
            f"점수={_display_value(sentence.get('dubbing_score'))}, "
            f"TTS 길이={_display_value(sentence.get('sentence_tts_duration'))}초, "
            f"이슈={issues}"
        )

    lines.append("")
    return "\n".join(lines)


def _display_value(value: Any) -> str:
    """Markdown summary에 표시할 값을 안정적으로 문자열화한다."""
    if value is None:
        return "없음"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _display_bool(value: Any) -> str:
    """Markdown summary에 표시할 bool 값을 한국어로 변환한다."""
    if value is True:
        return "예"
    if value is False:
        return "아니요"
    return "알 수 없음"


def _display_action(value: Any) -> str:
    """Markdown summary에 표시할 action 값을 한국어로 변환한다."""
    return _progress_result_label(str(value))


def _quality_report_data(state: DubbingState) -> dict[str, Any]:
    """최종 품질 리포트에 쓸 직렬화 가능한 데이터를 만든다."""
    segments = state.get("segments", [])
    translation_scores = [
        float(seg["qc_score"]) for seg in segments if "qc_score" in seg
    ]
    dubbing_scores = [
        float(seg["dubbing_score"]) for seg in segments if "dubbing_score" in seg
    ]
    translation_failures = [
        seg.get("id")
        for seg in segments
        if seg.get("qc_passed") is False or seg.get("qc_issues")
    ]
    dubbing_failures = [
        seg.get("id")
        for seg in segments
        if seg.get("duration_passed") is False or seg.get("dubbing_issues")
    ]

    return {
        "job_id": state.get("job_id", ""),
        "source_url": state.get("source_url", ""),
        "input_video_path": state.get("input_video_path", ""),
        "subtitle_path": state.get("subtitle_path", ""),
        "final_audio_path": state.get("final_audio_path", ""),
        "output_video_path": state.get("output_video_path", ""),
        "target_language": state.get("target_language", ""),
        "source_language": _source_language(state),
        "summary": {
            "segment_count": len(segments),
            "sentence_count": len(_quality_sentence_entries(state)),
            "translation_failures": translation_failures,
            "dubbing_failures": dubbing_failures,
            "min_translation_score": min(translation_scores) if translation_scores else None,
            "min_dubbing_score": min(dubbing_scores) if dubbing_scores else None,
        },
        "sentences": _quality_sentence_entries(state),
        "segments": [_quality_segment_entry(seg) for seg in segments],
        "metadata": state.get("metadata", {}),
    }


def _quality_gate_failures(state: DubbingState) -> list[str]:
    """최종 출력 전 차단해야 하는 품질 실패 목록을 반환한다."""
    segments = state.get("segments", [])
    if not segments:
        return ["원본 transcript 세그먼트가 없어 더빙 품질을 검증할 수 없습니다."]

    failures: list[str] = []
    pending_reprocess_ids = state.get("global_translation_reprocess_segment_ids", [])
    if pending_reprocess_ids:
        failures.append(
            "전체 번역 리뷰: 재처리 대기 세그먼트 "
            f"{pending_reprocess_ids}"
        )

    global_review = state.get("metadata", {}).get("global_translation_review", {})
    if not global_review:
        failures.append("전체 번역 리뷰: 실행되지 않음")
    elif global_review.get("passed") is not True:
        issues = global_review.get("issues", [])
        if issues:
            issue_text = "; ".join(
                _global_review_issue_text(issue) for issue in issues
            )
        else:
            issue_text = "전체 번역 리뷰를 통과하지 못했습니다."
        failures.append(f"전체 번역 리뷰: {issue_text}")

    for segment in segments:
        segment_id = segment.get("id", "?")
        translated_text = segment.get("adjusted_text") or segment.get("translated_text", "")
        if not str(translated_text).strip():
            failures.append(f"세그먼트 {segment_id} 번역: 번역문이 비어 있습니다.")
        if segment.get("qc_passed") is not True:
            issues = "; ".join(segment.get("qc_issues", [])) or "번역 QC를 통과하지 못했습니다."
            failures.append(f"세그먼트 {segment_id} 번역: {issues}")
            continue

        translation_metrics = segment.get("translation_metrics", {})
        if not isinstance(translation_metrics, dict) or not translation_metrics:
            failures.append(f"세그먼트 {segment_id} 번역: QC 지표가 없습니다.")
        elif translation_metrics.get("heuristic_passed") is not True:
            failures.append(
                f"세그먼트 {segment_id} 번역: "
                f"{_metrics_issues_text(translation_metrics, '번역 휴리스틱 검사를 통과하지 못했습니다.')}"
            )
        adjusted_metrics = segment.get("adjusted_translation_metrics", {})
        if _requires_adjusted_translation_metrics(segment):
            if not isinstance(adjusted_metrics, dict) or not adjusted_metrics:
                failures.append(
                    f"세그먼트 {segment_id} 번역: 조정문 QC 지표가 없습니다."
                )
            elif adjusted_metrics.get("heuristic_passed") is not True:
                failures.append(
                    f"세그먼트 {segment_id} 번역: 조정문 "
                    f"{_metrics_issues_text(adjusted_metrics, '번역 휴리스틱 검사를 통과하지 못했습니다.')}"
                )
        elif adjusted_metrics:
            if not isinstance(adjusted_metrics, dict):
                failures.append(
                    f"세그먼트 {segment_id} 번역: 조정문 QC 지표가 잘못되었습니다."
                )
            elif adjusted_metrics.get("heuristic_passed") is not True:
                failures.append(
                    f"세그먼트 {segment_id} 번역: 조정문 "
                    f"{_metrics_issues_text(adjusted_metrics, '번역 휴리스틱 검사를 통과하지 못했습니다.')}"
                )

    seen_sentence_ids: set[Any] = set()
    for segment in segments:
        sentence_id = segment.get("sentence_id", segment.get("id"))
        if sentence_id in seen_sentence_ids:
            continue
        seen_sentence_ids.add(sentence_id)

        group = _sentence_group(state, segment)
        if not group:
            failures.append(f"문장 {sentence_id} 더빙: 빈 문장 그룹입니다.")
            continue

        final_segment = group[-1]
        if not final_segment.get("sentence_tts_path"):
            failures.append(f"문장 {sentence_id} 더빙: 문장 TTS가 없습니다.")
        if final_segment.get("duration_passed") is not True:
            issues = (
                "; ".join(final_segment.get("dubbing_issues", []))
                or "더빙 QC를 통과하지 못했습니다."
            )
            failures.append(f"문장 {sentence_id} 더빙: {issues}")
            continue

        dubbing_metrics = final_segment.get("dubbing_metrics", {}).get("sentence")
        if not isinstance(dubbing_metrics, dict) or not dubbing_metrics:
            failures.append(f"문장 {sentence_id} 더빙: QC 지표가 없습니다.")
        elif dubbing_metrics.get("speed_passed") is not True:
            failures.append(
                f"문장 {sentence_id} 더빙: "
                f"{_metrics_issues_text(dubbing_metrics, '발화 속도 검사를 통과하지 못했습니다.')}"
            )
        elif dubbing_metrics.get("heuristic_passed") is not True:
            failures.append(
                f"문장 {sentence_id} 더빙: "
                f"{_metrics_issues_text(dubbing_metrics, '더빙 휴리스틱 검사를 통과하지 못했습니다.')}"
            )

    return failures


def _requires_adjusted_translation_metrics(segment: SegmentState) -> bool:
    """최종 TTS 입력이 별도 조정된 경우 adjusted_text 품질 metric이 필요한지 반환한다."""
    if int(segment.get("duration_rewrite_attempts", 0)) > 0:
        return True
    if "adjusted_text" not in segment:
        return False
    adjusted = str(segment.get("adjusted_text") or "").strip()
    translated = str(segment.get("translated_text") or "").strip()
    return bool(adjusted) and adjusted != translated


def _metrics_issues_text(metrics: dict[str, Any], default: str) -> str:
    """품질 metric failure를 최종 게이트 메시지로 변환한다."""
    issues = metrics.get("issues", [])
    if isinstance(issues, list):
        text = "; ".join(str(issue) for issue in issues if str(issue).strip())
        if text:
            return text
    return default


def _quality_sentence_entries(state: DubbingState) -> list[dict[str, Any]]:
    """문장 단위 더빙 QC 리포트 항목을 만든다."""
    entries: list[dict[str, Any]] = []
    seen_sentence_ids: set[Any] = set()
    for segment in state.get("segments", []):
        sentence_id = segment.get("sentence_id", segment.get("id"))
        if sentence_id in seen_sentence_ids:
            continue
        seen_sentence_ids.add(sentence_id)
        group = _sentence_group(state, segment)
        if not group:
            continue
        final_segment = group[-1]
        entries.append(
            {
                "sentence_id": sentence_id,
                "segment_ids": [item.get("id") for item in group],
                "start": final_segment.get("sentence_start", group[0].get("start")),
                "end": final_segment.get("sentence_end", group[-1].get("end")),
                "source_text": final_segment.get("sentence_text", ""),
                "translated_text": _sentence_tts_text(group),
                "sentence_tts_path": final_segment.get("sentence_tts_path", ""),
                "sentence_tts_duration": final_segment.get("sentence_tts_duration"),
                "duration_passed": final_segment.get("duration_passed"),
                "dubbing_score": final_segment.get("dubbing_score"),
                "dubbing_action": final_segment.get("dubbing_action", ""),
                "dubbing_issues": final_segment.get("dubbing_issues", []),
                "dubbing_metrics": final_segment.get("dubbing_metrics", {}),
            }
        )
    return entries


def _quality_segment_entry(segment: SegmentState) -> dict[str, Any]:
    """세그먼트 단위 번역 QC 리포트 항목을 만든다."""
    return {
        "id": segment.get("id"),
        "sentence_id": segment.get("sentence_id"),
        "start": segment.get("start"),
        "end": segment.get("end"),
        "source_text": segment.get("source_text", ""),
        "translated_text": segment.get("translated_text", ""),
        "adjusted_text": segment.get("adjusted_text", ""),
        "translation_attempts": segment.get("translation_attempts", 0),
        "sentence_translation_retry_attempts": segment.get(
            "sentence_translation_retry_attempts",
            0,
        ),
        "duration_rewrite_attempts": segment.get("duration_rewrite_attempts", 0),
        "qc_passed": segment.get("qc_passed"),
        "qc_score": segment.get("qc_score"),
        "qc_action": segment.get("qc_action", ""),
        "qc_issues": segment.get("qc_issues", []),
        "translation_metrics": segment.get("translation_metrics", {}),
        "adjusted_translation_metrics": segment.get(
            "adjusted_translation_metrics",
            {},
        ),
        "revision_feedback": segment.get("revision_feedback", []),
    }


def _chains(state: DubbingState):
    """현재 상태 설정으로 LangChain chain 묶음을 생성한다."""
    return build_agent_chains(
        model=state.get("llm_model", "gemini-2.5-flash"),
        provider=state.get("llm_provider", "vertexai"),
        temperature=float(state.get("llm_temperature", 0.2)),
        google_project=state.get("google_project"),
        google_location=state.get("google_location", "us-central1"),
        google_credentials=state.get("google_credentials"),
    )


def _current_segment(state: DubbingState) -> SegmentState:
    """현재 처리 중인 세그먼트 상태를 반환한다."""
    return state["segments"][state.get("current_segment_index", 0)]


def _source_language(state: DubbingState) -> str:
    """감지된 언어, 사용자 지정 언어, 자동 감지 순서로 원본 언어를 반환한다."""
    return state.get("detected_language") or state.get("source_language") or "auto"


def _segment_duration(segment: SegmentState) -> float:
    """세그먼트 길이를 초 단위로 반환하되 최소값을 보장한다."""
    return max(0.1, float(segment["end"]) - float(segment["start"]))


def _sentence_duration(segment: SegmentState) -> float:
    """문장 그룹 길이를 초 단위로 반환한다."""
    start = float(segment.get("sentence_start", segment["start"]))
    end = float(segment.get("sentence_end", segment["end"]))
    return max(_segment_duration(segment), end - start)


def _segment_available_duration(state: DubbingState, segment: SegmentState) -> float:
    """Return how much time this cue can use before the next cue starts."""
    slot = _segment_slot(state, segment)
    return slot["duration"]


def _translation_timing_duration(state: DubbingState, segment: SegmentState) -> float:
    """Return the timing budget used by translation density checks."""
    group = _sentence_group(state, segment)
    if len(group) > 1:
        return _dubbing_slot_duration(state, segment)
    return _segment_available_duration(state, segment)


def _segment_slot(state: DubbingState, segment: SegmentState) -> dict[str, float]:
    """Return the usable non-overlapping time slot for one source segment."""
    start = float(segment["start"])
    base_end = float(segment["end"])
    next_start = _next_segment_start(state, segment)
    end = _expanded_slot_end(state, base_end, next_start)
    return {
        "start": start,
        "base_end": base_end,
        "end": end,
        "duration": max(_segment_duration(segment), end - start),
        "extension": max(0.0, end - base_end),
        "next_start": next_start if next_start is not None else 0.0,
    }


def _dubbing_slot(state: DubbingState, segment: SegmentState) -> dict[str, float]:
    """Return the usable non-overlapping slot for the current sentence group."""
    group = _sentence_group(state, segment)
    if not group:
        return _segment_slot(state, segment)

    start = float(group[0].get("sentence_start", group[0]["start"]))
    base_end = float(group[-1].get("sentence_end", group[-1]["end"]))
    next_start = _next_sentence_group_start(state, group)
    end = _expanded_slot_end(state, base_end, next_start)
    return {
        "start": start,
        "base_end": base_end,
        "end": end,
        "duration": max(_sentence_duration(segment), end - start),
        "extension": max(0.0, end - base_end),
        "next_start": next_start if next_start is not None else 0.0,
    }


def _dubbing_audio_slot(state: DubbingState, segment: SegmentState) -> dict[str, float]:
    """Return the slot used for final audio composition, including fallback TTS."""
    if (
        not segment.get("sentence_tts_path")
        and segment.get("fallback_sentence_tts_path")
        and "fallback_sentence_tts_slot_start" in segment
        and "fallback_sentence_tts_slot_end" in segment
    ):
        start = float(segment["fallback_sentence_tts_slot_start"])
        end = float(segment["fallback_sentence_tts_slot_end"])
        return {
            "start": start,
            "base_end": float(segment.get("sentence_end", segment["end"])),
            "end": end,
            "duration": max(0.1, end - start),
            "extension": max(
                0.0,
                end - float(segment.get("sentence_end", segment["end"])),
            ),
            "next_start": 0.0,
        }
    return _dubbing_slot(state, segment)


def _dubbing_slot_duration(state: DubbingState, segment: SegmentState) -> float:
    """Return the usable TTS/QC duration for the current sentence group."""
    return _dubbing_slot(state, segment)["duration"]


def _expanded_slot_end(
    state: DubbingState,
    base_end: float,
    next_start: float | None,
) -> float:
    """Extend a slot into available silence without overlapping the next cue."""
    guard = _tts_slot_guard_seconds(state)
    boundary = next_start
    if boundary is None:
        boundary = _state_video_duration(state)
    if boundary is None:
        return base_end

    candidate = max(base_end, float(boundary) - guard)
    return max(base_end, candidate)


def _next_segment_start(
    state: DubbingState,
    segment: SegmentState,
) -> float | None:
    """Return the next source segment start time after the current segment."""
    current_id = segment.get("id")
    segments = state.get("segments", [])
    for idx, item in enumerate(segments):
        if item.get("id") != current_id:
            continue
        if idx + 1 < len(segments):
            return float(segments[idx + 1]["start"])
        return None
    return None


def _next_sentence_group_start(
    state: DubbingState,
    group: list[SegmentState],
) -> float | None:
    """Return the first segment start after the current sentence group."""
    if not group:
        return None

    group_ids = {item.get("id") for item in group}
    for item in state.get("segments", []):
        if item.get("id") in group_ids:
            continue
        if float(item["start"]) >= float(group[-1]["end"]):
            return float(item["start"])
    return None


def _state_video_duration(state: DubbingState) -> float | None:
    """Return video duration, caching it in metadata when available."""
    metadata = state.setdefault("metadata", {})
    cached = metadata.get("video_duration_seconds")
    if cached is not None:
        try:
            return float(cached)
        except (TypeError, ValueError):
            pass

    video_path = state.get("input_video_path")
    if not video_path:
        return None
    try:
        duration = float(get_video_duration(video_path))
    except Exception as exc:  # noqa: BLE001
        metadata["video_duration_read_error"] = str(exc)
        return None
    metadata["video_duration_seconds"] = duration
    return duration


def _tts_slot_guard_seconds(state: DubbingState) -> float:
    """Return the guard gap kept before the next cue starts."""
    try:
        return max(0.0, float(state.get("tts_slot_guard_seconds", TTS_SLOT_GUARD_SECONDS)))
    except (TypeError, ValueError):
        return TTS_SLOT_GUARD_SECONDS


def _segment_context(state: DubbingState) -> str:
    """현재 세그먼트 번역에 필요한 문장/주변 원문 문맥을 만든다."""
    segments = state.get("segments", [])
    if not segments:
        return ""

    current_idx = state.get("current_segment_index", 0)
    current = segments[current_idx]
    sentence_id = current.get("sentence_id")
    same_sentence_indices = [
        idx for idx, item in enumerate(segments) if item.get("sentence_id") == sentence_id
    ]
    start_idx = max(0, min(same_sentence_indices + [current_idx]) - SEGMENT_CONTEXT_RADIUS)
    end_idx = min(
        len(segments),
        max(same_sentence_indices + [current_idx]) + SEGMENT_CONTEXT_RADIUS + 1,
    )

    lines = []
    for idx in range(start_idx, end_idx):
        item = segments[idx]
        marker = "CURRENT" if idx == current_idx else "CONTEXT"
        scope = "same_sentence" if item.get("sentence_id") == sentence_id else "nearby"
        lines.append(
            f"{marker} [{item['id']}] {item['start']:.2f}-{item['end']:.2f}s "
            f"{scope}: {item['source_text']}"
        )
    return "\n".join(lines)


def _global_transcript_context(state: DubbingState) -> str:
    """전체 transcript 흐름을 길이 제한 안에서 반환한다."""
    segments = state.get("segments", [])
    if not segments:
        return ""

    lines = [f"[{item['id']}] {item['source_text']}" for item in segments]
    full_context = "\n".join(lines)
    if len(full_context) <= GLOBAL_CONTEXT_MAX_CHARS:
        return full_context

    current_idx = state.get("current_segment_index", 0)
    head = "\n".join(lines[:20])
    start = max(0, current_idx - 20)
    end = min(len(lines), current_idx + 21)
    current_window = "\n".join(lines[start:end])
    tail = "\n".join(lines[-20:])
    compact = (
        f"{head}\n"
        "[... transcript middle omitted ...]\n"
        f"{current_window}\n"
        "[... transcript middle omitted ...]\n"
        f"{tail}"
    )
    return compact[:GLOBAL_CONTEXT_MAX_CHARS]


def _neighbor_translations(state: DubbingState) -> str:
    """현재 문맥 안에서 이미 생성된 주변 번역을 반환한다."""
    segments = state.get("segments", [])
    current_idx = state.get("current_segment_index", 0)
    if not segments:
        return "없음"

    start_idx = max(0, current_idx - SEGMENT_CONTEXT_RADIUS)
    end_idx = min(len(segments), current_idx + SEGMENT_CONTEXT_RADIUS + 1)
    lines = []
    for idx in range(start_idx, end_idx):
        if idx == current_idx:
            continue
        item = segments[idx]
        translated = item.get("adjusted_text") or item.get("translated_text")
        if translated:
            lines.append(f"[{item['id']}] {translated}")
    return "\n".join(lines) if lines else "없음"


def _global_translation_review_payload(state: DubbingState) -> dict[str, Any]:
    """전체 번역 리뷰 chain에 전달할 원문/번역 transcript payload를 만든다."""
    return {
        "source_language": _source_language(state),
        "target_language": state.get("target_language", "ko"),
        "summary": state.get("summary", ""),
        "domain": state.get("domain", ""),
        "glossary": state.get("glossary", {}),
        "translation_style": state.get("translation_style", ""),
        "source_transcript": _review_source_transcript(state),
        "translated_transcript": _review_translated_transcript(state),
        "sentence_review_transcript": _review_sentence_transcript(state),
    }


def _review_source_transcript(state: DubbingState) -> str:
    """전체 리뷰용 원문 transcript를 만든다."""
    return "\n".join(
        f"[{segment['id']}] {segment['source_text']}"
        for segment in state.get("segments", [])
    )


def _review_translated_transcript(state: DubbingState) -> str:
    """전체 리뷰용 번역 transcript를 만든다."""
    lines = []
    for segment in state.get("segments", []):
        translated = segment.get("adjusted_text") or segment.get("translated_text", "")
        lines.append(f"[{segment['id']}] {translated}")
    return "\n".join(lines)


def _review_sentence_transcript(state: DubbingState) -> str:
    """전체 리뷰가 문장 단위 흐름과 더빙 시간 부담을 함께 보도록 문맥을 만든다."""
    lines: list[str] = []
    seen_sentence_ids: set[Any] = set()
    for segment in state.get("segments", []):
        sentence_id = segment.get("sentence_id", segment.get("id"))
        if sentence_id in seen_sentence_ids:
            continue
        seen_sentence_ids.add(sentence_id)

        group = _sentence_group(state, segment)
        if not group:
            continue
        final_segment = group[-1]
        source_text = final_segment.get(
            "sentence_text",
            " ".join(item.get("source_text", "") for item in group),
        )
        translated_text = _sentence_tts_text(group)
        metrics = final_segment.get("dubbing_metrics", {}).get("sentence", {})
        tts_seconds = final_segment.get("sentence_tts_duration")
        lines.append(
            "sentence "
            f"{sentence_id} segments={ [item.get('id') for item in group] } "
            f"{final_segment.get('sentence_start', group[0].get('start')):.2f}-"
            f"{final_segment.get('sentence_end', group[-1].get('end')):.2f}s\n"
            f"source: {source_text}\n"
            f"translation: {translated_text}\n"
            f"dubbing: {_review_sentence_dubbing_context(tts_seconds, metrics)}"
        )
    return "\n\n".join(lines) if lines else "없음"


def _review_sentence_dubbing_context(
    tts_seconds: Any,
    metrics: dict[str, Any],
) -> str:
    """문장 리뷰 문맥에 표시할 더빙 길이/속도 요약을 만든다."""
    if not isinstance(metrics, dict) or not metrics:
        if tts_seconds is None:
            return "not checked"
        return f"tts_seconds={tts_seconds}"

    parts = [
        f"tts_seconds={metrics.get('tts_seconds', tts_seconds)}",
        f"allowed_max_seconds={metrics.get('allowed_max_seconds')}",
        f"actual_units_per_second={metrics.get('actual_units_per_second')}",
        f"required_units_per_second={metrics.get('required_units_per_second')}",
        f"max_units_per_second={metrics.get('max_units_per_second')}",
        f"speed_passed={metrics.get('speed_passed')}",
        f"heuristic_passed={metrics.get('heuristic_passed')}",
    ]
    issues = metrics.get("issues", [])
    if issues:
        parts.append(f"issues={issues}")
    return ", ".join(parts)


def _global_review_issues(review: Any) -> list[dict[str, Any]]:
    """structured review 응답에서 issue 목록을 안정적인 dict 목록으로 정규화한다."""
    raw_issues = _field(review, "issues", [])
    normalized: list[dict[str, Any]] = []
    for issue in raw_issues or []:
        data = _to_plain_data(issue)
        if not isinstance(data, dict):
            continue
        try:
            segment_id = int(data.get("segment_id"))
        except (TypeError, ValueError):
            continue
        issue_text = str(data.get("issue", "")).strip()
        suggested = str(data.get("suggested_translation", "")).strip()
        if not issue_text and not suggested:
            continue
        normalized.append(
            {
                "segment_id": segment_id,
                "issue": issue_text,
                "suggested_translation": suggested,
            }
        )
    return normalized


def _apply_global_translation_suggestions(
    state: DubbingState,
    issues: list[dict[str, Any]],
) -> list[int]:
    """전체 리뷰의 수정 제안을 적용하고 재처리할 세그먼트 인덱스를 반환한다."""
    changed_indices: list[int] = []
    for issue in issues:
        suggested = str(issue.get("suggested_translation", "")).strip()
        if not suggested:
            continue
        segment_idx = _find_segment_index_by_id(state, int(issue["segment_id"]))
        if segment_idx is None:
            continue
        segment = state["segments"][segment_idx]
        current = segment.get("translated_text", "")
        if suggested == current and suggested == segment.get("adjusted_text", ""):
            continue
        segment["translated_text"] = suggested
        segment["adjusted_text"] = suggested
        feedback = _global_review_issue_text(issue)
        segment["revision_feedback"] = _bounded_revision_feedback(
            list(segment.get("revision_feedback", []))
            + [f"Global translation review: {feedback}"]
        )
        _reset_translation_qc_artifacts(segment)
        _clear_sentence_tts_artifacts(state, segment)
        changed_indices.append(segment_idx)
    return sorted(set(changed_indices))


def _global_review_reprocess_segment_ids(
    state: DubbingState,
    changed_indices: list[int],
) -> list[int]:
    """수정된 세그먼트부터 해당 문장 끝까지 재처리할 id 목록을 만든다."""
    reprocess_indices: set[int] = set()
    segments = state.get("segments", [])
    for changed_idx in changed_indices:
        if changed_idx < 0 or changed_idx >= len(segments):
            continue
        group = _sentence_group(state, segments[changed_idx])
        group_ids = {group_item.get("id") for group_item in group}
        group_indices = [
            idx
            for idx, item in enumerate(segments)
            if item.get("id") in group_ids and idx >= changed_idx
        ]
        reprocess_indices.update(group_indices)

    return [
        int(segments[idx]["id"])
        for idx in sorted(reprocess_indices)
        if "id" in segments[idx]
    ]


def _global_review_retry_budget_segment_ids(
    state: DubbingState,
    changed_indices: list[int],
) -> list[int]:
    """전체 리뷰 수정으로 다시 번역할 수 있어야 하는 문장 세그먼트 id를 만든다."""
    reset_indices: set[int] = set()
    segments = state.get("segments", [])
    for changed_idx in changed_indices:
        if changed_idx < 0 or changed_idx >= len(segments):
            continue
        group = _sentence_group(state, segments[changed_idx])
        group_ids = {group_item.get("id") for group_item in group}
        reset_indices.update(
            idx for idx, item in enumerate(segments) if item.get("id") in group_ids
        )

    return [
        int(segments[idx]["id"])
        for idx in sorted(reset_indices)
        if "id" in segments[idx]
    ]


def _reset_global_review_retry_budgets(
    state: DubbingState,
    segment_ids: list[int],
) -> None:
    """전체 리뷰가 바꾼 문장은 QC 재처리 중 재번역/재작성 예산을 새로 쓴다."""
    for segment_id in segment_ids:
        segment_idx = _find_segment_index_by_id(state, int(segment_id))
        if segment_idx is None:
            continue
        segment = state["segments"][segment_idx]
        segment["translation_attempts"] = 0
        segment["total_translation_attempts"] = 0
        segment["sentence_translation_retry_attempts"] = 0
        segment["duration_rewrite_attempts"] = 0


def _append_global_translation_review_history(
    metadata: dict[str, Any],
    review_metadata: dict[str, Any],
) -> None:
    """전체 번역 리뷰 실행 이력을 리포트용 metadata에 누적한다."""
    history = metadata.setdefault("global_translation_review_history", [])
    history.append(deepcopy(_to_plain_data(review_metadata)))


def _global_review_issue_text(issue: Any) -> str:
    """전역 번역 리뷰 issue를 사람이 읽을 수 있는 문자열로 만든다."""
    if isinstance(issue, dict):
        segment_id = issue.get("segment_id", "?")
        text = issue.get("issue") or "번역 이슈"
        return f"세그먼트 {segment_id}: {text}"
    return str(issue)


def _global_review_history_text(item: Any) -> str:
    """전역 번역 리뷰 이력 항목을 Markdown 한 줄로 요약한다."""
    if not isinstance(item, dict):
        return str(item)

    parts = [
        f"시도 {item.get('attempts', '?')}",
        f"통과={_display_bool(item.get('passed'))}",
        f"조치={_display_action(item.get('action', ''))}",
        f"점수={_display_value(item.get('score'))}",
    ]
    changed_segment_ids = item.get("changed_segment_ids", [])
    if changed_segment_ids:
        parts.append(f"변경 세그먼트={changed_segment_ids}")
    reprocess_segment_ids = item.get("reprocess_segment_ids", [])
    if reprocess_segment_ids:
        parts.append(f"재처리 세그먼트={reprocess_segment_ids}")
    retry_budget_reset_ids = item.get("retry_budget_reset_segment_ids", [])
    if retry_budget_reset_ids:
        parts.append(f"재시도 예산 초기화 세그먼트={retry_budget_reset_ids}")

    issues = item.get("issues", [])
    issue_text = "; ".join(_global_review_issue_text(issue) for issue in issues)
    parts.append(f"이슈={issue_text or '없음'}")
    return ", ".join(parts)


def _find_segment_index_by_id(state: DubbingState, segment_id: int) -> int | None:
    """세그먼트 id로 인덱스를 찾고 없으면 None을 반환한다."""
    for idx, item in enumerate(state.get("segments", [])):
        if item.get("id") == segment_id:
            return idx
    return None


def _revision_feedback_text(segment: SegmentState) -> str:
    """재번역 프롬프트에 전달할 이전 QC 피드백을 만든다."""
    feedback: list[str] = list(segment.get("revision_feedback", []))
    if segment.get("qc_issues"):
        feedback.append("Translation QC: " + "; ".join(segment["qc_issues"]))
    if segment.get("dubbing_issues"):
        feedback.append("Dubbing QC: " + "; ".join(segment["dubbing_issues"]))

    metrics = segment.get("dubbing_metrics", {}).get("sentence") or segment.get(
        "dubbing_metrics", {}
    ).get("segment")
    if isinstance(metrics, dict) and metrics:
        feedback.append(
            "Dubbing metrics: "
            f"duration={metrics.get('tts_seconds')}s, "
            f"allowed={metrics.get('allowed_max_seconds')}s, "
            f"actual_units_per_second={metrics.get('actual_units_per_second')}, "
            f"required_units_per_second={metrics.get('required_units_per_second')}, "
            f"units_per_second={metrics.get('units_per_second')} "
            f"(max={metrics.get('max_units_per_second')})"
        )

    cleaned = _bounded_revision_feedback(feedback)
    return "\n".join(cleaned) if cleaned else "없음"


def _reset_segment_after_new_translation(
    segment: SegmentState,
    revision_feedback: str,
) -> None:
    """새 번역을 만들 때 이전 QC/TTS 결과를 지우고 피드백만 보존한다."""
    if revision_feedback != "없음":
        segment["revision_feedback"] = _bounded_revision_feedback(
            list(segment.get("revision_feedback", []))
            + revision_feedback.splitlines()
        )

    for key in (
        "adjusted_text",
        "tts_path",
        "qc_score",
        "qc_passed",
        "qc_action",
        "translation_metrics",
        "adjusted_translation_metrics",
        "tts_duration",
        "duration_passed",
        "dubbing_score",
        "dubbing_action",
        "dubbing_metrics",
        "translation_budget_exhausted",
    ):
        segment.pop(key, None)
    segment["qc_issues"] = []
    segment["dubbing_issues"] = []
    segment["duration_rewrite_attempts"] = 0


def _bounded_revision_feedback(items: list[Any]) -> list[str]:
    """Return compact retry feedback so prompts cannot grow across retries."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = " ".join(str(item).split())
        if not text:
            continue
        if len(text) > REVISION_FEEDBACK_ITEM_MAX_CHARS:
            text = text[: REVISION_FEEDBACK_ITEM_MAX_CHARS - 3].rstrip() + "..."
        if text in seen:
            continue
        seen.add(text)
        cleaned.append(text)

    cleaned = cleaned[-REVISION_FEEDBACK_MAX_ITEMS:]
    while (
        len("\n".join(cleaned)) > REVISION_FEEDBACK_TOTAL_MAX_CHARS
        and len(cleaned) > 1
    ):
        cleaned.pop(0)
    return cleaned


def _reset_translation_qc_artifacts(segment: SegmentState) -> None:
    """번역문을 직접 교체할 때 이전 QC와 길이 재작성 결과를 지운다."""
    for key in (
        "qc_score",
        "qc_passed",
        "qc_action",
        "translation_metrics",
        "adjusted_translation_metrics",
        "duration_rewrite_attempts",
        "translation_budget_exhausted",
    ):
        segment.pop(key, None)
    segment["qc_issues"] = []
    segment["duration_rewrite_attempts"] = 0


def _translation_quality_metrics(
    source_text: str,
    translated_text: str,
    duration_seconds: float,
    target_language: str,
) -> dict[str, Any]:
    """번역 QC 도구를 노드 내부의 강제 휴리스틱으로 실행한다."""
    return analyze_translation_quality.invoke(
        {
            "source_text": source_text,
            "translated_text": translated_text,
            "duration_seconds": duration_seconds,
            "target_language": target_language,
        }
    )


def _sentence_translation_quality_metrics(
    state: DubbingState,
    segment: SegmentState,
    target_language: str,
) -> dict[str, Any] | None:
    """문장 그룹 전체의 번역 발화량과 기본 품질을 검사한다."""
    group = _sentence_group(state, segment)
    if len(group) <= 1 or group[-1].get("id") != segment.get("id"):
        return None

    translated_text = _sentence_tts_text(group)
    if not translated_text.strip():
        return None

    source_text = str(
        segment.get("sentence_text") or " ".join(item["source_text"] for item in group)
    )
    metrics = _translation_quality_metrics(
        source_text,
        translated_text,
        _dubbing_slot_duration(state, segment),
        target_language,
    )
    metrics["sentence_id"] = segment.get("sentence_id")
    metrics["segment_ids"] = [item.get("id") for item in group]
    if metrics.get("issues"):
        metrics["issues"] = [
            f"문장 단위 번역: {issue}" for issue in metrics["issues"]
        ]
    return metrics


def _combined_translation_quality_metrics(
    segment_metrics: dict[str, Any],
    sentence_metrics: dict[str, Any] | None,
) -> dict[str, Any]:
    """세그먼트 및 문장 단위 번역 QC 결과를 하나의 판단으로 결합한다."""
    combined = dict(segment_metrics)
    combined["segment"] = segment_metrics

    issues = list(segment_metrics.get("issues", []))
    scores = [float(segment_metrics.get("heuristic_score", 0.0))]
    passed = bool(segment_metrics.get("heuristic_passed", False))

    if sentence_metrics is not None:
        combined["sentence"] = sentence_metrics
        issues.extend(sentence_metrics.get("issues", []))
        scores.append(float(sentence_metrics.get("heuristic_score", 0.0)))
        passed = passed and bool(sentence_metrics.get("heuristic_passed", False))

    combined["issues"] = _unique_strings(issues)
    combined["heuristic_score"] = min(scores) if scores else 0.0
    combined["heuristic_passed"] = passed
    return combined


def _dubbing_quality_metrics(
    source_seconds: float,
    tts_seconds: float,
    tolerance: float,
    text: str,
    target_language: str,
) -> dict[str, Any]:
    """더빙 QC 도구를 노드 내부의 강제 휴리스틱으로 실행한다."""
    return analyze_dubbing_quality.invoke(
        {
            "source_seconds": source_seconds,
            "tts_seconds": tts_seconds,
            "tolerance": tolerance,
            "text": text,
            "target_language": target_language,
        }
    )


def _tts_alignment_metrics(
    state: DubbingState,
    segment: SegmentState,
    tts_seconds: float,
    slot: dict[str, float],
) -> dict[str, Any]:
    """Check whether TTS starts on time and leaves an abnormal gap before the next cue."""
    next_start = slot.get("next_start") or 0.0
    if next_start <= 0.0:
        return {
            "alignment_passed": True,
            "gap_to_next_seconds": None,
            "allowed_gap_to_next_seconds": None,
            "issues": [],
        }

    speech_end = float(slot["start"]) + max(0.0, float(tts_seconds))
    actual_gap = max(0.0, float(next_start) - speech_end)
    original_gap = max(0.0, float(next_start) - float(slot["base_end"]))
    allowed_gap = max(
        TTS_ABNORMAL_GAP_MIN_SECONDS,
        original_gap + TTS_ABNORMAL_GAP_GRACE_SECONDS,
    )
    issues: list[str] = []
    if actual_gap > allowed_gap:
        issues.append(
            "다음 발화 전 공백이 원본보다 비정상적으로 깁니다 "
            f"({actual_gap:.2f}초 > {allowed_gap:.2f}초)."
        )

    return {
        "alignment_passed": not issues,
        "gap_to_next_seconds": round(actual_gap, 3),
        "original_gap_to_next_seconds": round(original_gap, 3),
        "allowed_gap_to_next_seconds": round(allowed_gap, 3),
        "speech_end_seconds": round(speech_end, 3),
        "next_start_seconds": round(float(next_start), 3),
        "issues": issues,
    }


def _merge_dubbing_alignment_metrics(
    metrics: dict[str, Any],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    """Merge post-TTS timing alignment checks into dubbing QC metrics."""
    merged = dict(metrics)
    alignment_issues = list(alignment.get("issues", []))
    merged["alignment"] = alignment
    merged["issues"] = _unique_strings(
        list(merged.get("issues", [])) + alignment_issues
    )
    merged["alignment_passed"] = bool(alignment.get("alignment_passed", True))
    merged["heuristic_passed"] = bool(merged.get("heuristic_passed", False)) and bool(
        alignment.get("alignment_passed", True)
    )
    if alignment_issues:
        merged["heuristic_score"] = min(float(merged.get("heuristic_score", 0.0)), 0.82)
        merged["action_hint"] = "rewrite"
    return merged


def _sentence_dubbing_metrics(
    state: DubbingState,
    segment: SegmentState,
    tolerance: float,
    target_language: str,
) -> dict[str, Any] | None:
    """문장 그룹 전체의 TTS 길이와 속도를 검사한다."""
    group = _sentence_group(state, segment)
    if len(group) <= 1 or group[-1].get("id") != segment.get("id"):
        return None

    durations = []
    text_parts = []
    for item in group:
        if "tts_duration" not in item:
            return None
        durations.append(float(item["tts_duration"]))
        text_parts.append(item.get("adjusted_text") or item.get("translated_text", ""))

    metrics = _dubbing_quality_metrics(
        _dubbing_slot_duration(state, segment),
        sum(durations),
        tolerance,
        " ".join(part for part in text_parts if part),
        target_language,
    )
    metrics["sentence_id"] = segment.get("sentence_id")
    metrics["segment_ids"] = [item.get("id") for item in group]
    if metrics.get("issues"):
        metrics["issues"] = [f"문장 단위: {issue}" for issue in metrics["issues"]]
    return metrics


def _sentence_group(state: DubbingState, segment: SegmentState) -> list[SegmentState]:
    """현재 세그먼트와 같은 문장 그룹의 세그먼트를 반환한다."""
    sentence_id = segment.get("sentence_id")
    return [
        item
        for item in state.get("segments", [])
        if item.get("sentence_id") == sentence_id
    ]


def _dubbing_action(
    agent_action: str,
    segment_metrics: dict[str, Any],
    sentence_metrics: dict[str, Any] | None,
) -> str:
    """더빙 실패 원인에 따라 재작성 또는 번역 재시도를 선택한다."""
    metrics = [segment_metrics]
    if sentence_metrics is not None:
        metrics.append(sentence_metrics)

    if any(not item.get("speed_passed", True) for item in metrics):
        return "retry_translation"
    if any(not item.get("heuristic_passed", False) for item in metrics):
        return "rewrite"
    return str(agent_action)


def _duration_issue(segment: SegmentState) -> str:
    """duration rewrite chain에 전달할 길이 문제 설명을 만든다."""
    issues = "; ".join(segment.get("dubbing_issues", []))
    metrics = segment.get("dubbing_metrics", {}).get("sentence") or segment.get(
        "dubbing_metrics", {}
    ).get("segment")
    metric_details = ""
    if isinstance(metrics, dict) and metrics:
        metric_details = (
            f" TTS 길이={metrics.get('tts_seconds')}초, "
            f"허용 길이={metrics.get('allowed_max_seconds')}초, "
            f"실제 발화 단위/초={metrics.get('actual_units_per_second')}, "
            f"슬롯 필요 발화 단위/초={metrics.get('required_units_per_second')}, "
            f"판정 발화 단위/초={metrics.get('units_per_second')} "
            f"(상한={metrics.get('max_units_per_second')})."
        )

    if "tts_duration" not in segment:
        return (
            "문장 단위 흐름을 유지하면서 원본 timestamp 안에서 자연스럽게 말할 수 "
            f"있도록 간결하게 조정해야 합니다. {issues}{metric_details}"
        ).strip()

    details = (
        f"TTS 길이가 {segment['tts_duration']:.2f}초입니다. "
        f"원본 세그먼트 길이는 {_segment_duration(segment):.2f}초이고 "
        f"문장 단위 길이는 {_sentence_duration(segment):.2f}초입니다."
    )
    return f"{details} {issues}{metric_details}".strip()


def _unique_strings(values) -> list[str]:
    """순서를 유지하며 빈 문자열과 중복 문자열을 제거한다."""
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique.append(text)
    return unique


def _total_translation_attempts(segment: SegmentState) -> int:
    """Return the highest known translation attempt count for a segment."""
    return max(
        int(segment.get("translation_attempts", 0) or 0),
        int(segment.get("total_translation_attempts", 0) or 0),
    )


def _max_translation_attempts(state: DubbingState) -> int:
    """Return the configured per-segment translation attempt cap."""
    return int(state.get("max_translation_attempts", 3) or 3)


def _field(result: Any, name: str, default: Any) -> Any:
    """dict 또는 객체 형태의 structured output에서 필드를 안전하게 읽는다."""
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def _write_context_artifacts(state: DubbingState, transcript: str) -> None:
    """Persist the global transcript context generated after subtitle/STT loading."""
    try:
        if not state.get("output_dir"):
            return
        output_dir = ensure_dir(state["output_dir"])
        json_path = os.path.join(output_dir, CONTEXT_SUMMARY_JSON)
        markdown_path = os.path.join(output_dir, CONTEXT_SUMMARY_MARKDOWN)
        data = _context_artifact_data(state, transcript)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")

        with open(markdown_path, "w", encoding="utf-8") as f:
            f.write(_context_artifact_markdown(data))

        state["context_summary_json_path"] = json_path
        state["context_summary_markdown_path"] = markdown_path
        metadata = state.setdefault("metadata", {})
        metadata["context_summary_json_path"] = json_path
        metadata["context_summary_markdown_path"] = markdown_path
        _log_progress(
            state,
            "build_context",
            f"문맥 요약 기록 완료: {markdown_path}",
        )
    except Exception as exc:  # noqa: BLE001
        state.setdefault("metadata", {})["context_artifact_write_error"] = str(exc)


def _context_artifact_data(
    state: DubbingState,
    transcript: str,
) -> dict[str, Any]:
    """Build serializable context summary artifact data."""
    return {
        "job_id": state.get("job_id", ""),
        "source_url": state.get("source_url", ""),
        "source_language": _source_language(state),
        "target_language": state.get("target_language", "ko"),
        "summary": state.get("summary", ""),
        "domain": state.get("domain", ""),
        "glossary": state.get("glossary", {}),
        "translation_style": state.get("translation_style", ""),
        "transcript": transcript,
        "segments": [
            {
                "id": segment.get("id"),
                "start": segment.get("start"),
                "end": segment.get("end"),
                "sentence_id": segment.get("sentence_id"),
                "source_text": segment.get("source_text", ""),
            }
            for segment in state.get("segments", [])
        ],
    }


def _context_artifact_markdown(data: dict[str, Any]) -> str:
    """Render the context summary artifact in a human-readable Markdown format."""
    glossary = data.get("glossary", {}) or {}
    glossary_lines = (
        [f"- {source}: {target}" for source, target in glossary.items()]
        if isinstance(glossary, dict) and glossary
        else ["- 없음"]
    )
    transcript = str(data.get("transcript", "")).strip() or "없음"
    lines = [
        "# 전체 문맥 요약",
        "",
        f"- 작업 ID: {data.get('job_id', '')}",
        f"- 원본 언어: {data.get('source_language', '')}",
        f"- 대상 언어: {data.get('target_language', '')}",
        "",
        "## 요약",
        str(data.get("summary", "")).strip() or "없음",
        "",
        "## 분야",
        str(data.get("domain", "")).strip() or "없음",
        "",
        "## 번역 스타일",
        str(data.get("translation_style", "")).strip() or "없음",
        "",
        "## 용어집",
        *glossary_lines,
        "",
        "## 원문 Transcript",
        "",
        _markdown_code_block("text", transcript),
        "",
    ]
    return "\n".join(lines)


def _invoke_llm(
    runnable: Any,
    payload: Any,
    operation: str,
    state: DubbingState | None = None,
    *,
    max_attempts: int | None = None,
    delay_seconds: float | None = None,
    max_delay_seconds: float | None = None,
) -> Any:
    """Invoke an LLM runnable, pausing and retrying when provider quota is exhausted."""
    if max_attempts is None:
        max_attempts = _env_int(
            "DUBBING_AGENT_LLM_RESOURCE_EXHAUSTED_RETRIES",
            DEFAULT_RESOURCE_EXHAUSTED_RETRIES,
        )
    if delay_seconds is None:
        delay_seconds = _env_float(
            "DUBBING_AGENT_LLM_RESOURCE_EXHAUSTED_DELAY_SECONDS",
            DEFAULT_RESOURCE_EXHAUSTED_DELAY_SECONDS,
        )
    if max_delay_seconds is None:
        max_delay_seconds = _env_float(
            "DUBBING_AGENT_LLM_RESOURCE_EXHAUSTED_MAX_DELAY_SECONDS",
            DEFAULT_RESOURCE_EXHAUSTED_MAX_DELAY_SECONDS,
        )

    attempt = 1
    while True:
        trace_callback = _LLMTraceCallback()
        try:
            result = _invoke_runnable_with_trace_callback(
                runnable,
                payload,
                trace_callback,
            )
            _record_llm_call(
                state=state,
                operation=operation,
                payload=payload,
                response=result,
                attempts=attempt,
                rendered_prompts=trace_callback.records(),
            )
            return result
        except Exception as exc:
            if not _is_resource_exhausted_error(exc) or attempt >= max_attempts:
                _record_llm_call(
                    state=state,
                    operation=operation,
                    payload=payload,
                    response=None,
                    attempts=attempt,
                    error=exc,
                    rendered_prompts=trace_callback.records(),
                )
                raise

            sleep_seconds = min(
                delay_seconds * (2 ** (attempt - 1)),
                max_delay_seconds,
            )
            print(
                "LLM 호출이 일시적으로 실패하여 "
                f"{sleep_seconds:.0f}초 후 재시도합니다. "
                f"[{attempt}/{max_attempts}] 작업={operation}"
            )
            time.sleep(sleep_seconds)
            attempt += 1


def _invoke_optional_llm(
    runnable: Any,
    payload: Any,
    operation: str,
    state: DubbingState | None = None,
) -> Any:
    """Invoke a non-essential QC/review LLM with a short retry budget."""
    return _invoke_llm(
        runnable,
        payload,
        operation,
        state,
        max_attempts=_env_int(
            "DUBBING_AGENT_OPTIONAL_LLM_RETRIES",
            DEFAULT_OPTIONAL_LLM_RETRIES,
        ),
        delay_seconds=_env_float(
            "DUBBING_AGENT_OPTIONAL_LLM_DELAY_SECONDS",
            DEFAULT_OPTIONAL_LLM_DELAY_SECONDS,
        ),
        max_delay_seconds=_env_float(
            "DUBBING_AGENT_OPTIONAL_LLM_MAX_DELAY_SECONDS",
            DEFAULT_OPTIONAL_LLM_MAX_DELAY_SECONDS,
        ),
    )


def _invoke_runnable_with_trace_callback(
    runnable: Any,
    payload: Any,
    trace_callback: _LLMTraceCallback,
) -> Any:
    """Invoke a runnable with callbacks, falling back for simple test doubles."""
    try:
        return runnable.invoke(payload, config={"callbacks": [trace_callback]})
    except TypeError as exc:
        if _is_invoke_config_type_error(exc):
            return runnable.invoke(payload)
        raise


def _is_invoke_config_type_error(exc: TypeError) -> bool:
    """Return whether a TypeError only means the runnable has no config parameter."""
    message = str(exc).lower()
    if "config" not in message:
        return False
    return (
        "unexpected keyword" in message
        or "got an unexpected" in message
        or "positional" in message
        or "keyword argument" in message
    )


def _record_llm_call(
    *,
    state: DubbingState | None,
    operation: str,
    payload: Any,
    response: Any,
    attempts: int,
    rendered_prompts: dict[str, Any] | None = None,
    error: BaseException | None = None,
) -> None:
    """Append an LLM input/output trace to the job output directory."""
    if state is None or not state.get("output_dir"):
        return

    try:
        output_dir = ensure_dir(state["output_dir"])
        jsonl_path = os.path.join(output_dir, LLM_TRACE_JSONL)
        markdown_path = os.path.join(output_dir, LLM_TRACE_MARKDOWN)
        metadata = state.setdefault("metadata", {})
        call_index = _next_llm_trace_index(metadata, jsonl_path)
        record: dict[str, Any] = {
            "index": call_index,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "operation": operation,
            "status": "error" if error is not None else "success",
            "attempts": attempts,
            "llm_provider": state.get("llm_provider", ""),
            "llm_model": state.get("llm_model", ""),
            "payload": _trace_payload(payload),
            "response": _trace_response(response),
        }
        if _has_rendered_prompts(rendered_prompts):
            record["rendered_prompts"] = _trace_rendered_prompts(rendered_prompts)
        if error is not None:
            record["error"] = {
                "type": error.__class__.__name__,
                "message": str(error),
            }

        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str))
            f.write("\n")

        _append_llm_trace_markdown(markdown_path, record)

        metadata["llm_trace_count"] = call_index
        metadata["llm_trace_jsonl_path"] = jsonl_path
        metadata["llm_trace_markdown_path"] = markdown_path
        state["llm_trace_jsonl_path"] = jsonl_path
        state["llm_trace_markdown_path"] = markdown_path
    except Exception as exc:  # noqa: BLE001
        state.setdefault("metadata", {})["llm_trace_write_error"] = str(exc)


def _record_llm_fallback(
    state: DubbingState,
    operation: str,
    error: BaseException,
) -> None:
    """Record that a non-essential LLM QC/review call fell back to local logic."""
    metadata = state.setdefault("metadata", {})
    fallbacks = metadata.setdefault("llm_fallbacks", [])
    fallbacks.append(
        {
            "operation": operation,
            "error_type": error.__class__.__name__,
            "error": _compact_trace_text(str(error)),
        }
    )
    node = operation
    if operation.startswith("translation_qc"):
        node = "translation_qc"
    elif operation.startswith("tts_duration_check"):
        node = "tts_duration_check"
    _log_progress(state, node, "LLM 호출 실패, 로컬 검사로 대체")


def _next_llm_trace_index(metadata: dict[str, Any], jsonl_path: str) -> int:
    """Return the next trace index, using existing files as a fallback."""
    try:
        current = int(metadata.get("llm_trace_count", 0) or 0)
    except (TypeError, ValueError):
        current = 0

    if current <= 0 and os.path.exists(jsonl_path):
        with open(jsonl_path, encoding="utf-8") as f:
            current = sum(1 for line in f if line.strip())
    return current + 1


def _append_llm_trace_markdown(path: str, record: dict[str, Any]) -> None:
    """Append one LLM trace entry to the Markdown trace document."""
    needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", encoding="utf-8") as f:
        if needs_header:
            f.write("# LLM 호출 기록\n\n")
            f.write(
                "번역, 품질 검사, 길이 조정, 전체 리뷰 등에 사용한 LLM 입력과 "
                "응답을 시간순으로 기록합니다.\n\n"
            )

        status = "실패" if record.get("status") == "error" else "성공"
        provider = record.get("llm_provider") or "미지정"
        model = record.get("llm_model") or "미지정"
        f.write(
            f"## {int(record.get('index', 0)):04d}. "
            f"{record.get('operation', '')}\n\n"
        )
        f.write(f"- 시간: {record.get('timestamp', '')}\n")
        f.write(f"- 상태: {status}\n")
        f.write(f"- 시도 횟수: {record.get('attempts', '')}\n")
        f.write(f"- Provider: {provider}\n")
        f.write(f"- Model: {model}\n")
        if record.get("error"):
            error = record["error"]
            f.write(
                f"- 오류: {error.get('type', '')}: {error.get('message', '')}\n"
            )
        if record.get("rendered_prompts"):
            f.write("\n### 렌더링된 프롬프트\n\n")
            f.write(
                _markdown_code_block(
                    "json",
                    _json_dumps(record.get("rendered_prompts")),
                )
            )
            f.write("\n\n")

        f.write("\n### 프롬프트/입력 Payload\n\n")
        f.write(_markdown_code_block("json", _json_dumps(record.get("payload"))))
        f.write("\n\n### 응답\n\n")
        f.write(_markdown_code_block("json", _json_dumps(record.get("response"))))
        f.write("\n\n")


def _has_rendered_prompts(rendered_prompts: dict[str, Any] | None) -> bool:
    """Return whether callback prompt records contain any model input."""
    if not rendered_prompts:
        return False
    return bool(rendered_prompts.get("chat_messages")) or bool(
        rendered_prompts.get("text_prompts")
    )


def _trace_payload(payload: Any) -> Any:
    """Convert LLM input payload to a readable trace shape."""
    plain = _to_plain_data(payload)
    if isinstance(plain, dict) and isinstance(plain.get("messages"), list):
        compact = dict(plain)
        compact["messages"] = [
            _compact_message(message) for message in plain["messages"]
        ]
        return _json_safe(compact)
    return _json_safe(plain)


def _trace_response(response: Any) -> Any:
    """Convert LLM output to a readable trace shape without provider internals."""
    plain = _to_plain_data(response)
    if isinstance(plain, dict) and isinstance(plain.get("messages"), list):
        compact: dict[str, Any] = {}
        if "structured_response" in plain:
            compact["structured_response"] = _json_safe(plain["structured_response"])
        else:
            for key, value in plain.items():
                if key != "messages":
                    compact[key] = _json_safe(value)

        tool_activity = _tool_activity_from_messages(plain["messages"])
        if tool_activity:
            compact["tool_activity"] = tool_activity
        if not compact:
            compact["messages"] = [
                _compact_message(message) for message in plain["messages"]
            ]
        return compact
    return _json_safe(plain)


def _trace_rendered_prompts(rendered_prompts: dict[str, Any] | None) -> dict[str, Any]:
    """Compact callback-collected rendered prompts for Markdown/JSONL traces."""
    if not rendered_prompts:
        return {}

    compact: dict[str, Any] = {}
    chat_messages = rendered_prompts.get("chat_messages")
    if isinstance(chat_messages, list):
        compact["chat_messages"] = [
            [_compact_message(message) for message in message_group]
            for message_group in chat_messages
            if isinstance(message_group, list)
        ]

    text_prompts = rendered_prompts.get("text_prompts")
    if isinstance(text_prompts, list) and text_prompts:
        compact["text_prompts"] = [
            _compact_trace_text(prompt) for prompt in text_prompts
        ]

    return compact


def _tool_activity_from_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Extract compact tool calls/results from a LangChain agent message history."""
    activity: list[dict[str, Any]] = []
    for message in messages:
        compact = _compact_message(message)
        tool_calls = compact.get("tool_calls")
        if tool_calls:
            activity.append(
                {
                    "type": "assistant_tool_call",
                    "tool_calls": tool_calls,
                }
            )
        if compact.get("type") == "tool":
            item = {
                "type": "tool_result",
                "name": compact.get("name", ""),
                "content": compact.get("content", ""),
            }
            if compact.get("status"):
                item["status"] = compact["status"]
            activity.append(item)
    return activity


def _message_to_plain(message: Any) -> dict[str, Any]:
    """Convert a LangChain message object to stable trace data."""
    return _compact_message(message)


def _compact_message(message: Any) -> dict[str, Any]:
    """Keep only role/content/tool data from a LangChain message."""
    plain = _to_plain_data(message)
    if not isinstance(plain, dict):
        return {
            "type": message.__class__.__name__,
            "content": _json_safe(getattr(message, "content", str(message))),
        }

    compact: dict[str, Any] = {}
    message_type = plain.get("type") or plain.get("role")
    if message_type:
        compact["type"] = str(message_type)
    name = plain.get("name")
    if name:
        compact["name"] = str(name)
    content = plain.get("content")
    if content not in (None, ""):
        compact["content"] = _maybe_json_value(content)
    status = plain.get("status")
    if status:
        compact["status"] = str(status)

    tool_calls = plain.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        compact["tool_calls"] = [_compact_tool_call(call) for call in tool_calls]

    invalid_tool_calls = plain.get("invalid_tool_calls")
    if invalid_tool_calls:
        compact["invalid_tool_calls"] = _json_safe(invalid_tool_calls)
    return compact


def _compact_tool_call(call: Any) -> dict[str, Any]:
    """Keep the readable part of a LangChain tool call."""
    plain = _to_plain_data(call)
    if not isinstance(plain, dict):
        return {"value": _json_safe(plain)}

    compact: dict[str, Any] = {}
    name = plain.get("name")
    if name:
        compact["name"] = str(name)
    if "args" in plain:
        compact["args"] = _json_safe(plain["args"])
    if "type" in plain:
        compact["type"] = str(plain["type"])
    return compact


def _maybe_json_value(value: Any) -> Any:
    """Parse JSON-looking strings in traces while leaving normal text alone."""
    if not isinstance(value, str):
        return _json_safe(value)

    text = value.strip()
    if not text or text[0] not in "[{":
        return _compact_trace_text(value)
    try:
        return _truncate_trace_value(json.loads(text))
    except json.JSONDecodeError:
        return _compact_trace_text(value)


def _json_safe(value: Any) -> Any:
    """Recursively convert common structured objects to JSON-safe values."""
    plain = _to_plain_data(value)
    if isinstance(plain, dict):
        return {str(key): _json_safe(item) for key, item in plain.items()}
    if isinstance(plain, (list, tuple, set)):
        return [_json_safe(item) for item in plain]
    if isinstance(plain, (str, int, float, bool)) or plain is None:
        return _compact_trace_text(plain) if isinstance(plain, str) else plain
    if hasattr(plain, "content"):
        return {
            "type": plain.__class__.__name__,
            "content": _json_safe(getattr(plain, "content", "")),
        }
    return _compact_trace_text(repr(plain))


def _truncate_trace_value(value: Any) -> Any:
    """Recursively cap large trace values while keeping the shape readable."""
    if isinstance(value, dict):
        return {str(key): _truncate_trace_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_truncate_trace_value(item) for item in value]
    if isinstance(value, str):
        return _compact_trace_text(value)
    return value


def _compact_trace_text(value: Any) -> str:
    """Cap long text in trace artifacts with an explicit truncation marker."""
    text = str(value)
    if len(text) <= LLM_TRACE_TEXT_MAX_CHARS:
        return text
    omitted = len(text) - LLM_TRACE_TEXT_MAX_CHARS
    return (
        text[: LLM_TRACE_TEXT_MAX_CHARS - 80].rstrip()
        + f"\n...[trace truncated, {omitted} chars omitted]"
    )


def _json_dumps(value: Any) -> str:
    """Serialize a JSON-safe value with stable Korean-preserving formatting."""
    return json.dumps(_json_safe(value), ensure_ascii=False, indent=2, default=str)


def _markdown_code_block(language: str, content: str) -> str:
    """Wrap content in a Markdown code fence that will not be broken by content."""
    fence = "```"
    while fence in content:
        fence += "`"
    return f"{fence}{language}\n{content}\n{fence}"


def _is_resource_exhausted_error(exc: BaseException) -> bool:
    """Return whether an exception chain represents a provider quota exhaustion."""
    seen: set[int] = set()
    stack: list[BaseException] = [exc]

    while stack:
        current = stack.pop()
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)

        class_name = current.__class__.__name__.lower()
        message = str(current).lower()
        status_code = getattr(current, "code", None)
        if callable(status_code):
            try:
                status_code = status_code()
            except Exception:
                status_code = None
        status_text = str(status_code).lower()

        if (
            "resourceexhausted" in class_name
            or "resource_exhausted" in message
            or "resource exhausted" in message
            or "statuscode.resource_exhausted" in status_text
            or "429" in message
        ):
            return True

        cause = getattr(current, "__cause__", None)
        context = getattr(current, "__context__", None)
        if cause is not None:
            stack.append(cause)
        if context is not None:
            stack.append(context)

    return False


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with validation."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} 값은 정수여야 합니다.") from exc
    return max(1, value)


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable with validation."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} 값은 숫자여야 합니다.") from exc
    return max(0.0, value)


def _structured_response(result: Any) -> Any:
    """LangChain agent 실행 결과에서 structured_response를 꺼낸다."""
    if isinstance(result, dict) and "structured_response" in result:
        return result["structured_response"]
    return result


def _normalize_glossary(value: Any) -> dict[str, str]:
    """Context structured output을 내부 용어집 dict로 정규화한다."""
    if isinstance(value, dict):
        return {str(key): str(item) for key, item in value.items() if key and item}

    normalized: dict[str, str] = {}
    if not isinstance(value, list):
        return normalized

    for item in value:
        if isinstance(item, dict):
            source_term = item.get("source_term") or item.get("term") or item.get("source")
            target_term = item.get("target_term") or item.get("translation") or item.get("target")
        else:
            source_term = (
                getattr(item, "source_term", None)
                or getattr(item, "term", None)
                or getattr(item, "source", None)
            )
            target_term = (
                getattr(item, "target_term", None)
                or getattr(item, "translation", None)
                or getattr(item, "target", None)
            )
        if source_term and target_term:
            normalized[str(source_term)] = str(target_term)
    return normalized


def _to_plain_data(value: Any) -> Any:
    """Pydantic model 또는 일반 값을 직렬화 가능한 값으로 변환한다."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value


def _subtitle_segments(state: DubbingState) -> list[dict[str, Any]]:
    """자막 생성 함수가 기대하는 세그먼트 dict 목록으로 변환한다."""
    return [
        {
            "start": seg["start"],
            "end": seg["end"],
            "original": seg["source_text"],
            "translated": seg.get("adjusted_text") or seg.get("translated_text", ""),
        }
        for seg in state.get("segments", [])
    ]
