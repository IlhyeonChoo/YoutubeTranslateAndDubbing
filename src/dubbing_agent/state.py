"""워크플로우 상태 정의."""

from __future__ import annotations

from typing import Any, Literal, NotRequired, TypedDict


SubtitleMode = Literal["full", "translated", "original"]
LLMProvider = Literal["vertexai", "google-genai"]


class SegmentState(TypedDict, total=False):
    """단일 transcript 세그먼트에 누적되는 상태."""

    id: int
    start: float
    end: float
    source_text: str
    translated_text: str
    adjusted_text: str
    sentence_id: int
    sentence_start: float
    sentence_end: float
    sentence_text: str
    sentence_tts_path: str
    sentence_tts_duration: float
    sentence_tts_slot_start: float
    sentence_tts_slot_end: float
    sentence_tts_slot_duration: float
    fallback_sentence_tts_path: str
    fallback_sentence_tts_duration: float
    fallback_sentence_tts_slot_start: float
    fallback_sentence_tts_slot_end: float
    fallback_sentence_tts_slot_duration: float
    tts_path: str
    qc_score: float
    qc_passed: bool
    qc_issues: list[str]
    qc_action: str
    translation_metrics: dict[str, Any]
    adjusted_translation_metrics: dict[str, Any]
    translation_attempts: int
    sentence_translation_retry_attempts: int
    duration_rewrite_attempts: int
    tts_duration: float
    duration_passed: bool
    dubbing_score: float
    dubbing_issues: list[str]
    dubbing_action: str
    dubbing_metrics: dict[str, Any]
    revision_feedback: list[str]


class DubbingState(TypedDict, total=False):
    """LangGraph 노드 사이에서 전달되는 직렬화 가능한 상태."""

    job_id: str
    source_url: str
    input_video_path: NotRequired[str]
    output_dir: str

    target_language: str
    source_language: NotRequired[str | None]
    detected_language: NotRequired[str]
    whisper_model: str
    whisper_device: str

    subtitle_mode: SubtitleMode

    llm_model: str
    llm_provider: LLMProvider
    llm_temperature: float
    google_project: NotRequired[str | None]
    google_location: str
    google_credentials: NotRequired[str | None]
    duration_rewrite_enabled: bool

    tts_voice: NotRequired[str | None]
    tts_rate: str
    max_translation_attempts: int
    max_global_translation_review_attempts: int
    global_translation_review_attempts: int
    global_translation_reprocess_segment_ids: list[int]
    max_duration_rewrite_attempts: int
    tts_duration_tolerance: float
    enforce_quality_gates: bool

    audio_path: NotRequired[str]
    final_audio_path: NotRequired[str]
    output_video_path: NotRequired[str]
    subtitle_path: NotRequired[str]
    quality_report_path: NotRequired[str]
    quality_summary_path: NotRequired[str]
    context_summary_json_path: NotRequired[str]
    context_summary_markdown_path: NotRequired[str]
    llm_trace_jsonl_path: NotRequired[str]
    llm_trace_markdown_path: NotRequired[str]

    segments: list[SegmentState]
    current_segment_index: int

    summary: NotRequired[str]
    domain: NotRequired[str]
    glossary: dict[str, str]
    translation_style: NotRequired[str]

    errors: list[str]
    metadata: dict[str, Any]
