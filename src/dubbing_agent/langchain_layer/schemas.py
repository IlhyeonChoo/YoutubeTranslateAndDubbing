"""LLM 호출용 structured output schema."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class GlossaryEntry(BaseModel):
    """A term mapping used to keep translations consistent."""

    source_term: str = Field(description="원문 transcript에 나온 중요 용어 또는 이름.")
    target_term: str = Field(description="대상 언어에서 사용할 일관된 표현.")


class ContextResult(BaseModel):
    """세그먼트 번역 품질을 높이기 위한 전체 transcript 문맥."""

    summary: str = Field(description="영상 transcript의 간결한 요약.")
    domain: str = Field(description="영상의 주제 분야 또는 장르.")
    glossary: list[GlossaryEntry] = Field(
        default_factory=list,
        description="중요한 용어 또는 이름과 대상 언어 표현의 매핑 목록.",
    )
    translation_style: str = Field(
        description="자연스러운 더빙 대사 스타일을 위한 짧은 지침.",
    )


class SegmentTranslationResult(BaseModel):
    """단일 세그먼트 번역 결과."""

    translated_text: str = Field(description="대상 언어로 자연스럽게 번역한 문장.")
    confidence: float = Field(ge=0.0, le=1.0, description="번역 신뢰도.")
    notes: str = Field(default="", description="필요한 경우 짧은 참고 사항.")


class TranslationQCResult(BaseModel):
    """번역된 세그먼트의 품질 검증 결과."""

    action: Literal["pass", "retry", "rewrite"] = Field(
        default="pass",
        description="통과, 재번역, 즉시 개선 번역 적용 중 하나의 권장 조치.",
    )
    passed: bool = Field(description="번역을 통과로 볼 수 있는지 여부.")
    score: float = Field(ge=0.0, le=1.0, description="0부터 1까지의 품질 점수.")
    issues: list[str] = Field(default_factory=list, description="간결한 문제 목록.")
    suggested_translation: str = Field(
        default="",
        description="기존 번역이 QC를 통과하지 못했을 때의 개선 번역.",
    )


class DurationRewriteResult(BaseModel):
    """더빙 발화 길이를 고려한 재작성 결과."""

    adjusted_text: str = Field(description="발화 시간에 맞게 재작성한 대상 언어 문장.")
    reason: str = Field(default="", description="간결한 재작성 이유.")
    estimated_seconds: float | None = Field(
        default=None,
        description="모델이 추정할 수 있는 경우의 예상 발화 길이.",
    )


class TranslationReviewIssue(BaseModel):
    """A whole-transcript translation issue tied to a segment."""

    segment_id: int = Field(description="문제가 있는 세그먼트 id.")
    issue: str = Field(description="전체 문맥 기준의 번역 문제.")
    suggested_translation: str = Field(
        default="",
        description="문제가 있는 세그먼트에 적용할 자연스러운 대체 번역.",
    )


class GlobalTranslationReviewResult(BaseModel):
    """전체 transcript 기준의 번역 일관성 검토 결과."""

    passed: bool = Field(description="전체 번역 흐름을 통과로 볼 수 있는지 여부.")
    score: float = Field(ge=0.0, le=1.0, description="0부터 1까지의 전체 번역 품질 점수.")
    issues: list[TranslationReviewIssue] = Field(
        default_factory=list,
        description="전체 문맥 기준의 문제와 가능한 수정 제안 목록.",
    )
    summary: str = Field(default="", description="간결한 전체 검토 요약.")


class SubtitleInspectionDecision(BaseModel):
    """영상에 포함된 기존 자막 사용 가능성 판단 결과."""

    action: Literal["use_subtitle", "run_stt", "defer"] = Field(
        description="기존 자막 사용, STT 실행, 또는 판단 보류 중 하나의 권장 조치.",
    )
    has_subtitles: bool = Field(description="수동 또는 자동 자막이 하나라도 있는지 여부.")
    subtitle_source: Literal["manual", "automatic", "none"] = Field(
        description="가장 적합하다고 판단한 자막 출처.",
    )
    selected_language: str = Field(default="", description="선택 가능한 경우의 자막 언어 코드.")
    confidence: float = Field(ge=0.0, le=1.0, description="판단 신뢰도.")
    reason: str = Field(description="간결한 판단 이유.")


class DubbingQCResult(BaseModel):
    """생성된 TTS 더빙 세그먼트의 품질 검증 결과."""

    action: Literal["pass", "rewrite", "continue"] = Field(
        description="통과, 문장 재작성 후 재생성, 또는 재시도 없이 계속 진행 중 하나의 권장 조치.",
    )
    passed: bool = Field(description="현재 TTS 세그먼트를 통과로 볼 수 있는지 여부.")
    score: float = Field(ge=0.0, le=1.0, description="0부터 1까지의 더빙 적합도 점수.")
    issues: list[str] = Field(default_factory=list, description="간결한 문제 목록.")
    measured_tts_seconds: float = Field(description="실제 생성된 TTS 길이.")
    source_seconds: float = Field(description="원본 세그먼트 길이.")
    tolerance: float = Field(description="허용한 길이 초과 비율.")
    reason: str = Field(default="", description="간결한 판단 이유.")
