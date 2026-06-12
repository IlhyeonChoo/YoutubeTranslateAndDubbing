"""LangChain chain 생성 함수."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from dubbing_agent.langchain_layer.schemas import (
    ContextResult,
    DurationRewriteResult,
    GlobalTranslationReviewResult,
    SegmentTranslationResult,
    TranslationQCResult,
)


def build_context_chain(llm: BaseChatModel):
    """Transcript 문맥과 용어집을 요약하는 chain을 만든다."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 숙련된 영상 현지화 편집자입니다. 일관되고 자연스러운 더빙에 "
                "도움이 되는 문맥을 transcript에서 분석하세요.",
            ),
            (
                "user",
                "대상 언어: {target_language}\n\nTranscript:\n{transcript}",
            ),
        ]
    )
    return prompt | llm.with_structured_output(ContextResult)


def build_translation_chain(llm: BaseChatModel):
    """Timestamp가 있는 세그먼트 하나를 번역하는 chain을 만든다."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 전문 영상 더빙 번역가입니다. 현재 세그먼트만 출력하되 "
                "같은 문장과 앞뒤 문맥 전체를 읽고 자연스러운 더빙 대사로 번역하세요. "
                "직역보다 의미, 흐름, 말맛, 발화 가능성을 우선하고 원문에 없는 사실은 "
                "추가하지 마세요. 개별 세그먼트 timestamp에 억지로 맞추지 말고 "
                "문장 단위 목표 길이 안에서 자연스럽게 말할 수 있게 조정하세요.",
            ),
            (
                "user",
                "원본 언어: {source_language}\n"
                "대상 언어: {target_language}\n"
                "참고용 세그먼트 길이: {duration_seconds:.2f}초\n"
                "문장 단위 목표 길이: {sentence_duration_seconds:.2f}초\n"
                "영상 요약: {summary}\n"
                "분야: {domain}\n"
                "용어집: {glossary}\n"
                "스타일: {translation_style}\n\n"
                "전체 원문 흐름:\n{global_transcript_context}\n\n"
                "같은 문장 전체 원문:\n{sentence_source_text}\n\n"
                "현재 문장 번역 초안:\n{sentence_translation_draft}\n\n"
                "같은 문장과 주변 문맥:\n{segment_context}\n\n"
                "이미 확정된 주변 번역:\n{neighbor_translations}\n\n"
                "이전 품질/더빙 피드백:\n{revision_feedback}\n\n"
                "원본 세그먼트:\n{source_text}",
            ),
        ]
    )
    return prompt | llm.with_structured_output(SegmentTranslationResult)


def build_translation_qc_chain(llm: BaseChatModel):
    """번역 품질을 검증하는 chain을 만든다."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 더빙 영상 번역 품질 검토자입니다. 후보 번역이 원문과 "
                "문장 단위 발화 시간 요구사항에 맞는지 판단하세요. 개별 자막 "
                "세그먼트 길이보다 문장 전체의 자연스러운 흐름과 말하기 속도를 "
                "우선하세요.",
            ),
            (
                "user",
                "원본 언어: {source_language}\n"
                "대상 언어: {target_language}\n"
                "세그먼트 길이: {duration_seconds:.2f}초\n"
                "영상 요약: {summary}\n"
                "용어집: {glossary}\n\n"
                "원문:\n{source_text}\n\n"
                "후보 번역:\n{translated_text}\n\n"
                "의미가 보존되고, 표현이 자연스러우며, 환각 내용이 없고, "
                "문장 단위 길이 안에서 자연스럽게 말할 수 있을 때만 통과로 "
                "판단하세요.",
            ),
        ]
    )
    return prompt | llm.with_structured_output(TranslationQCResult)


def build_duration_rewrite_chain(llm: BaseChatModel):
    """번역 대사를 발화 길이에 맞게 재작성하는 chain을 만든다."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 더빙 대사를 문장 단위 흐름과 발화 속도에 맞게 재작성합니다. "
                "현재 세그먼트만 출력하되 같은 문장 전체와 주변 번역을 고려하세요. "
                "너무 빠르게 말해야 한다면 직역을 버리고 자연스러운 의역으로 짧게 "
                "만드세요. 개별 세그먼트 길이에 기계적으로 맞추지 말고 문장 전체 "
                "길이와 자연스러운 연결을 우선하세요. 원문의 의미와 장면 흐름은 "
                "유지하세요. 도입부의 말 걸기, 감탄, 머뭇거림, 마무리 뉘앙스 같은 "
                "담화 기능을 삭제하지 마세요. 짧게 줄이더라도 설명체로 딱딱하게 "
                "바꾸지 말고 원래 화자의 구어체 톤을 보존하세요.",
            ),
            (
                "user",
                "대상 언어: {target_language}\n"
                "참고용 세그먼트 길이: {duration_seconds:.2f}초\n"
                "문장 단위 목표 길이: {sentence_duration_seconds:.2f}초\n"
                "문제: {issue}\n\n"
                "재작성 기준:\n"
                "- 원문이 청자에게 말을 거는 도입이면 한국어도 자연스러운 도입 표현을 유지하세요.\n"
                "- 원문이 설명을 마무리하는 말이면 갑작스럽거나 무례한 뉘앙스로 바꾸지 마세요.\n"
                "- 너무 짧게 압축해 의미, 톤, 담화 기능이 사라지면 실패입니다.\n"
                "- 길이를 줄여야 할 때는 핵심 의미를 유지한 자연스러운 구어체 의역을 쓰세요.\n\n"
                "같은 문장과 주변 문맥:\n{segment_context}\n\n"
                "이미 확정된 주변 번역:\n{neighbor_translations}\n\n"
                "원문:\n{source_text}\n\n"
                "현재 번역:\n{translated_text}",
            ),
        ]
    )
    return prompt | llm.with_structured_output(DurationRewriteResult)


def build_global_translation_review_chain(llm: BaseChatModel):
    """전체 transcript 기준의 번역 일관성 리뷰 chain을 만든다."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 더빙 영상의 최종 번역 감수자입니다. 전체 원문 흐름과 전체 "
                "번역문을 함께 보고 용어 일관성, 문맥 연결, 의미 누락/추가, 어색한 "
                "직역, 장면 흐름과 맞지 않는 표현을 검토하세요. 문제가 있는 세그먼트만 "
                "issues에 넣고, 바로 고칠 수 있으면 suggested_translation을 제공하세요. "
                "대체 번역은 원문 의미를 유지하되 자연스러운 의역을 허용하고, 문장 단위 "
                "더빙 길이를 과도하게 늘리지 않아야 합니다.",
            ),
            (
                "user",
                "원본 언어: {source_language}\n"
                "대상 언어: {target_language}\n"
                "영상 요약: {summary}\n"
                "분야: {domain}\n"
                "용어집: {glossary}\n"
                "스타일: {translation_style}\n\n"
                "전체 원문 transcript:\n{source_transcript}\n\n"
                "전체 번역 transcript:\n{translated_transcript}\n\n"
                "문장 단위 리뷰 문맥:\n{sentence_review_transcript}",
            ),
        ]
    )
    return prompt | llm.with_structured_output(GlobalTranslationReviewResult)
