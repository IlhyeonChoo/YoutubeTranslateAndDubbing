"""LangChain agent 생성 함수."""

from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel

from dubbing_agent.langchain_layer.schemas import (
    DubbingQCResult,
    SubtitleInspectionDecision,
    TranslationQCResult,
)
from dubbing_agent.langchain_layer.tools import (
    analyze_dubbing_quality,
    analyze_translation_quality,
    inspect_subtitle_availability,
)


def build_subtitle_inspection_agent(llm: BaseChatModel):
    """기존 영상 자막 사용 가능성을 판단하는 agent를 만든다."""
    return create_agent(
        model=llm,
        tools=[inspect_subtitle_availability],
        response_format=SubtitleInspectionDecision,
        system_prompt=(
            "당신은 영상 현지화 파이프라인의 입력 검사자입니다. "
            "반드시 inspect_subtitle_availability 도구를 호출해 실제 자막 metadata를 "
            "확인한 뒤 판단하세요. selected_language는 번역할 원본 speech의 언어 자막이어야 "
            "합니다. target_language와 같다는 이유만으로 자동번역 자막을 고르지 마세요. "
            "source_language가 auto이면 수동 자막을 우선하고, 수동 자막이 없을 때 원본에 "
            "가장 가까운 자동 자막을 선택하세요. 적절한 원본 자막이 없거나 도구 오류가 "
            "있으면 run_stt를 권장하세요."
        ),
    )


def build_translation_qc_agent(llm: BaseChatModel):
    """번역 후보를 도구 분석과 LLM 판단으로 검증하는 agent를 만든다."""
    return create_agent(
        model=llm,
        tools=[analyze_translation_quality],
        response_format=TranslationQCResult,
        system_prompt=(
            "당신은 더빙 영상 번역 품질 검토자입니다. 반드시 "
            "analyze_translation_quality 도구를 호출한 뒤 원문 의미 보존, 같은 문장 안의 "
            "흐름, 앞뒤 번역과의 자연스러운 연결, 누락/추가 여부, 발화 길이를 함께 "
            "판단하세요. 발화 길이는 개별 자막 세그먼트보다 문장 단위 시간과 "
            "자연스러운 말하기 속도를 우선하세요. 직역이 어색하면 의역을 권장하세요. "
            "통과가 가능하면 action=pass, 재번역이 필요하면 action=retry, 현재 번역을 "
            "바로 고친 문장을 줄 수 있으면 action=rewrite로 응답하세요. rewrite 또는 "
            "retry일 때는 가능한 경우 suggested_translation을 채우세요."
        ),
    )


def build_dubbing_qc_agent(llm: BaseChatModel):
    """생성된 TTS 더빙 길이와 적합성을 검증하는 agent를 만든다."""
    return create_agent(
        model=llm,
        tools=[analyze_dubbing_quality],
        response_format=DubbingQCResult,
        system_prompt=(
            "당신은 더빙 엔지니어입니다. 반드시 analyze_dubbing_quality 도구를 호출한 뒤 "
            "생성된 TTS가 원본 문장 단위 시간과 자연스러운 말하기 속도에 맞는지 "
            "판단하세요. 너무 빠른 발화가 필요하면 재작성 또는 더 짧은 의역이 "
            "필요합니다. 재작성으로 개선할 수 있으면 action=rewrite와 passed=false를 "
            "반환하세요. 허용 범위를 벗어나도 더 이상 재작성 가치가 낮다고 판단되면 "
            "action=continue를 사용할 수 있습니다."
        ),
    )
