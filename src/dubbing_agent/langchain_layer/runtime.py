"""LangChain chain runtime 구성."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Literal

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_vertexai import ChatVertexAI

from dubbing_agent.langchain_layer.agents import (
    build_dubbing_qc_agent,
    build_subtitle_inspection_agent,
    build_translation_qc_agent,
)


LLMProvider = Literal["vertexai", "google-genai"]
DEFAULT_VERTEXAI_REQUESTS_PER_MINUTE = 12.0
DEFAULT_GOOGLE_GENAI_REQUESTS_PER_MINUTE = 4.0
DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS = 120.0
DEFAULT_LLM_PROVIDER_MAX_RETRIES = 1
from dubbing_agent.langchain_layer.chains import (
    build_context_chain,
    build_duration_rewrite_chain,
    build_global_translation_review_chain,
    build_translation_chain,
)


@dataclass(frozen=True)
class AgentChains:
    """그래프 노드에서 사용하는 LangChain runnable 묶음."""

    context_chain: object
    translation_chain: object
    global_translation_review_chain: object
    duration_rewrite_chain: object
    subtitle_inspection_agent: object
    translation_qc_agent: object
    dubbing_qc_agent: object


def build_chat_model(
    model: str = "gemini-2.5-flash",
    provider: LLMProvider = "vertexai",
    temperature: float = 0.2,
    google_api_key: str | None = None,
    google_project: str | None = None,
    google_location: str = "us-central1",
    google_credentials: str | None = None,
) -> BaseChatModel:
    """설정된 provider에 맞는 Gemini chat model을 생성한다."""
    if provider == "google-genai":
        return _build_google_genai_model(
            model=model,
            temperature=temperature,
            google_api_key=google_api_key,
        )
    if provider == "vertexai":
        return _build_vertexai_model(
            model=model,
            temperature=temperature,
            google_project=google_project,
            google_location=google_location,
            google_credentials=google_credentials,
        )
    raise ValueError(f"지원하지 않는 LLM provider입니다: {provider}")


def _build_vertexai_model(
    model: str,
    temperature: float,
    google_project: str | None,
    google_location: str,
    google_credentials: str | None,
) -> ChatVertexAI:
    """Vertex AI용 ChatVertexAI model을 생성한다."""
    if google_credentials:
        credentials_path = Path(google_credentials).expanduser()
        if not credentials_path.exists():
            raise FileNotFoundError(
                f"서비스 계정 JSON 키 파일을 찾을 수 없습니다: {credentials_path}"
            )
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials_path)
        google_project = google_project or _read_project_id(credentials_path)

    kwargs = {
        "model": model,
        "temperature": temperature,
        "location": google_location,
        "max_retries": _provider_max_retries_from_env(),
    }
    request_timeout = _request_timeout_from_env()
    if request_timeout is not None:
        kwargs["timeout"] = request_timeout
    rate_limiter = _rate_limiter_from_env(provider="vertexai")
    if rate_limiter is not None:
        kwargs["rate_limiter"] = rate_limiter
    if google_project:
        kwargs["project"] = google_project
    return ChatVertexAI(**kwargs)


def _build_google_genai_model(
    model: str,
    temperature: float,
    google_api_key: str | None,
) -> ChatGoogleGenerativeAI:
    """Google AI Studio Gemini API용 ChatGoogleGenerativeAI model을 생성한다."""
    kwargs = {
        "model": model,
        "temperature": temperature,
        "vertexai": False,
        "retries": _provider_max_retries_from_env(),
    }
    request_timeout = _request_timeout_from_env()
    if request_timeout is not None:
        kwargs["request_timeout"] = request_timeout
    rate_limiter = _rate_limiter_from_env(provider="google-genai")
    if rate_limiter is not None:
        kwargs["rate_limiter"] = rate_limiter
    if google_api_key:
        kwargs["api_key"] = google_api_key
    return ChatGoogleGenerativeAI(**kwargs)


def build_agent_chains(
    model: str = "gemini-2.5-flash",
    provider: LLMProvider = "vertexai",
    temperature: float = 0.2,
    google_api_key: str | None = None,
    google_project: str | None = None,
    google_location: str = "us-central1",
    google_credentials: str | None = None,
) -> AgentChains:
    """워크플로우 실행에 필요한 모든 LLM chain을 생성한다."""
    llm = build_chat_model(
        model=model,
        provider=provider,
        temperature=temperature,
        google_api_key=google_api_key,
        google_project=google_project,
        google_location=google_location,
        google_credentials=google_credentials,
    )
    return AgentChains(
        context_chain=build_context_chain(llm),
        translation_chain=build_translation_chain(llm),
        global_translation_review_chain=build_global_translation_review_chain(llm),
        duration_rewrite_chain=build_duration_rewrite_chain(llm),
        subtitle_inspection_agent=build_subtitle_inspection_agent(llm),
        translation_qc_agent=build_translation_qc_agent(llm),
        dubbing_qc_agent=build_dubbing_qc_agent(llm),
    )


def _read_project_id(credentials_path: Path) -> str | None:
    """서비스 계정 JSON에서 project_id를 읽는다."""
    try:
        data = json.loads(credentials_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    project_id = data.get("project_id")
    return str(project_id) if project_id else None


def _rate_limiter_from_env(provider: LLMProvider) -> InMemoryRateLimiter | None:
    """환경 변수 설정에 따라 LangChain rate limiter를 반환한다."""
    raw_limit = os.getenv("DUBBING_AGENT_LLM_REQUESTS_PER_MINUTE")
    if raw_limit is None:
        requests_per_minute = (
            DEFAULT_GOOGLE_GENAI_REQUESTS_PER_MINUTE
            if provider == "google-genai"
            else DEFAULT_VERTEXAI_REQUESTS_PER_MINUTE
        )
    else:
        try:
            requests_per_minute = float(raw_limit)
        except ValueError as exc:
            raise ValueError(
                "DUBBING_AGENT_LLM_REQUESTS_PER_MINUTE 값은 숫자여야 합니다."
            ) from exc

    if requests_per_minute <= 0:
        return None
    return _cached_rate_limiter(requests_per_minute)


def _request_timeout_from_env() -> float | None:
    """Return the provider request timeout, or None when explicitly disabled."""
    raw_timeout = os.getenv("DUBBING_AGENT_LLM_REQUEST_TIMEOUT_SECONDS")
    if raw_timeout is None:
        return DEFAULT_LLM_REQUEST_TIMEOUT_SECONDS
    try:
        timeout = float(raw_timeout)
    except ValueError as exc:
        raise ValueError(
            "DUBBING_AGENT_LLM_REQUEST_TIMEOUT_SECONDS must be numeric."
        ) from exc
    if timeout <= 0:
        return None
    return timeout


def _provider_max_retries_from_env() -> int:
    """Return the provider client's internal retry budget."""
    raw_retries = os.getenv("DUBBING_AGENT_LLM_PROVIDER_MAX_RETRIES")
    if raw_retries is None:
        return DEFAULT_LLM_PROVIDER_MAX_RETRIES
    try:
        retries = int(raw_retries)
    except ValueError as exc:
        raise ValueError(
            "DUBBING_AGENT_LLM_PROVIDER_MAX_RETRIES must be an integer."
        ) from exc
    return max(0, retries)


@lru_cache(maxsize=8)
def _cached_rate_limiter(requests_per_minute: float) -> InMemoryRateLimiter:
    """같은 프로세스의 여러 chain이 공유하는 rate limiter를 만든다."""
    return InMemoryRateLimiter(
        requests_per_second=requests_per_minute / 60.0,
        check_every_n_seconds=0.5,
        max_bucket_size=1.0,
    )
