"""실행 전 외부 의존성과 인증 설정을 점검한다."""

from __future__ import annotations

from importlib.util import find_spec
import os
from pathlib import Path
import shutil
from typing import Literal, TypedDict


PreflightStatus = Literal["ok", "warn", "fail"]


class PreflightCheck(TypedDict):
    """CLI preflight 출력에 사용할 단일 점검 결과."""

    name: str
    status: PreflightStatus
    message: str


def run_preflight_checks(
    *,
    llm_provider: str,
    google_api_key: str | None = None,
    google_project: str | None = None,
    google_location: str = "us-central1",
    google_credentials: str | None = None,
) -> list[PreflightCheck]:
    """워크플로우 실행 전에 빠르게 검증할 수 있는 의존성과 설정을 점검한다."""
    checks: list[PreflightCheck] = []
    checks.extend(_python_dependency_checks())
    checks.extend(_binary_dependency_checks())
    checks.extend(
        _llm_configuration_checks(
            llm_provider=llm_provider,
            google_api_key=google_api_key,
            google_project=google_project,
            google_location=google_location,
            google_credentials=google_credentials,
        )
    )
    return checks


def has_preflight_failures(
    checks: list[PreflightCheck],
    *,
    strict: bool = False,
) -> bool:
    """preflight 결과가 CLI 실패로 처리되어야 하는지 반환한다."""
    failed_statuses = {"fail", "warn"} if strict else {"fail"}
    return any(check["status"] in failed_statuses for check in checks)


def _python_dependency_checks() -> list[PreflightCheck]:
    """필수 Python package가 import 가능한지 확인한다."""
    modules = {
        "yt-dlp": "yt_dlp",
        "edge-tts": "edge_tts",
        "openai-whisper": "whisper",
        "moviepy": "moviepy",
        "pydub": "pydub",
    }
    checks = []
    for label, module_name in modules.items():
        if find_spec(module_name):
            checks.append(
                _check(label, "ok", f"Python 모듈 '{module_name}'을 사용할 수 있습니다.")
            )
        else:
            checks.append(
                _check(label, "fail", f"Python 모듈 '{module_name}'을 찾을 수 없습니다.")
            )
    return checks


def _binary_dependency_checks() -> list[PreflightCheck]:
    """필수 외부 실행 파일이 PATH에 있는지 확인한다."""
    if shutil.which("ffmpeg"):
        return [_check("ffmpeg", "ok", "PATH에서 ffmpeg 실행 파일을 찾았습니다.")]
    return [_check("ffmpeg", "fail", "PATH에서 ffmpeg 실행 파일을 찾을 수 없습니다.")]


def _llm_configuration_checks(
    *,
    llm_provider: str,
    google_api_key: str | None,
    google_project: str | None,
    google_location: str,
    google_credentials: str | None,
) -> list[PreflightCheck]:
    """LLM provider별 최소 인증 설정을 확인한다."""
    if llm_provider == "google-genai":
        key_present = bool(
            google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        )
        if key_present:
            return [
                _check(
                    "google-genai 인증",
                    "ok",
                    "Google AI Studio API key가 설정되어 있습니다.",
                )
            ]
        return [
            _check(
                "google-genai 인증",
                "fail",
                "google-genai에는 GOOGLE_API_KEY 또는 GEMINI_API_KEY가 필요합니다.",
            )
        ]

    if llm_provider != "vertexai":
        return [_check("LLM provider", "fail", f"지원하지 않는 LLM provider입니다: {llm_provider}")]

    checks: list[PreflightCheck] = []
    if google_location:
        checks.append(_check("vertexai 리전", "ok", f"Vertex AI 리전: {google_location}"))
    else:
        checks.append(_check("vertexai 리전", "fail", "GOOGLE_CLOUD_LOCATION이 비어 있습니다."))

    credentials_path = google_credentials or os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if credentials_path:
        path = Path(credentials_path).expanduser()
        if path.exists():
            checks.append(
                _check(
                    "vertexai 인증 파일",
                    "ok",
                    "GOOGLE_APPLICATION_CREDENTIALS가 존재하는 파일을 가리킵니다.",
                )
            )
        else:
            checks.append(
                _check(
                    "vertexai 인증 파일",
                    "fail",
                    "GOOGLE_APPLICATION_CREDENTIALS가 없는 파일을 가리킵니다.",
                )
            )
    else:
        checks.append(
            _check(
                "vertexai 인증 파일",
                "warn",
                "GOOGLE_APPLICATION_CREDENTIALS가 설정되지 않았습니다. 환경 기본 인증(ADC)이 필요합니다.",
            )
        )

    if google_project or os.getenv("GOOGLE_CLOUD_PROJECT"):
        checks.append(_check("vertexai 프로젝트", "ok", "Google Cloud 프로젝트가 설정되어 있습니다."))
    else:
        checks.append(
            _check(
                "vertexai 프로젝트",
                "warn",
                "GOOGLE_CLOUD_PROJECT가 설정되지 않았습니다. Vertex AI가 ADC 기본값에 의존할 수 있습니다.",
            )
        )
    return checks


def _check(name: str, status: PreflightStatus, message: str) -> PreflightCheck:
    """점검 결과 dict를 만든다."""
    return {"name": name, "status": status, "message": message}
