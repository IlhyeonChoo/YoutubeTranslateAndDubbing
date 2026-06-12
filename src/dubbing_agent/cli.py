"""LangGraph 더빙 에이전트용 명령행 인터페이스."""

from __future__ import annotations

import os

import click
from dotenv import load_dotenv

from dubbing_agent.graph.build_graph import build_dubbing_graph
from dubbing_agent.preflight import has_preflight_failures, run_preflight_checks
from dubbing_agent.state import DubbingState
from dubbing_agent.utils.paths import reserve_job_output_dir


# .env 파일 로드
load_dotenv()


@click.group()
def main() -> None:
    """LangGraph 기반 더빙 워크플로우를 실행합니다."""


@main.command()
@click.option(
    "--llm-provider",
    default="vertexai",
    envvar="DUBBING_AGENT_LLM_PROVIDER",
    type=click.Choice(["vertexai", "google-genai"]),
    show_default=True,
    help="점검할 Gemini provider입니다.",
)
@click.option(
    "--google-api-key",
    default=None,
    envvar=["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    help="google-genai provider에서 사용할 Google AI Studio/Gemini API key입니다.",
)
@click.option(
    "--google-project",
    default=None,
    envvar="GOOGLE_CLOUD_PROJECT",
    help="vertexai provider에서 사용할 Google Cloud 프로젝트 ID입니다.",
)
@click.option(
    "--google-location",
    default="us-central1",
    envvar="GOOGLE_CLOUD_LOCATION",
    show_default=True,
    help="vertexai provider에서 사용할 Gemini 호출 리전입니다.",
)
@click.option(
    "--google-credentials",
    default=None,
    envvar="GOOGLE_APPLICATION_CREDENTIALS",
    help="vertexai provider에서 사용할 서비스 계정 JSON 키 파일 경로입니다.",
)
@click.option(
    "--strict",
    is_flag=True,
    help="경고 항목도 실패로 처리합니다.",
)
def preflight(
    llm_provider: str,
    google_api_key: str | None,
    google_project: str | None,
    google_location: str,
    google_credentials: str | None,
    strict: bool,
) -> None:
    """실행 전 외부 도구와 Gemini 인증 설정을 점검합니다."""
    checks = run_preflight_checks(
        llm_provider=llm_provider,
        google_api_key=google_api_key,
        google_project=google_project,
        google_location=google_location,
        google_credentials=google_credentials,
    )
    _print_preflight_results(checks)
    if has_preflight_failures(checks, strict=strict):
        raise click.ClickException("실행 전 점검을 통과하지 못했습니다.")


# 기본 시작
@main.command()

@click.argument("url")
@click.option("-o", "--output-dir", default="data/output", help="기본 출력 디렉터리입니다.")
@click.option("-t", "--target-lang", default="ko", help="번역 대상 언어 코드입니다.")
@click.option("-s", "--source-lang", default=None, help="원본 언어 코드입니다.")
@click.option(
    "-m",
    "--whisper-model",
    default="large",
    type=click.Choice(["tiny", "base", "small", "medium", "large"]),
    help="Whisper 모델 크기입니다.",
)
@click.option(
    "--whisper-device",
    default="auto",
    type=click.Choice(["auto", "cpu", "cuda"]),
    show_default=True,
    help="Whisper 실행 장치입니다. auto는 CUDA 사용 가능 시 GPU를 사용합니다.",
)
@click.option(
    "--subtitle-mode",
    default="full",
    type=click.Choice(["full", "translated", "original"]),
    show_default=True,
    help="SRT 자막 내용입니다. full=원문+발음+번역, translated=번역만, original=원문만.",
)
@click.option(
    "--llm-provider",
    default="vertexai",
    envvar="DUBBING_AGENT_LLM_PROVIDER",
    type=click.Choice(["vertexai", "google-genai"]),
    show_default=True,
    help="Gemini 호출 provider입니다. vertexai=Google Cloud Vertex AI, google-genai=Google AI Studio API입니다.",
)
@click.option("--llm-model", default="gemini-2.5-flash", help="Gemini 모델명입니다.")
@click.option("--llm-temperature", default=0.2, type=float, help="LLM 응답 다양성을 조절하는 temperature 값입니다.")
@click.option(
    "--google-api-key",
    default=None,
    envvar=["GOOGLE_API_KEY", "GEMINI_API_KEY"],
    help="google-genai provider에서 사용할 Google AI Studio/Gemini API key입니다.",
)
@click.option(
    "--google-project",
    default=None,
    envvar="GOOGLE_CLOUD_PROJECT",
    help="vertexai provider에서 사용할 Google Cloud 프로젝트 ID입니다.",
)
@click.option(
    "--google-location",
    default="us-central1",
    envvar="GOOGLE_CLOUD_LOCATION",
    show_default=True,
    help="vertexai provider에서 사용할 Gemini 호출 리전입니다.",
)
@click.option(
    "--google-credentials",
    default=None,
    envvar="GOOGLE_APPLICATION_CREDENTIALS",
    help="vertexai provider에서 사용할 서비스 계정 JSON 키 파일 경로입니다.",
)
@click.option(
    "--disable-duration-rewrite",
    is_flag=True,
    help="LLM 발화 길이 재작성을 건너뛰고 TTS 길이 맞춤에만 의존합니다.",
)
@click.option("--tts-voice", default=None, help="사용할 TTS 음성명입니다.")
@click.option("--tts-rate", default="+0%", help="기본 TTS 말하기 속도입니다. 예: +0%, +10%, -5%.")
@click.option(
    "--max-translation-attempts",
    default=3,
    type=int,
    show_default=True,
    help="번역 QC 실패 시 세그먼트별 최대 재번역 횟수입니다.",
)
@click.option(
    "--max-global-translation-review-attempts",
    default=2,
    type=int,
    show_default=True,
    help="전체 번역 리뷰가 수정 제안을 적용해 재처리할 최대 횟수입니다.",
)
@click.option(
    "--max-duration-rewrite-attempts",
    default=3,
    type=int,
    show_default=True,
    help="TTS 길이 검증 실패 시 세그먼트별 최대 문장 재작성 횟수입니다.",
)
@click.option(
    "--tts-duration-tolerance",
    default=0.12,
    type=float,
    show_default=True,
    help="원본 발화 길이 대비 허용할 TTS 길이 초과 비율입니다.",
)
@click.option(
    "--allow-quality-failures",
    is_flag=True,
    help="번역/더빙 QC 실패가 남아 있어도 최종 파일 생성을 계속합니다.",
)
@click.option("--checkpoint", is_flag=True, help="메모리 기반 LangGraph checkpoint를 활성화합니다.")

# 실제 옵션들을 받아서 실행
def run(
    url: str,
    output_dir: str,
    target_lang: str,
    source_lang: str | None,
    whisper_model: str,
    whisper_device: str,
    subtitle_mode: str,
    llm_provider: str,
    llm_model: str,
    llm_temperature: float,
    google_api_key: str | None,
    google_project: str | None,
    google_location: str,
    google_credentials: str | None,
    disable_duration_rewrite: bool,
    tts_voice: str | None,
    tts_rate: str,
    max_translation_attempts: int,
    max_global_translation_review_attempts: int,
    max_duration_rewrite_attempts: int,
    tts_duration_tolerance: float,
    allow_quality_failures: bool,
    checkpoint: bool,
) -> None:
    """YouTube URL로부터 더빙 작업을 실행합니다."""
    resolved_job_id, resolved_output_dir = reserve_job_output_dir(output_dir)   # 디렉토리 생성
    if google_api_key:
        os.environ["GOOGLE_API_KEY"] = google_api_key

    state: DubbingState = {
        "job_id": resolved_job_id,
        "source_url": url,
        "output_dir": resolved_output_dir,
        "target_language": target_lang,
        "source_language": source_lang,
        "whisper_model": whisper_model,
        "whisper_device": whisper_device,
        "subtitle_mode": subtitle_mode,
        "llm_model": llm_model,
        "llm_provider": llm_provider,
        "llm_temperature": llm_temperature,
        "google_project": google_project,
        "google_location": google_location,
        "google_credentials": google_credentials,
        "duration_rewrite_enabled": not disable_duration_rewrite,
        "tts_voice": tts_voice,
        "tts_rate": tts_rate,
        "max_translation_attempts": max_translation_attempts,
        "max_global_translation_review_attempts": max_global_translation_review_attempts,
        "global_translation_review_attempts": 0,
        "global_translation_reprocess_segment_ids": [],
        "max_duration_rewrite_attempts": max_duration_rewrite_attempts,
        "tts_duration_tolerance": tts_duration_tolerance,
        "enforce_quality_gates": not allow_quality_failures,
        "segments": [],
        "current_segment_index": 0,
        "glossary": {},
        "errors": [],
        "metadata": {},
    }

    # LangGraph 그래프 생성 및 실행
    graph = build_dubbing_graph(checkpoint=checkpoint)
    config = {"configurable": {"thread_id": resolved_job_id}}

    click.echo(f"작업 ID: {resolved_job_id}")
    click.echo(f"출력 디렉터리: {os.path.abspath(resolved_output_dir)}")
    result = graph.invoke(state, config=config)

    click.echo("")
    click.echo("완료되었습니다.")
    
    # 아래는 중간 산출물들과 결과 등 부과적인 문서 출력
    _print_quality_gate_status(result)
    click.echo(f"자막: {result.get('subtitle_path', '')}")
    if result.get("quality_report_path"):
        click.echo(f"품질 리포트: {result['quality_report_path']}")
    if result.get("quality_summary_path"):
        click.echo(f"품질 요약: {result['quality_summary_path']}")
    if result.get("context_summary_markdown_path"):
        click.echo(f"문맥 요약: {result['context_summary_markdown_path']}")
    if result.get("llm_trace_markdown_path"):
        click.echo(f"LLM 호출 기록: {result['llm_trace_markdown_path']}")
    if result.get("output_video_path"):
        click.echo(f"더빙 영상: {result['output_video_path']}")
    elif result.get("final_audio_path"):
        click.echo(f"더빙 오디오: {result['final_audio_path']}")


def _print_quality_gate_status(result: DubbingState) -> None:
    """CLI output에 최종 품질 게이트 상태를 명확히 표시한다."""
    quality_gate = result.get("metadata", {}).get("quality_gate", {})
    if not quality_gate:
        return

    if quality_gate.get("passed") is True:
        click.echo("품질 게이트: 통과")
        return

    if quality_gate.get("enforced") is False:
        click.echo(
            "품질 경고: 품질 게이트가 실패했지만 --allow-quality-failures 설정으로 "
            "산출물을 생성했습니다."
        )
    else:
        click.echo("품질 경고: 품질 게이트가 실패했지만 산출물을 생성했습니다.")

    failures = quality_gate.get("failures", [])
    for failure in failures[:5]:
        click.echo(f"- {failure}")
    if len(failures) > 5:
        click.echo(f"- ... (+{len(failures) - 5}개 더)")


def _print_preflight_results(checks: list[dict[str, str]]) -> None:
    """CLI output에 preflight 점검 결과를 표시한다."""
    labels = {
        "ok": "정상",
        "warn": "경고",
        "fail": "실패",
    }
    click.echo("실행 전 점검 결과:")
    for check in checks:
        status = labels.get(check["status"], check["status"].upper())
        click.echo(f"- [{status}] {check['name']}: {check['message']}")


if __name__ == "__main__":
    main()
