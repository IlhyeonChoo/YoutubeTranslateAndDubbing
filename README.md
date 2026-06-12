# YouTube 더빙 에이전트

클라우드 AI프로그래밍 텀프로젝트용 레포

LangChain/LangGraph를 이용하여 YouTube 링크를 입력받아 번역 및 더빙 뿐만 아니라 퀄리티 검증까지 진행해주는 AI 에이전트

## Agent/Tool 구조

워크트로우는 LangGraph `StateGraph`로 구성되며, LLM 판단이 필요한 지점은
LangChain agent와 tool을 사용합니다.

- `inspect_subtitle_availability`: YouTube metadata에서 수동/자동 자막을 확인합니다.
- `analyze_translation_quality`: 번역문 길이, 원문 복사 가능성, 발화 부담을 계산합니다.
- `analyze_dubbing_quality`: 생성된 문장 단위 TTS 길이와 발화 속도가 원본 timestamp에 맞는지 계산합니다.

agent 판단 결과에 따라 다음처럼 분기합니다.
```text
자막 사용 가능 → YouTube 자막을 세그먼트로 로드 → STT 건너뜀
자막 없음/로드 실패 → 오디오 추출 → Whisper STT
번역 QC 실패 → 세그먼트/문장 단위 재번역 또는 개선 번역 적용
전체 번역 리뷰 실패 → 제안 번역 적용 후 해당 세그먼트부터 QC/TTS 재처리
문장 단위 TTS 길이/속도 QC 실패 → 음성 trimming 없이 문장 그룹 재작성 또는 번역 재시도 후 TTS 재생성
해결되지 않은 품질 실패가 남음 → 기본적으로 최종 출력 중단
QC 통과 후 오디오 합성 → speedup/trimming 없이 문장 단위 TTS를 배치
STT/자막 로드 후 전체 문맥 요약 → JSON과 Markdown으로 저장
번역/QC/리뷰 LLM 호출 → 입력과 응답을 JSONL과 Markdown으로 저장
완료 후 번역/더빙 QC 결과 → 품질 리포트 JSON과 Markdown 요약 저장
```

## 설정

Linux에서 처음 준비할 때:
```bash
sudo apt update
sudo apt install -y ffmpeg git curl
uv python install 3.11
uv venv --python 3.11
UV_TORCH_BACKEND=cu128 uv sync
```

requirements 파일을 요구하는 환경에서는 다음처럼 설치할 수 있습니다.
```bash
uv venv --python 3.11
uv pip install -r requirements.txt
```

Windows PowerShell에서 준비할 때:

```powershell
uv python install 3.11
uv venv --python 3.11
UV_TORCH_BACKEND=cu128 uv sync
```

LangChain 번역/QC는 Gemini를 사용하며, provider는 두 가지를 지원합니다.

- `vertexai`: Google Cloud Vertex AI. 기본값입니다.
- `google-genai`: Google AI Studio/Gemini Developer API.

기본 방식인 Vertex AI를 쓸 때는 `.env`에 서비스 계정 JSON 키 파일과 프로젝트/리전을
지정하세요.
```bash
cp .env.example .env
# .env 파일에서 GOOGLE_APPLICATION_CREDENTIALS와 GOOGLE_CLOUD_PROJECT 값을 채우세요.
```

CLI는 실행 시 저장소 루트의 `.env`를 자동으로 로드합니다. PowerShell에서 환경 변수로
직접 지정하려면 다음처럼 설정할 수도 있습니다.

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="C:\path\to\service-account.json"
$env:GOOGLE_CLOUD_PROJECT="your-gcp-project-id"
$env:GOOGLE_CLOUD_LOCATION="us-central1"
```

Google AI Studio API를 쓸 때는 provider를 바꾸고 API key를 지정하세요.

```powershell
$env:DUBBING_AGENT_LLM_PROVIDER="google-genai"
$env:GOOGLE_API_KEY="your-google-ai-studio-api-key"
```

긴 영상은 세그먼트마다 여러 번 LLM을 호출하므로 provider quota에 걸릴 수 있습니다.
기본값은 Vertex AI 분당 12회, Google AI Studio 분당 4회로 제한하며, 429
`ResourceExhausted`가 발생하면 같은 LLM 요청을 backoff 후 재시도합니다. quota가 더
높으면 `.env`에서 값을 조정할 수 있습니다.
## 현재 quota 값을 조절할 경우 긴 영상에서는 중간에 실패하는 문제가 지속적으로 발생하고 있습니다.
```bash
DUBBING_AGENT_LLM_REQUESTS_PER_MINUTE=12
DUBBING_AGENT_LLM_RESOURCE_EXHAUSTED_RETRIES=8
DUBBING_AGENT_LLM_RESOURCE_EXHAUSTED_DELAY_SECONDS=60
DUBBING_AGENT_LLM_RESOURCE_EXHAUSTED_MAX_DELAY_SECONDS=300
```

## 실행

진행 중에는 `[진행 HH:MM:SS] 노드=번역 세그먼트=1/4 ...`처럼 현재 LangGraph 노드와 세그먼트 위치가 출력됩니다. 로그를 끄려면 `DUBBING_AGENT_PROGRESS=0`을 설정하세요.

작업 결과는 `data/output/job_...` 형태의 자동 생성 디렉터리에 저장됩니다.
같은 이름의 작업 디렉터리가 이미 있으면 뒤에 `-2`, `-3`처럼 숫자를 붙입니다.
각 작업 디렉터리에는 전체 문맥 요약(`context_summary.json`, `context_summary.md`)과
LLM 호출 기록(`llm_calls.jsonl`, `llm_calls.md`)도 함께 저장됩니다.

서비스 계정 키 경로를 명령어에서 직접 지정할 수도 있습니다.

```powershell
uv run dubbing-agent run "https://www.youtube.com/watch?v=..." `
  --google-credentials "C:\path\to\service-account.json" `
  --google-project "your-gcp-project-id" `
  --google-location "us-central1"
```

Google AI Studio API key를 명령어에서 직접 지정할 수도 있습니다.

```powershell
uv run dubbing-agent run "https://www.youtube.com/watch?v=..." `
  --llm-provider google-genai `
  --google-api-key "your-google-ai-studio-api-key" `
  --target-lang ko
```
