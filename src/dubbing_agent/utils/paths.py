"""워크플로우 출력 파일 경로 헬퍼."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from uuid import uuid4

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def make_job_id(prefix: str = "job") -> str:
    """짧고 파일시스템에 안전한 작업 ID를 만든다."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = uuid4().hex[:8]
    return f"{prefix}_{timestamp}_{suffix}"


def safe_job_id(value: str) -> str:
    """사용자가 입력한 작업 ID를 경로에 사용할 수 있게 정규화한다."""
    cleaned = _SAFE_ID_RE.sub("_", value.strip())
    return cleaned.strip("._") or make_job_id()


def ensure_dir(path: str) -> str:
    """필요하면 디렉터리를 만들고 해당 경로를 반환한다."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def reserve_job_output_dir(base_output_dir: str, preferred_job_id: str | None = None) -> tuple[str, str]:
    """충돌하지 않는 작업 ID와 출력 디렉터리를 만들고 반환한다."""
    base_dir = Path(base_output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    base_job_id = safe_job_id(preferred_job_id) if preferred_job_id else make_job_id()
    candidate_job_id = base_job_id
    suffix = 2

    while True:
        candidate_dir = base_dir / candidate_job_id
        try:
            candidate_dir.mkdir()
        except FileExistsError:
            candidate_job_id = f"{base_job_id}-{suffix}"
            suffix += 1
            continue
        return candidate_job_id, str(candidate_dir)
