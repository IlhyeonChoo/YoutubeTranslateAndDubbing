"""Audio duration helpers."""

from __future__ import annotations

import math
import subprocess


FFPROBE_TIMEOUT_SECONDS = 15.0


def get_audio_duration_seconds(path: str) -> float:
    """Return audio duration in seconds without fully decoding the file."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nw=1:nk=1",
                path,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=FFPROBE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"ffprobe가 오디오 길이를 읽는 중 시간 초과되었습니다: {path}") from exc
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"ffprobe가 오디오 길이를 읽지 못했습니다: {path}") from exc

    output = result.stdout.strip()
    try:
        duration = float(output)
    except ValueError as exc:
        raise RuntimeError(
            f"ffprobe가 잘못된 오디오 길이를 반환했습니다: {path}: {output!r}"
        ) from exc

    if not math.isfinite(duration) or duration < 0:
        raise RuntimeError(f"ffprobe가 잘못된 오디오 길이를 반환했습니다: {path}: {duration}")
    return duration
