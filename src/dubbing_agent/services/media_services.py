"""독립 미디어, 음성, 자막 서비스."""

from __future__ import annotations

import asyncio
import glob
import html
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any

from pydub import AudioSegment
from pydub.effects import speedup
from pydub.silence import detect_nonsilent


DEFAULT_TTS_RATE = "+0%"
MIN_AUTO_RATE_PERCENT = -20
MAX_AUTO_RATE_PERCENT = 40
SLOT_MARGIN_MS = 80
CLAUSE_GAP_MS = 40
EDGE_TTS_MAX_RETRIES = 5
EDGE_TTS_RETRY_BASE_SECONDS = 2.0
MIN_TARGET_OCCUPANCY = 0.42
MAX_TARGET_OCCUPANCY = 0.94
TARGET_OCCUPANCY_SLOPE = 0.03
TARGET_DURATION_TOLERANCE = 0.12
MIN_TARGET_DURATION_MS = 350
TTS_EDGE_SILENCE_KEEP_MS = 20
TTS_EDGE_MIN_SILENCE_MS = 60
TTS_EDGE_SILENCE_FLOOR_DBFS = -50.0
TTS_EDGE_SILENCE_RELATIVE_DBFS = 24.0
TTS_CACHE_VERSION = "adaptive-v2"

DEFAULT_VOICES = {
    "ko": "ko-KR-SunHiNeural",
    "en": "en-US-JennyNeural",
    "ja": "ja-JP-NanamiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "es": "es-ES-ElviraNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
}

_RATE_RE = re.compile(r"^[+-]?\d+%?$")
_CLAUSE_SPLIT_RE = re.compile(r"(?<=[,;:.!?])\s+")
_SENTENCE_END_RE = re.compile(r'[.!?…]["\')\]]*$')
_SHORT_FRAGMENT_RE = re.compile(r"^[^\w]*[\w'-]+(?:\s+[\w'-]+){0,2}[^\w]*$")
_VTT_TIMESTAMP_RE = re.compile(
    r"(?P<start>(?:\d+:)?\d{2}:\d{2}\.\d{3})\s+-->\s+"
    r"(?P<end>(?:\d+:)?\d{2}:\d{2}\.\d{3})"
)
_VTT_TAG_RE = re.compile(r"<[^>]+>")


def prepare_video_from_url(source_url: str, output_dir: str) -> str:
    """YouTube 영상을 다운로드하거나 기존 파일을 재사용해 로컬 경로를 반환한다."""
    import yt_dlp

    os.makedirs(output_dir, exist_ok=True)

    with yt_dlp.YoutubeDL(_build_ydl_opts(output_dir, quiet=True)) as ydl:
        info = ydl.extract_info(source_url, download=False)
        if isinstance(info, dict) and info.get("_type") != "playlist":
            for path in _candidate_output_paths(ydl, info):
                if os.path.exists(path):
                    return path
                cached_path = _cached_video_path_for_candidate(path, output_dir)
                if cached_path:
                    shutil.copy2(cached_path, path)
                    return path

    with yt_dlp.YoutubeDL(_build_ydl_opts(output_dir, quiet=False)) as ydl:
        info = ydl.extract_info(source_url, download=True)
        if isinstance(info, dict):
            for path in _candidate_output_paths(ydl, info):
                if os.path.exists(path):
                    return path

    raise RuntimeError("다운로더가 영상 파일 경로를 반환하지 않았습니다.")


def _cached_video_path_for_candidate(candidate_path: str, output_dir: str) -> str | None:
    """Find an existing same-named video in nearby job directories."""
    basename = os.path.basename(candidate_path)
    if not basename:
        return None

    roots = [
        os.path.dirname(output_dir),
        os.path.dirname(os.path.dirname(output_dir)),
    ]
    for root in dict.fromkeys(root for root in roots if root and os.path.isdir(root)):
        pattern = os.path.join(root, "**", basename)
        for path in sorted(glob.glob(pattern, recursive=True), reverse=True):
            if os.path.abspath(path) == os.path.abspath(candidate_path):
                continue
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                return path
    return None


def extract_source_audio(video_path: str, output_dir: str) -> str:
    """영상에서 원본 오디오를 16 kHz WAV로 추출한다."""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(output_dir, f"{base_name}_audio.wav")

    clip = _open_video_clip(video_path)
    try:
        clip.audio.write_audiofile(audio_path, fps=16000, nbytes=2, codec="pcm_s16le")
    finally:
        clip.close()

    return audio_path


def transcribe_and_normalize(
    audio_path: str,
    whisper_model: str,
    source_language: str | None,
    whisper_device: str = "auto",
) -> tuple[list[dict[str, Any]], str]:
    """Whisper STT를 실행하고 원본 세그먼트를 번역용으로 정규화한다."""
    transcription = _transcribe_with_whisper(
        audio_path,
        whisper_model,
        source_language,
        whisper_device,
    )
    normalized = normalize_segments(transcription["segments"])
    detected_language = transcription.get("language", source_language or "unknown")
    return normalized, detected_language


def _transcribe_with_whisper(
    audio_path: str,
    whisper_model: str,
    source_language: str | None,
    whisper_device: str,
) -> dict[str, Any]:
    """Run Whisper on the requested device and release GPU memory afterwards."""
    import gc
    import whisper

    device = _resolve_whisper_device(whisper_device)
    model = whisper.load_model(whisper_model, device=device)
    try:
        options: dict[str, Any] = {}
        if source_language:
            options["language"] = source_language
        if device == "cpu":
            options["fp16"] = False
        transcription = model.transcribe(audio_path, **options)
    finally:
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass

    return transcription


def _resolve_whisper_device(whisper_device: str) -> str:
    """Resolve the CLI device setting to a Whisper device string."""
    if whisper_device in {"cpu", "cuda"}:
        return whisper_device
    if whisper_device != "auto":
        raise ValueError(f"지원하지 않는 Whisper device입니다: {whisper_device}")

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def download_youtube_subtitle_segments(
    source_url: str,
    output_dir: str,
    language: str,
    subtitle_source: str,
) -> tuple[list[dict[str, Any]], str]:
    """YouTube 자막을 VTT로 내려받아 번역용 세그먼트로 변환한다."""
    import yt_dlp

    if not language:
        raise ValueError("자막 언어 코드가 필요합니다.")
    if subtitle_source not in {"manual", "automatic"}:
        raise ValueError(f"지원하지 않는 자막 출처입니다: {subtitle_source}")

    os.makedirs(output_dir, exist_ok=True)
    output_template = os.path.join(output_dir, "source_subtitle.%(ext)s")
    before = set(glob.glob(os.path.join(output_dir, "source_subtitle*")))
    opts = {
        "skip_download": True,
        "writesubtitles": subtitle_source == "manual",
        "writeautomaticsub": subtitle_source == "automatic",
        "subtitleslangs": [language],
        "subtitlesformat": "vtt/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.extract_info(source_url, download=True)

    subtitle_path = _find_downloaded_subtitle(output_dir, language, before)
    raw_segments = _parse_vtt_subtitles(subtitle_path)
    if not raw_segments:
        raise RuntimeError("다운로드한 자막에서 유효한 세그먼트를 찾지 못했습니다.")
    return normalize_segments(raw_segments), subtitle_path


def normalize_segments(
    segments: list[dict[str, Any]],
    max_gap: float = 0.45,
    max_duration: float = 12.0,
    max_chars: int = 140,
) -> list[dict[str, Any]]:
    """Whisper 세그먼트를 번역하기 좋은 발화 단위로 병합한다."""
    normalized: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for seg in segments:
        text = " ".join(str(seg.get("text", "")).split())
        if not text:
            continue

        candidate = {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": text,
        }

        if current is None:
            current = candidate
            continue

        if _should_merge_segment(current, candidate, max_gap, max_duration, max_chars):
            current["end"] = candidate["end"]
            current["text"] = _join_text(current["text"], candidate["text"])
            continue

        normalized.append(current)
        current = candidate

    if current is not None:
        normalized.append(current)

    return normalized


def _find_downloaded_subtitle(
    output_dir: str,
    language: str,
    before: set[str],
) -> str:
    """yt-dlp가 생성한 VTT 자막 파일 경로를 찾는다."""
    candidates = [
        path
        for path in glob.glob(os.path.join(output_dir, "source_subtitle*"))
        if path not in before and path.lower().endswith(".vtt")
    ]
    if not candidates:
        candidates = [
            path
            for path in glob.glob(os.path.join(output_dir, "source_subtitle*"))
            if path.lower().endswith(".vtt")
        ]
    language_marker = f".{language}."
    candidates.sort(key=lambda path: (language_marker not in path, len(path)))
    if not candidates:
        raise FileNotFoundError("yt-dlp가 VTT 자막 파일을 생성하지 않았습니다.")
    return candidates[0]


def _parse_vtt_subtitles(path: str) -> list[dict[str, Any]]:
    """VTT 자막 파일을 start/end/text 세그먼트 목록으로 파싱한다."""
    segments: list[dict[str, Any]] = []
    current_start: float | None = None
    current_end: float | None = None
    text_lines: list[str] = []

    with open(path, encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.strip()
            timestamp_match = _VTT_TIMESTAMP_RE.search(line)
            if timestamp_match:
                _append_vtt_segment(segments, current_start, current_end, text_lines)
                current_start = _vtt_time_to_seconds(timestamp_match.group("start"))
                current_end = _vtt_time_to_seconds(timestamp_match.group("end"))
                text_lines = []
                continue

            if not line:
                _append_vtt_segment(segments, current_start, current_end, text_lines)
                current_start = None
                current_end = None
                text_lines = []
                continue

            if _is_vtt_metadata_line(line) or current_start is None:
                continue
            text_lines.append(line)

    _append_vtt_segment(segments, current_start, current_end, text_lines)
    return segments


def _append_vtt_segment(
    segments: list[dict[str, Any]],
    start: float | None,
    end: float | None,
    text_lines: list[str],
) -> None:
    """파싱 중인 VTT cue 하나를 세그먼트 목록에 추가한다."""
    if start is None or end is None or end <= start:
        return
    text = _clean_vtt_text(" ".join(text_lines))
    if not text:
        return
    if segments and segments[-1]["text"] == text:
        return
    segments.append({"start": start, "end": end, "text": text})


def _clean_vtt_text(text: str) -> str:
    """VTT cue 텍스트에서 태그, entity, 중복 공백을 제거한다."""
    cleaned = _VTT_TAG_RE.sub("", text)
    cleaned = html.unescape(cleaned)
    return " ".join(cleaned.split())


def _is_vtt_metadata_line(line: str) -> bool:
    """VTT 본문 텍스트가 아닌 metadata 줄인지 판단한다."""
    return (
        line == "WEBVTT"
        or line.startswith(("Kind:", "Language:", "NOTE", "STYLE", "REGION"))
    )


def _vtt_time_to_seconds(value: str) -> float:
    """VTT 시간 문자열을 초 단위 float로 변환한다."""
    parts = value.split(":")
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds_part = parts[2]
    else:
        hours = 0
        minutes = int(parts[0])
        seconds_part = parts[1]
    seconds, millis = seconds_part.split(".")
    return hours * 3600 + minutes * 60 + int(seconds) + int(millis) / 1000


def synthesize_segment_tts(
    text: str,
    target_language: str,
    output_path: str,
    voice: str | None,
    rate: str,
    slot_duration_ms: int,
    *,
    allow_trimming: bool = True,
) -> str:
    """원본 세그먼트 길이에 맞춘 TTS 파일 하나를 생성한다."""
    cleaned_text = _cleanup_tts_text(text)
    if not cleaned_text:
        raise ValueError("TTS 텍스트가 비어 있습니다.")
    if not any(char.isalnum() for char in cleaned_text):
        raise ValueError("TTS로 읽을 수 있는 내용이 없습니다.")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    resolved_voice = voice or DEFAULT_VOICES.get(target_language, DEFAULT_VOICES["en"])
    normalized_rate = _normalize_rate(rate)
    return _synthesize_fitted_tts(
        cleaned_text,
        resolved_voice,
        normalized_rate,
        output_path,
        slot_duration_ms=slot_duration_ms,
        allow_trimming=allow_trimming,
    )


def write_subtitle_file(
    segments: list[dict[str, Any]],
    source_language: str,
    output_path: str,
    subtitle_mode: str,
    subtitle_format: str,
) -> str:
    """SRT 또는 ASS 자막을 생성한다."""
    if subtitle_format == "ass":
        return _generate_ass(segments, source_language, output_path)
    return _generate_srt(segments, source_language, output_path, subtitle_mode)


def compose_dubbed_audio(
    tts_segments: list[dict[str, Any]],
    total_duration: float,
    output_path: str,
    *,
    fit_overlong_segments: bool = True,
) -> str:
    """세그먼트 오디오를 timestamp에 맞춰 하나의 WAV 트랙으로 합성한다."""
    total_ms = int(total_duration * 1000)
    combined = AudioSegment.silent(duration=total_ms)

    for seg in tts_segments:
        start_ms = int(seg["start"] * 1000)
        tts_audio = AudioSegment.from_file(seg["audio_path"])
        max_duration_ms = int((seg["end"] - seg["start"]) * 1000)
        if max_duration_ms <= 0:
            continue

        if len(tts_audio) > max_duration_ms + 100:
            if not fit_overlong_segments:
                raise RuntimeError(
                    "TTS audio exceeds the composition slot after quality checks: "
                    f"audio={len(tts_audio) / 1000:.2f}s, "
                    f"slot={max_duration_ms / 1000:.2f}s, "
                    f"path={seg['audio_path']}"
                )
            playback_speed = min(max(len(tts_audio) / max_duration_ms, 1.01), 2.0)
            tts_audio = speedup(
                tts_audio,
                playback_speed=playback_speed,
                chunk_size=120,
                crossfade=20,
            )

        if len(tts_audio) > max_duration_ms + 100:
            tts_audio = tts_audio[: max_duration_ms + 100]

        combined = combined.overlay(tts_audio, position=start_ms)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    combined.export(output_path, format="wav")
    return output_path


def mux_dubbed_video(video_path: str, audio_path: str, output_path: str) -> str:
    """ffmpeg로 영상 오디오를 더빙 트랙으로 교체한다."""
    video = _open_video_clip(video_path)
    try:
        video_duration = float(video.duration)
    finally:
        video.close()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    root, ext = os.path.splitext(output_path)
    temp_output = f"{root}.tmp{ext}"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-filter_complex",
        f"[1:a]apad,atrim=0:{video_duration:.6f}[a]",
        "-map",
        "0:v:0",
        "-map",
        "[a]",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        temp_output,
    ]

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg 영상 합성 실패: {exc.stderr.strip()}") from exc

    os.replace(temp_output, output_path)
    return output_path


def get_video_duration(video_path: str) -> float:
    """영상 길이를 초 단위로 반환한다."""
    clip = _open_video_clip(video_path)
    try:
        return float(clip.duration)
    finally:
        clip.close()


def _build_ydl_opts(output_dir: str, quiet: bool) -> dict[str, Any]:
    """yt-dlp 실행에 사용할 공통 옵션을 만든다."""
    return {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": os.path.join(output_dir, "%(title)s.%(ext)s"),
        "merge_output_format": "mp4",
        "quiet": quiet,
        "no_warnings": quiet,
        "restrictfilenames": True,
    }


def _candidate_output_paths(ydl: Any, info: dict[str, Any]) -> list[str]:
    """yt-dlp 메타데이터로부터 다운로드 결과 후보 경로를 계산한다."""
    prepared = ydl.prepare_filename(info)
    base, _ = os.path.splitext(prepared)
    candidates = []
    merge_ext = ydl.params.get("merge_output_format")
    if merge_ext:
        candidates.append(f"{base}.{merge_ext}")
    candidates.append(prepared)

    unique_candidates = []
    for path in candidates:
        if path not in unique_candidates:
            unique_candidates.append(path)
    return unique_candidates


def _should_merge_segment(
    current: dict[str, Any],
    candidate: dict[str, Any],
    max_gap: float,
    max_duration: float,
    max_chars: int,
) -> bool:
    """두 Whisper 세그먼트를 하나의 발화로 병합해도 되는지 판단한다."""
    gap = candidate["start"] - current["end"]
    if gap > max_gap:
        return False

    merged_text = _join_text(current["text"], candidate["text"])
    merged_duration = candidate["end"] - current["start"]
    if merged_duration > max_duration or len(merged_text) > max_chars:
        return False

    current_text = current["text"]
    next_text = candidate["text"]
    if _ends_with_sentence(current_text):
        return _looks_like_fragment(next_text)
    if current_text.endswith((",", ";", ":", "-", "(", "[", "{", "/")):
        return True
    if _looks_like_fragment(next_text):
        return True
    return True


def _ends_with_sentence(text: str) -> bool:
    """문장이 종결 부호로 끝나는지 확인한다."""
    return bool(_SENTENCE_END_RE.search(text.strip()))


def _looks_like_fragment(text: str) -> bool:
    """텍스트가 앞 세그먼트에 붙이기 좋은 짧은 조각인지 판단한다."""
    stripped = text.strip()
    if not stripped:
        return False
    if stripped[0] in ",.;:!?)]}\"'":
        return True
    if len(stripped) <= 24 or len(stripped.split()) <= 3:
        return True
    return bool(_SHORT_FRAGMENT_RE.match(stripped))


def _join_text(left: str, right: str) -> str:
    """구두점과 괄호를 고려해 두 텍스트 조각을 자연스럽게 결합한다."""
    if not left:
        return right
    if not right:
        return left
    if right[0] in ",.;:!?)]}\"'":
        return f"{left}{right}"
    if left[-1] in "([{\"'":
        return f"{left}{right}"
    return f"{left} {right}"


def _synthesize_fitted_tts(
    text: str,
    voice: str,
    base_rate: str,
    output_path: str,
    slot_duration_ms: int | None = None,
    allow_trimming: bool = True,
    synthesize_func=None,
) -> str:
    """TTS를 생성한 뒤 목표 발화 구간에 맞도록 속도와 길이를 보정한다."""
    if synthesize_func is None:
        synthesize_func = _synthesize_edge_tts_to_file

    synthesize_func(text, voice, base_rate, output_path)
    if not slot_duration_ms:
        return output_path

    audio = _trim_edge_silence(_load_audio(output_path))
    if not allow_trimming:
        min_target_ms, _ = _target_duration_bounds(text, slot_duration_ms)
        if len(audio) < min_target_ms:
            target_rate = _target_rate_for_duration(
                len(audio),
                min_target_ms,
                min_target_ms,
                base_rate,
            )
            if _rate_to_int(target_rate) < _rate_to_int(base_rate):
                synthesize_func(text, voice, target_rate, output_path)
                audio = _trim_edge_silence(_load_audio(output_path))
        _export_audio(audio, output_path)
        return output_path

    min_target_ms, max_target_ms = _target_duration_bounds(text, slot_duration_ms)
    target_rate = _target_rate_for_duration(
        len(audio), min_target_ms, max_target_ms, base_rate
    )
    if target_rate != base_rate:
        synthesize_func(text, voice, target_rate, output_path)
        audio = _trim_edge_silence(_load_audio(output_path))

    if len(audio) > slot_duration_ms:
        fit_rate = _target_rate_for_slot(len(audio), slot_duration_ms, target_rate)
        if fit_rate != target_rate:
            target_rate = fit_rate
            synthesize_func(text, voice, target_rate, output_path)
            audio = _trim_edge_silence(_load_audio(output_path))

    if len(audio) <= slot_duration_ms:
        _export_audio(audio, output_path)
        return output_path

    clause_audio = _try_clause_synthesis(
        text,
        voice,
        target_rate,
        output_path,
        synthesize_func,
    )
    if clause_audio is not None:
        audio = clause_audio

    if len(audio) > slot_duration_ms:
        playback_speed = min(max(len(audio) / slot_duration_ms, 1.01), 2.0)
        audio = speedup(audio, playback_speed=playback_speed, chunk_size=120, crossfade=20)

    if allow_trimming and len(audio) > slot_duration_ms + 100:
        audio = audio[: slot_duration_ms + 100]

    _export_audio(audio, output_path)
    return output_path


def _try_clause_synthesis(
    text: str,
    voice: str,
    rate: str,
    output_path: str,
    synthesize_func,
) -> AudioSegment | None:
    """문장을 절 단위로 나눠 합성해 긴 pause를 줄일 수 있는지 시도한다."""
    clauses = [part.strip() for part in _CLAUSE_SPLIT_RE.split(text) if part.strip()]
    if len(clauses) <= 1:
        return None

    combined = AudioSegment.empty()
    suffix = os.path.splitext(output_path)[1].lstrip(".") or "mp3"
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, clause in enumerate(clauses):
            clause_path = os.path.join(tmp_dir, f"clause_{i:02d}.{suffix}")
            synthesize_func(clause, voice, rate, clause_path)
            combined += _load_audio(clause_path)
            if i != len(clauses) - 1:
                combined += AudioSegment.silent(duration=CLAUSE_GAP_MS)
    return combined


def _synthesize_edge_tts_to_file(text: str, voice: str, rate: str, output_path: str) -> None:
    """edge-tts로 음성을 생성하고 일시 오류는 제한 횟수만큼 재시도한다."""
    try:
        import edge_tts
    except ImportError as exc:
        raise RuntimeError("TTS 생성을 위해 edge-tts가 필요합니다.") from exc

    async def _save() -> None:
        """edge-tts 비동기 저장 호출을 감싼다."""
        communicator = edge_tts.Communicate(text=text, voice=voice, rate=rate)
        await communicator.save(output_path)

    last_error = None
    for attempt in range(1, EDGE_TTS_MAX_RETRIES + 1):
        try:
            asyncio.run(_save())
            return
        except Exception as exc:
            last_error = exc
            _remove_partial_output(output_path)
            if attempt == EDGE_TTS_MAX_RETRIES or not _is_retryable_tts_error(exc):
                raise
            delay = EDGE_TTS_RETRY_BASE_SECONDS * attempt
            print(
                f"    TTS 일시 오류로 재시도 중... "
                f"[{attempt}/{EDGE_TTS_MAX_RETRIES}] {exc.__class__.__name__}"
            )
            time.sleep(delay)

    if last_error is not None:
        raise last_error


def _cleanup_tts_text(text: str) -> str:
    """TTS에 넘기기 전에 JSON/마크다운 wrapper와 불필요한 공백을 제거한다."""
    cleaned = text.strip()
    if not cleaned:
        return ""

    if "```" in cleaned:
        parts = cleaned.split("```")
        for block in parts:
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            if block:
                cleaned = block
                break

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, str):
        return " ".join(parsed.split())

    if isinstance(parsed, list):
        for item in parsed:
            candidate = _cleanup_tts_text(str(item))
            if candidate:
                return candidate
        return ""

    return " ".join(cleaned.split())


def _normalize_rate(rate: str | None) -> str:
    """사용자 입력 말하기 속도를 edge-tts가 기대하는 %+/- 형식으로 정규화한다."""
    if not rate:
        return DEFAULT_TTS_RATE

    cleaned = rate.strip()
    if not _RATE_RE.match(cleaned):
        raise ValueError(f"잘못된 TTS 속도 형식입니다: {rate}")

    number = int(cleaned[:-1] if cleaned.endswith("%") else cleaned)
    return _int_to_rate(number)


def _target_duration_bounds(text: str, slot_ms: int) -> tuple[int, int]:
    """텍스트 밀도와 슬롯 길이를 바탕으로 목표 TTS 길이 범위를 계산한다."""
    text_units = _estimate_speech_units(text)
    density = text_units / max(slot_ms / 1000, 0.1)
    occupancy = MIN_TARGET_OCCUPANCY + min(
        MAX_TARGET_OCCUPANCY - MIN_TARGET_OCCUPANCY,
        density * TARGET_OCCUPANCY_SLOPE,
    )
    target_ms = int(slot_ms * occupancy)
    target_ms = min(slot_ms, max(min(slot_ms, MIN_TARGET_DURATION_MS), target_ms))
    min_target_ms = max(200, int(target_ms * (1 - TARGET_DURATION_TOLERANCE)))
    max_target_ms = min(slot_ms, int(target_ms * (1 + TARGET_DURATION_TOLERANCE)))
    return min_target_ms, max(min_target_ms, max_target_ms)


def _estimate_speech_units(text: str) -> int:
    """텍스트 길이와 구두점을 이용해 대략적인 발화 단위를 추정한다."""
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return 1
    punctuation_weight = sum(1 for ch in text if ch in ",;:.!?") * 2
    return len(compact) + punctuation_weight


def _target_rate_for_duration(
    current_ms: int,
    min_target_ms: int,
    max_target_ms: int,
    base_rate: str,
) -> str:
    """현재 TTS 길이가 목표 범위에 들어오도록 새 말하기 속도를 계산한다."""
    if min_target_ms <= current_ms <= max_target_ms:
        return base_rate

    current_percent = _rate_to_int(base_rate)
    if current_ms < min_target_ms:
        adjustment = math.floor((current_ms / max(min_target_ms, 1) - 1) * 100)
        return _int_to_rate(max(MIN_AUTO_RATE_PERCENT, current_percent + adjustment))

    adjustment = math.ceil((current_ms / max(max_target_ms, 1) - 1) * 100)
    return _int_to_rate(min(MAX_AUTO_RATE_PERCENT, current_percent + adjustment))


def _target_rate_for_slot(current_ms: int, slot_ms: int, base_rate: str) -> str:
    """TTS가 슬롯을 초과할 때 슬롯 안에 맞추기 위한 말하기 속도를 계산한다."""
    needed_speedup = math.ceil((current_ms / slot_ms - 1) * 100)
    current_percent = _rate_to_int(base_rate)
    boosted_percent = min(
        MAX_AUTO_RATE_PERCENT,
        max(current_percent, current_percent + needed_speedup),
    )
    return _int_to_rate(boosted_percent)


def _rate_to_int(rate: str) -> int:
    """'+10%' 형식의 속도를 정수 퍼센트로 변환한다."""
    return int(_normalize_rate(rate)[:-1])


def _int_to_rate(value: int) -> str:
    """정수 퍼센트를 edge-tts 속도 문자열로 변환한다."""
    sign = "+" if value >= 0 else ""
    return f"{sign}{value}%"


def _export_audio(audio: AudioSegment, output_path: str) -> None:
    """AudioSegment를 출력 파일 확장자에 맞춰 저장한다."""
    ext = os.path.splitext(output_path)[1].lstrip(".") or "mp3"
    handle = audio.export(output_path, format=ext)
    handle.close()


def _trim_edge_silence(audio: AudioSegment) -> AudioSegment:
    """Trim leading/trailing synthesis silence while preserving a tiny edge pad."""
    if not audio or len(audio) <= 0:
        return audio

    if audio.dBFS == float("-inf"):
        return audio

    silence_threshold = max(
        TTS_EDGE_SILENCE_FLOOR_DBFS,
        audio.dBFS - TTS_EDGE_SILENCE_RELATIVE_DBFS,
    )
    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=TTS_EDGE_MIN_SILENCE_MS,
        silence_thresh=silence_threshold,
        seek_step=5,
    )
    if not nonsilent_ranges:
        return audio

    start = max(0, nonsilent_ranges[0][0] - TTS_EDGE_SILENCE_KEEP_MS)
    end = min(len(audio), nonsilent_ranges[-1][1] + TTS_EDGE_SILENCE_KEEP_MS)
    if start <= 0 and end >= len(audio):
        return audio
    return audio[start:end]


def _remove_partial_output(path: str) -> None:
    """실패한 합성에서 남은 부분 출력 파일을 제거한다."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass


def _is_retryable_tts_error(exc: Exception) -> bool:
    """TTS 예외가 재시도 가능한 일시 오류인지 판단한다."""
    status = getattr(exc, "status", None)
    if status in {429, 500, 502, 503, 504}:
        return True
    message = str(exc).lower()
    class_name = exc.__class__.__name__.lower()
    return any(
        token in f"{class_name} {message}"
        for token in (
            "503",
            "tempor",
            "timeout",
            "handshake",
            "connection reset",
            "service unavailable",
            "noaudioreceived",
            "no audio was received",
        )
    )


def _load_audio(path: str) -> AudioSegment:
    """오디오 파일을 확장자 기반 포맷으로 로드한다."""
    ext = os.path.splitext(path)[1].lstrip(".") or None
    with open(path, "rb") as f:
        return AudioSegment.from_file(f, format=ext)


def _generate_srt(
    segments: list[dict[str, Any]],
    source_language: str,
    output_path: str,
    mode: str,
) -> str:
    """세그먼트 목록으로 SRT 자막 파일을 생성한다."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    lines = []
    for i, seg in enumerate(segments, 1):
        lines.append(str(i))
        lines.append(
            f"{_seconds_to_srt_time(seg['start'])} --> {_seconds_to_srt_time(seg['end'])}"
        )
        if mode == "full":
            original = seg["original"]
            lines.append(original)
            lines.append(romanize(original, source_language))
            lines.append(seg["translated"])
        elif mode == "translated":
            lines.append(seg["translated"])
        else:
            lines.append(seg["original"])
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return output_path


def _generate_ass(
    segments: list[dict[str, Any]],
    source_language: str,
    output_path: str,
) -> str:
    """세그먼트 목록으로 ASS 자막 파일을 생성한다."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    header = """[Script Info]
Title: 더빙 에이전트 자막
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Original,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,1,2,10,10,120,1
Style: Romanized,Arial,40,&H0000FFFF,&H000000FF,&H00000000,&H80000000,0,1,0,0,100,100,0,0,1,2,1,2,10,10,70,1
Style: Translated,Arial,44,&H0000FF00,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,1,2,10,10,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for seg in segments:
        start_tc = _seconds_to_ass_time(seg["start"])
        end_tc = _seconds_to_ass_time(seg["end"])
        original = seg["original"].replace("\n", "\\N")
        pronunciation = romanize(original, source_language).replace("\n", "\\N")
        translated = seg["translated"].replace("\n", "\\N")
        events.append(f"Dialogue: 0,{start_tc},{end_tc},Original,,0,0,0,,{original}")
        events.append(
            f"Dialogue: 0,{start_tc},{end_tc},Romanized,,0,0,0,,{pronunciation}"
        )
        events.append(
            f"Dialogue: 0,{start_tc},{end_tc},Translated,,0,0,0,,{translated}"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(events))
        f.write("\n")
    return output_path


def romanize(text: str, language: str) -> str:
    """자막 표시용 발음 표기 줄을 반환한다."""
    if not text.strip():
        return ""
    if language == "ja":
        import pykakasi

        kks = pykakasi.kakasi()
        result = kks.convert(text)
        return " ".join(item["hepburn"] for item in result if item["hepburn"])
    if language == "ko":
        from hangul_romanize import Transliter
        from hangul_romanize.rule import academic

        return Transliter(academic).translit(text)
    if language == "zh":
        try:
            from pypinyin import Style, pinyin

            result = pinyin(text, style=Style.TONE)
            return " ".join(item[0] for item in result)
        except ImportError:
            return text
    if language in ("ru", "uk", "bg", "mk", "sr"):
        from transliterate import translit

        try:
            return translit(text, reversed=True)
        except Exception:
            return text
    return text


def _seconds_to_srt_time(seconds: float) -> str:
    """초 단위 시간을 SRT 타임코드 형식으로 변환한다."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _seconds_to_ass_time(seconds: float) -> str:
    """초 단위 시간을 ASS 타임코드 형식으로 변환한다."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _open_video_clip(video_path: str):
    """설치된 moviepy 버전에 맞춰 VideoFileClip을 열어 반환한다."""
    try:
        from moviepy import VideoFileClip
    except ImportError:
        from moviepy.editor import VideoFileClip

    return VideoFileClip(video_path)
