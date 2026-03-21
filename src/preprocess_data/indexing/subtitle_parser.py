import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

try:
    import pysrt
except ImportError:
    pysrt = None

from preprocess_data.config import PreprocessConfig as Cfg

logger = logging.getLogger(__name__)

_TIMECODE_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}[,.]\d{3})"
)


@dataclass
class SubtitleEntry:
    start_sec: float
    end_sec: float
    text: str


def _parse_timestamp(ts: str) -> float:
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    if len(parts) != 3:
        return 0.0
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text.replace("\n", " "))
    return text.strip()


def _read_srt_text(srt_path: Path) -> str:
    for encoding in ("utf-8-sig", "cp1252", "utf-8"):
        try:
            return srt_path.read_text(encoding=encoding).strip()
        except (UnicodeDecodeError, LookupError):
            continue
    return srt_path.read_text(encoding="utf-8", errors="replace").strip()


def _parse_srt_text(text: str) -> List[SubtitleEntry]:
    if not text:
        return []

    entries: List[SubtitleEntry] = []
    blocks = re.split(r"\r?\n\s*\r?\n", text)
    for block in blocks:
        lines = [line.strip("\ufeff").strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        time_line_index = None
        for idx, line in enumerate(lines[:2]):
            if _TIMECODE_RE.match(line):
                time_line_index = idx
                break
        if time_line_index is None:
            continue

        match = _TIMECODE_RE.match(lines[time_line_index])
        if not match:
            continue

        text_lines = lines[time_line_index + 1 :]
        if not text_lines:
            continue

        clean_text = _clean_text(" ".join(text_lines))
        if not clean_text:
            continue

        entries.append(
            SubtitleEntry(
                start_sec=_parse_timestamp(match.group("start")),
                end_sec=_parse_timestamp(match.group("end")),
                text=clean_text,
            )
        )

    return entries


def _parse_with_pysrt(srt_path: Path) -> List[SubtitleEntry]:
    if pysrt is None:
        return []

    try:
        subs = pysrt.open(str(srt_path))
    except Exception as exc:
        logger.debug(f"  pysrt failed for {srt_path}: {exc}")
        return []

    entries: List[SubtitleEntry] = []
    for sub in subs:
        clean_text = _clean_text(getattr(sub, "text", ""))
        if not clean_text:
            continue
        entries.append(
            SubtitleEntry(
                start_sec=float(sub.start.ordinal) / 1000.0,
                end_sec=float(sub.end.ordinal) / 1000.0,
                text=clean_text,
            )
        )
    return entries

class SubtitleParser:
    """Parses SRT files and aligns text with temporal boundaries."""
    
    @staticmethod
    def load_for_movie(movie_id: str) -> List[SubtitleEntry]:
        srt_path = Cfg.get_subtitle_dir() / f"{movie_id}.srt"
        if not srt_path.exists():
            logger.warning(f"  Missing Subtitle file at {srt_path}")
            return []

        try:
            pysrt_entries = _parse_with_pysrt(srt_path)
            if pysrt_entries:
                return pysrt_entries

            return _parse_srt_text(_read_srt_text(srt_path))
        except Exception as e:
            logger.error(f"  Failed parsing {srt_path}: {e}")
            return []
             
    @staticmethod
    def align(
        srt: Optional[Iterable[SubtitleEntry]], start_sec: float, end_sec: float
    ) -> List[str]:
        """Find subtitles falling within a temporal window."""
        if not srt:
            return []

        dialogues = []
        try:
            for sub in srt:
                if sub.end_sec < start_sec or sub.start_sec > end_sec:
                    continue
                if sub.text:
                    dialogues.append(sub.text)
        except Exception as e:
            logger.debug(f"Subtitle slice error: {e}")

        return dialogues
