"""
ScriptAligner - Hybrid Linear + Semantic-Window Anchor Alignment

Aligns screenplay scenes onto the video timeline with a two-pass strategy:
1. Linear estimate from screenplay character progress.
2. Subtitle anchor search constrained to the nearest semantic-scene window
   when that window already exists in annotation.

The semantic window is used as a hint, not as a hard scene boundary.
"""

from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


ANCHOR_WINDOW_SEC = 120.0
MIN_ANCHOR_SCORE = 0.60
MAX_SCENE_CONTEXT_CHARS = 800
SEMANTIC_SEARCH_MARGIN_SEC = 20.0
SUBSTANTIAL_DIALOGUE_MIN_CHARS = 18
SUBSTANTIAL_DIALOGUE_MIN_WORDS = 4
MAX_SUBTITLE_WINDOW_ENTRIES = 3


class ScriptScene:
    """A single screenplay scene with temporal alignment metadata."""

    __slots__ = (
        "scene_uid",
        "scene_num",
        "heading",
        "location",
        "time_of_day",
        "characters",
        "action_lines",
        "dialogue_lines",
        "start_sec",
        "end_sec",
        "anchor_quality",
        "confidence_score",
        "anchor_start_sec",
        "anchor_end_sec",
        "linear_start_sec",
        "linear_end_sec",
    )

    def __init__(
        self,
        scene_uid: str,
        scene_num: Optional[int],
        heading: str,
        blocks: List[Dict],
        start_sec: float = 0.0,
        end_sec: float = 0.0,
        anchor_quality: str = "unset",
        confidence_score: float = 0.0,
        anchor_start_sec: Optional[float] = None,
        anchor_end_sec: Optional[float] = None,
        linear_start_sec: float = 0.0,
        linear_end_sec: float = 0.0,
    ):
        self.scene_uid = scene_uid
        self.scene_num = scene_num
        self.heading = heading

        parts = heading.strip().split("-")
        self.location = parts[0].strip() if parts else heading
        self.time_of_day = parts[-1].strip() if len(parts) > 1 else "UNKNOWN"

        self.characters = sorted(
            {b["character"] for b in blocks if b["type"] == "Dialogue"}
        )
        self.action_lines = [b["text"] for b in blocks if b["type"] == "Action"]
        self.dialogue_lines = [
            {"char": b["character"], "text": b["text"]}
            for b in blocks
            if b["type"] == "Dialogue"
        ]

        self.start_sec = start_sec
        self.end_sec = end_sec
        self.anchor_quality = anchor_quality
        self.confidence_score = confidence_score
        self.anchor_start_sec = anchor_start_sec
        self.anchor_end_sec = anchor_end_sec
        self.linear_start_sec = linear_start_sec
        self.linear_end_sec = linear_end_sec

    def as_context_str(self, max_chars: int = MAX_SCENE_CONTEXT_CHARS) -> str:
        """Return a compact text description suitable for LLM injection."""
        parts = [f"[SCENE {self.heading}]"]
        if self.characters:
            parts.append(f"Characters: {', '.join(self.characters)}")

        written = 0
        for dl in self.dialogue_lines:
            line = f"  {dl['char']}: {dl['text']}"
            if written + len(line) > max_chars:
                parts.append("  ...")
                break
            parts.append(line)
            written += len(line)

        if not self.dialogue_lines and self.action_lines:
            action_line = self.action_lines[0][:200]
            parts.append(f"  {action_line}")

        return "\n".join(parts)

    def to_dict(self) -> Dict:
        return {
            "scene_uid": self.scene_uid,
            "scene_num": self.scene_num,
            "heading": self.heading,
            "location": self.location,
            "time_of_day": self.time_of_day,
            "characters": self.characters,
            "action_lines": self.action_lines,
            "dialogue_lines": self.dialogue_lines,
            "start_sec": round(self.start_sec, 2),
            "end_sec": round(self.end_sec, 2),
            "anchor_quality": self.anchor_quality,
            "confidence_score": round(float(self.confidence_score or 0.0), 3),
            "anchor_start_sec": round(self.anchor_start_sec, 2)
            if self.anchor_start_sec is not None
            else None,
            "anchor_end_sec": round(self.anchor_end_sec, 2)
            if self.anchor_end_sec is not None
            else None,
            "linear_start_sec": round(self.linear_start_sec, 2),
            "linear_end_sec": round(self.linear_end_sec, 2),
        }


def _parse_screenplay(script_text: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Parse a raw IMSDb screenplay text into blocks and grouped scenes.
    """
    lines = script_text.splitlines()
    blocks: List[Dict] = []

    scene_pat = re.compile(
        r"^\s*(\d+)?\s*(INT\.?|EXT\.?|INT/EXT\.?|EXT/INT\.?|I/E)\s*[\.\-\s].{2,}",
        re.IGNORECASE,
    )

    mode = "action"
    current_char: Optional[str] = None
    buf: List[str] = []

    def flush() -> None:
        nonlocal current_char, buf
        text = " ".join(buf).strip()
        if text:
            if mode == "dialogue" and current_char:
                blocks.append(
                    {"type": "Dialogue", "character": current_char, "text": text}
                )
            elif mode == "action":
                blocks.append({"type": "Action", "text": text})
        buf.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if scene_pat.match(line):
            flush()
            current_char = None
            mode = "action"
            match = re.match(
                r"^\s*(\d+)?\s*((?:INT|EXT|INT/EXT|EXT/INT|I/E)[^\n]+)",
                line,
                re.IGNORECASE,
            )
            scene_num = int(match.group(1)) if match and match.group(1) else None
            heading = match.group(2).strip() if match else stripped
            blocks.append(
                {"type": "SceneHeading", "text": heading, "scene_num": scene_num}
            )
            continue

        leading = len(line) - len(line.lstrip())
        is_upper = stripped == stripped.upper() and bool(re.search(r"[A-Z]", stripped))
        is_short = len(stripped) <= 42

        if 10 <= leading <= 45 and is_upper and is_short and len(stripped.split()) <= 5:
            if "." not in stripped[1:] or "(" in stripped:
                flush()
                current_char = stripped
                mode = "dialogue"
                buf.clear()
                continue

        if mode == "dialogue":
            if leading >= 8:
                buf.append(stripped)
            else:
                flush()
                mode = "action"
                current_char = None
                buf.append(stripped)
        else:
            buf.append(stripped)

    flush()

    raw_scenes: List[Dict] = []
    current: Optional[Dict] = None
    for block in blocks:
        if block["type"] == "SceneHeading":
            if current:
                raw_scenes.append(current)
            current = {
                "heading": block["text"],
                "scene_num": block.get("scene_num"),
                "blocks": [],
            }
        elif current:
            current["blocks"].append(block)
    if current:
        raw_scenes.append(current)

    return blocks, raw_scenes


def _parse_srt(srt_path: Path) -> List[Dict]:
    """Parse SRT into list of {start, end, text} records."""
    if not srt_path or not srt_path.exists():
        return []

    text = srt_path.read_text(encoding="utf-8-sig", errors="replace")
    entries = []
    time_pat = re.compile(
        r"(\d{2}:\d{2}:\d{2}[,\.]\d{3}) --> (\d{2}:\d{2}:\d{2}[,\.]\d{3})"
    )

    for block in re.split(r"\n{2,}", text.strip()):
        match = time_pat.search(block)
        if not match:
            continue

        def to_sec(value: str) -> float:
            h, minutes, rest = value.replace(",", ".").split(":", 2)
            return int(h) * 3600 + int(minutes) * 60 + float(rest)

        start = to_sec(match.group(1))
        end = to_sec(match.group(2))
        text_lines = [
            line.strip()
            for line in block.splitlines()
            if line.strip() and not line.strip().isdigit() and "-->" not in line
        ]
        subtitle_text = re.sub(r"<[^>]+>", "", " ".join(text_lines)).strip()
        if subtitle_text:
            entries.append({"start": start, "end": end, "text": subtitle_text})

    return entries


def _clean(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def _fuzzy_score(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def _linear_estimate(
    raw_scenes: List[Dict], total_duration: float, total_chars: int
) -> List[Tuple[float, float]]:
    """Estimate start/end for each screenplay scene by relative script position."""
    cumulative = [0]
    for scene in raw_scenes:
        scene_chars = len(scene["heading"]) + sum(len(b["text"]) for b in scene["blocks"])
        cumulative.append(cumulative[-1] + scene_chars)

    total = cumulative[-1] or total_chars or 1
    return [
        (
            cumulative[i] / total * total_duration,
            cumulative[i + 1] / total * total_duration,
        )
        for i in range(len(raw_scenes))
    ]


class ScriptAligner:
    """
    Align a movie screenplay to the video timeline and provide context lookups.

    Workflow:
        1. align(movie_id)         -> List[ScriptScene]
        2. get_script_context(...) -> str for local LLM context
    """

    def __init__(self):
        from preprocess_data.config import PreprocessConfig as Cfg

        self._cfg = Cfg

    def align(self, movie_id: str, force: bool = False) -> List[ScriptScene]:
        """Run screenplay alignment. Uses cache unless force=True."""
        cache_path = self._get_cache_path(movie_id)
        if not force:
            for candidate in self._get_cache_candidates(movie_id):
                if candidate.exists():
                    logger.info(f"  Script alignment cache hit: {movie_id}")
                    scenes = self._load_cache(candidate)
                    if scenes and candidate != cache_path:
                        self._save_cache(scenes, cache_path)
                    return scenes

        script_path = self._find_script(movie_id)
        srt_path = self._find_srt(movie_id)

        if not script_path or not script_path.exists():
            logger.info(f"  No script found for {movie_id}, skipping alignment.")
            return []

        logger.info(f"  Aligning script for {movie_id}...")
        script_text = script_path.read_text(encoding="utf-8", errors="replace")
        _, raw_scenes = _parse_screenplay(script_text)
        if not raw_scenes:
            logger.warning(f"  Parser found 0 scenes in script for {movie_id}.")
            return []

        srt_entries = _parse_srt(srt_path) if srt_path else []
        total_duration = srt_entries[-1]["end"] if srt_entries else 7200.0
        total_chars = len(script_text)
        linear_times = _linear_estimate(raw_scenes, total_duration, total_chars)
        semantic_windows = self._load_semantic_windows(movie_id)

        aligned: List[ScriptScene] = []
        prev_confirmed_end = 0.0

        for i, scene in enumerate(raw_scenes):
            linear_start, linear_end = linear_times[i]
            dialogues = [b for b in scene["blocks"] if b["type"] == "Dialogue"]
            semantic_window = self._pick_semantic_window(
                linear_start, linear_end, semantic_windows
            )

            window_start = linear_start
            window_end = linear_end
            if semantic_window:
                window_start = max(0.0, semantic_window[0] - SEMANTIC_SEARCH_MARGIN_SEC)
                window_end = semantic_window[1] + SEMANTIC_SEARCH_MARGIN_SEC

            start_anchor = self._find_anchor_timestamp(
                self._build_dialogue_candidates(dialogues, from_end=False),
                srt_entries,
                max(prev_confirmed_end, window_start),
                window_end,
                prefer_end=False,
            )
            effective_start = (
                round(start_anchor[0], 3)
                if start_anchor and start_anchor[1] >= MIN_ANCHOR_SCORE
                else max(prev_confirmed_end, linear_start)
            )
            end_anchor = self._find_anchor_timestamp(
                self._build_dialogue_candidates(dialogues, from_end=True),
                srt_entries,
                max(effective_start, window_start),
                window_end,
                prefer_end=True,
            )

            anchor_start_sec = (
                round(start_anchor[0], 3)
                if start_anchor and start_anchor[1] >= MIN_ANCHOR_SCORE
                else None
            )
            anchor_end_sec = (
                round(end_anchor[0], 3)
                if end_anchor and end_anchor[1] >= MIN_ANCHOR_SCORE
                else None
            )

            if (
                anchor_start_sec is not None
                and anchor_end_sec is not None
                and anchor_end_sec <= anchor_start_sec
            ):
                anchor_end_sec = None

            anchor_quality = "linear"
            if anchor_start_sec is not None and anchor_end_sec is not None:
                anchor_quality = "full"
            elif anchor_start_sec is not None or anchor_end_sec is not None:
                anchor_quality = "partial"

            start_sec = anchor_start_sec if anchor_start_sec is not None else linear_start
            start_sec = max(start_sec, prev_confirmed_end)

            end_sec = anchor_end_sec if anchor_end_sec is not None else linear_end
            end_sec = max(end_sec, start_sec + 1.0)
            end_sec = min(end_sec, total_duration)

            prev_confirmed_end = end_sec

            confidence_score = self._compute_confidence(
                anchor_quality=anchor_quality,
                start_score=start_anchor[1] if start_anchor else None,
                end_score=end_anchor[1] if end_anchor else None,
                semantic_window=semantic_window,
            )

            aligned.append(
                ScriptScene(
                    scene_uid=f"{movie_id}_script_scene_{i:04d}",
                    scene_num=scene["scene_num"],
                    heading=scene["heading"],
                    blocks=scene["blocks"],
                    start_sec=start_sec,
                    end_sec=end_sec,
                    anchor_quality=anchor_quality,
                    confidence_score=confidence_score,
                    anchor_start_sec=anchor_start_sec,
                    anchor_end_sec=anchor_end_sec,
                    linear_start_sec=linear_start,
                    linear_end_sec=linear_end,
                )
            )

        if aligned:
            aligned[-1].end_sec = max(aligned[-1].end_sec, total_duration)
            aligned[-1].linear_end_sec = max(aligned[-1].linear_end_sec, total_duration)

        full = sum(1 for s in aligned if s.anchor_quality == "full")
        partial = sum(1 for s in aligned if s.anchor_quality == "partial")
        linear = sum(1 for s in aligned if s.anchor_quality == "linear")
        avg_conf = (
            sum(s.confidence_score for s in aligned) / len(aligned) if aligned else 0.0
        )
        logger.info(
            "  Script alignment complete: %s scenes (full=%s, partial=%s, linear=%s, avg_conf=%.2f)",
            len(aligned),
            full,
            partial,
            linear,
            avg_conf,
        )

        self._save_cache(aligned, cache_path)
        return aligned

    def _get_cache_path(self, movie_id: str) -> Path:
        return self._cfg.get_script_alignment_dir() / f"{movie_id}_script_aligned.json"

    def _get_cache_candidates(self, movie_id: str) -> List[Path]:
        candidates = [
            self._get_cache_path(movie_id),
            self._cfg.get_meta_dir() / f"{movie_id}_script_aligned.json",
        ]
        seen = set()
        ordered: List[Path] = []
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            ordered.append(candidate)
        return ordered

    def get_script_context(
        self,
        aligned_scenes: List[ScriptScene],
        start_sec: float,
        end_sec: float,
        max_chars: int = 2500,
    ) -> str:
        """Extract screenplay context for a given [start_sec, end_sec] range."""
        if not aligned_scenes:
            return ""

        overlapping = [
            scene
            for scene in aligned_scenes
            if scene.end_sec > start_sec and scene.start_sec < end_sec
        ]
        if not overlapping:
            nearest = min(aligned_scenes, key=lambda s: abs(s.start_sec - start_sec))
            overlapping = [nearest]

        parts = ["=== SCREENPLAY CONTEXT ==="]
        total_chars = 0
        per_scene_budget = max(200, max_chars // max(len(overlapping), 1))

        for scene in overlapping:
            scene_text = scene.as_context_str(max_chars=per_scene_budget)
            total_chars += len(scene_text)
            parts.append(scene_text)
            if total_chars >= max_chars:
                parts.append("(more scenes omitted for brevity)")
                break

        parts.append("===========================")
        return "\n".join(parts)

    def summarize_arc(
        self, aligned_scenes: List[ScriptScene], max_scenes: int = 10
    ) -> str:
        """Build a compact macro-level narrative arc summary."""
        if not aligned_scenes:
            return ""

        step = max(1, len(aligned_scenes) // max_scenes)
        sampled = aligned_scenes[::step][:max_scenes]

        lines = ["=== NARRATIVE ARC ==="]
        for scene in sampled:
            ts = f"{scene.start_sec / 60:.1f}min"
            chars = ", ".join(scene.characters[:4]) if scene.characters else "-"
            lines.append(
                f"  [{ts}] {scene.heading[:60]} | {chars} | {scene.anchor_quality}:{scene.confidence_score:.2f}"
            )
        lines.append("=====================")
        return "\n".join(lines)

    def _save_cache(self, scenes: List[ScriptScene], path: Path) -> None:
        try:
            data = [scene.to_dict() for scene in scenes]
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning(f"  Could not save script alignment cache: {exc}")

    def _load_cache(self, path: Path) -> List[ScriptScene]:
        """Re-hydrate ScriptScene instances from cache JSON."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            scenes: List[ScriptScene] = []
            movie_id = path.stem.replace("_script_aligned", "")
            for idx, item in enumerate(data):
                scene = ScriptScene.__new__(ScriptScene)
                scene.scene_uid = item.get(
                    "scene_uid", f"{movie_id}_script_scene_{idx:04d}"
                )
                scene.scene_num = item.get("scene_num")
                scene.heading = item.get("heading", "")
                scene.location = item.get("location", "")
                scene.time_of_day = item.get("time_of_day", "")
                scene.characters = list(item.get("characters", []))
                scene.action_lines = list(item.get("action_lines", []))
                scene.dialogue_lines = list(item.get("dialogue_lines", []))
                scene.start_sec = float(item.get("start_sec", 0.0))
                scene.end_sec = float(item.get("end_sec", 0.0))
                scene.anchor_quality = item.get("anchor_quality", "cached")
                scene.confidence_score = float(item.get("confidence_score", 0.0))
                scene.anchor_start_sec = item.get("anchor_start_sec")
                scene.anchor_end_sec = item.get("anchor_end_sec")
                scene.linear_start_sec = float(
                    item.get("linear_start_sec", item.get("start_sec", 0.0))
                )
                scene.linear_end_sec = float(
                    item.get("linear_end_sec", item.get("end_sec", 0.0))
                )
                scenes.append(scene)
            return scenes
        except Exception as exc:
            logger.warning(f"  Failed to load script alignment cache: {exc}")
            return []

    def _find_srt(self, movie_id: str) -> Optional[Path]:
        candidates = [
            self._cfg.get_subtitle_dir() / f"{movie_id}.srt",
            self._cfg.MOVIENET_SUBSET_DIR / "subtitle" / f"{movie_id}.srt",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _find_script(self, movie_id: str) -> Optional[Path]:
        candidates = [
            self._cfg.get_script_dir() / f"{movie_id}.script",
            self._cfg.MOVIENET_SUBSET_DIR / "script" / f"{movie_id}.script",
        ]
        for path in candidates:
            if path.exists():
                return path
        return None

    def _load_semantic_windows(self, movie_id: str) -> List[Tuple[float, float]]:
        annotation_path = self._cfg.get_annotation_dir() / f"{movie_id}.json"
        if not annotation_path.exists():
            return []

        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        fps = float(annotation.get("fps", 24.0) or 24.0)
        windows: List[Tuple[float, float]] = []
        for scene in annotation.get("scene", []):
            start_sec = scene.get("start_seconds")
            end_sec = scene.get("end_seconds")
            if start_sec is None or end_sec is None:
                frame = scene.get("frame", [])
                if len(frame) >= 2:
                    start_sec = float(frame[0]) / fps
                    end_sec = float(frame[1]) / fps
            if start_sec is None or end_sec is None:
                continue
            start_val = float(start_sec)
            end_val = float(end_sec)
            if end_val <= start_val:
                continue
            windows.append((start_val, end_val))

        return windows

    @staticmethod
    def _pick_semantic_window(
        linear_start: float,
        linear_end: float,
        windows: Sequence[Tuple[float, float]],
    ) -> Optional[Tuple[float, float]]:
        if not windows:
            return None

        overlaps = [
            window
            for window in windows
            if window[1] > linear_start and window[0] < linear_end
        ]
        if overlaps:
            return max(
                overlaps,
                key=lambda window: min(window[1], linear_end)
                - max(window[0], linear_start),
            )

        linear_mid = (linear_start + linear_end) / 2.0
        return min(
            windows,
            key=lambda window: abs(((window[0] + window[1]) / 2.0) - linear_mid),
        )

    @staticmethod
    def _is_substantial_dialogue(text: str) -> bool:
        cleaned = _clean(text)
        return (
            len(cleaned) >= SUBSTANTIAL_DIALOGUE_MIN_CHARS
            and len(cleaned.split()) >= SUBSTANTIAL_DIALOGUE_MIN_WORDS
        )

    def _build_dialogue_candidates(
        self, dialogues: Sequence[Dict], from_end: bool = False
    ) -> List[str]:
        if not dialogues:
            return []

        selected = list(dialogues[-3:] if from_end else dialogues[:3])
        if from_end:
            selected.reverse()

        candidates: List[str] = []
        seen = set()
        for dialogue in selected:
            text = dialogue.get("text", "").strip()
            if not text:
                continue
            candidate_pool = [text, *self._extract_ngrams(text)]
            for candidate in candidate_pool:
                normalized = _clean(candidate)
                if not normalized or normalized in seen:
                    continue
                if not self._is_substantial_dialogue(candidate):
                    continue
                seen.add(normalized)
                candidates.append(candidate)
        return candidates

    @staticmethod
    def _extract_ngrams(text: str) -> List[str]:
        cleaned = _clean(text)
        words = cleaned.split()
        if len(words) <= 10:
            return [cleaned] if cleaned else []

        ngrams = {
            " ".join(words[:10]),
            " ".join(words[-10:]),
        }
        mid = len(words) // 2
        start = max(0, mid - 5)
        end = min(len(words), start + 10)
        ngrams.add(" ".join(words[start:end]))
        return [candidate for candidate in ngrams if candidate]

    def _find_anchor_timestamp(
        self,
        query_candidates: Sequence[str],
        srt_entries: Sequence[Dict],
        search_start: float,
        search_end: float,
        prefer_end: bool = False,
    ) -> Optional[Tuple[float, float]]:
        if not query_candidates or not srt_entries or search_end <= search_start:
            return None

        best_time: Optional[float] = None
        best_score = 0.0
        best_span = (search_start, search_start)

        candidate_entries = [
            entry
            for entry in srt_entries
            if float(entry["end"]) >= search_start and float(entry["start"]) <= search_end
        ]
        if not candidate_entries:
            return None

        for query in query_candidates:
            normalized_query = _clean(query)
            if len(normalized_query) < 6:
                continue

            query_tokens = set(normalized_query.split())
            for idx in range(len(candidate_entries)):
                for width in range(1, MAX_SUBTITLE_WINDOW_ENTRIES + 1):
                    window = candidate_entries[idx : idx + width]
                    if len(window) < width:
                        break

                    window_start = float(window[0]["start"])
                    window_end = float(window[-1]["end"])
                    if window_start < search_start or window_end > search_end:
                        continue

                    window_text = _clean(" ".join(entry["text"] for entry in window))
                    if len(window_text) < 6:
                        continue

                    window_tokens = set(window_text.split())
                    overlap = (
                        len(query_tokens & window_tokens) / max(len(query_tokens), 1)
                    )
                    similarity = _fuzzy_score(normalized_query, window_text)
                    containment = 1.0 if normalized_query in window_text else 0.0
                    length_penalty = min(
                        0.12,
                        abs(len(window_text) - len(normalized_query))
                        / max(len(normalized_query), 1)
                        * 0.12,
                    )
                    score = max(
                        0.0,
                        min(
                            1.0,
                            0.55 * similarity
                            + 0.35 * overlap
                            + 0.10 * containment
                            - length_penalty,
                        ),
                    )

                    if score > best_score:
                        best_score = score
                        best_time = window_end if prefer_end else window_start
                        best_span = (window_start, window_end)
                    elif (
                        abs(score - best_score) <= 0.015
                        and best_time is not None
                        and (
                            (prefer_end and window_end > best_span[1])
                            or (not prefer_end and window_start < best_span[0])
                        )
                    ):
                        best_time = window_end if prefer_end else window_start
                        best_span = (window_start, window_end)

        if best_time is None:
            return None
        return best_time, best_score

    @staticmethod
    def _compute_confidence(
        anchor_quality: str,
        start_score: Optional[float],
        end_score: Optional[float],
        semantic_window: Optional[Tuple[float, float]],
    ) -> float:
        scores = [score for score in (start_score, end_score) if score is not None]
        if anchor_quality == "full" and scores:
            return round(min(0.99, 0.15 + (sum(scores) / len(scores)) * 0.85), 3)
        if anchor_quality == "partial" and scores:
            return round(min(0.92, 0.10 + scores[0] * 0.80), 3)
        if semantic_window and scores:
            return round(min(0.55, 0.16 + max(scores) * 0.45), 3)
        if semantic_window:
            return 0.30
        if scores:
            return round(min(0.35, 0.06 + max(scores) * 0.35), 3)
        return 0.10
