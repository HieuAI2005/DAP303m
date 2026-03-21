"""
Utilities for turning screenplay scenes into reusable retrieval evidence.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List


ACTION_CUE_PHRASES = (
    "beat",
    "cut to",
    "camera",
    "close on",
    "angle on",
    "we see",
    "silence",
    "beat.",
)
ACTION_CUE_VERBS = (
    "arrives",
    "calls",
    "comes",
    "begins",
    "enters",
    "exits",
    "falls",
    "fills",
    "finds",
    "glances",
    "hands",
    "holds",
    "hurries",
    "ignores",
    "kneels",
    "leans",
    "looks",
    "moves",
    "nods",
    "opens",
    "points",
    "pulls",
    "pushes",
    "reads",
    "removes",
    "rests",
    "runs",
    "screeches",
    "searches",
    "settles",
    "shuffles",
    "sits",
    "slides",
    "surveys",
    "stands",
    "starts",
    "stares",
    "stops",
    "takes",
    "tickles",
    "turns",
    "walks",
    "watches",
    "whispers",
    "wipes",
)
FIRST_PERSON_TOKENS = {
    "i",
    "i'm",
    "ive",
    "i've",
    "me",
    "my",
    "mine",
    "we",
    "we're",
    "our",
    "ours",
    "you",
    "you're",
    "your",
    "yours",
}


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _read_field(scene_or_dict: Any, key: str, default=None):
    if isinstance(scene_or_dict, dict):
        return scene_or_dict.get(key, default)
    return getattr(scene_or_dict, key, default)


def _split_fragments(text: str) -> List[str]:
    normalized = clean_text(text)
    normalized = re.sub(r"\b(Mr|Mrs|Ms|Dr)\.\s+", lambda m: f"{m.group(1)}<DOT> ", normalized)
    fragments = re.split(r"(?<=[\.\!\?])\s+|\n+", normalized)
    fragments = [fragment.replace("<DOT>", ".") for fragment in fragments]
    return [fragment for fragment in fragments if fragment]


def _split_speaker_segments(speaker: str, text: str) -> List[tuple[str, str]]:
    normalized = clean_text(text)
    if not normalized:
        return []

    speaker_pattern = re.compile(
        r"([A-Z][A-Z0-9\.'\-]*(?:\s+[A-Z][A-Z0-9\.'\-]*){0,4}):\s*"
    )
    matches = list(speaker_pattern.finditer(normalized))
    if not matches:
        return [(clean_text(speaker), normalized)]

    segments: List[tuple[str, str]] = []
    first = matches[0]
    if first.start() > 0:
        leading = normalized[: first.start()].strip()
        if leading:
            segments.append((clean_text(speaker), leading))

    for idx, match in enumerate(matches):
        seg_speaker = clean_text(match.group(1))
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(normalized)
        seg_text = normalized[start:end].strip()
        if seg_text:
            segments.append((seg_speaker, seg_text))

    return segments or [(clean_text(speaker), normalized)]


def _looks_like_action_fragment(fragment: str, speaker: str = "") -> bool:
    raw = clean_text(fragment)
    lower = raw.lower()
    if not raw:
        return False

    if any(cue in lower for cue in ACTION_CUE_PHRASES):
        return True
    if re.search(r"\b[A-Z]{3,}\b", fragment):
        return True

    tokens = [token for token in re.findall(r"\w+", lower) if token]
    if not tokens:
        return False

    has_first_person = any(token in FIRST_PERSON_TOKENS for token in tokens)
    action_hits = sum(1 for verb in ACTION_CUE_VERBS if verb in lower)
    starts_with_subject = bool(
        re.match(
            r"^(He|She|They|His|Her|Their|Mr\.|Mrs\.|Ms\.|Dr\.|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b",
            raw,
        )
    )
    stage_subject_verb = bool(
        re.match(
            r"^(He|She|They|His|Her|Their|Mr\.|Mrs\.|Ms\.|Dr\.|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\s+[a-z][a-z'\-]{2,}\b",
            raw,
        )
    )
    speaker_norm = clean_text(speaker).lower()
    if speaker_norm and raw.lower().startswith(f"{speaker_norm}:"):
        starts_with_subject = False
    if speaker_norm and raw.lower().startswith(f"{speaker_norm} "):
        starts_with_subject = False

    return bool(
        action_hits
        and starts_with_subject
        and not has_first_person
        and len(tokens) >= 4
        or (stage_subject_verb and not has_first_person and len(tokens) >= 4)
    )


def _sanitize_dialogue_text(text: str, speaker: str = "") -> tuple[str, str]:
    kept: List[str] = []
    overflow: List[str] = []
    in_action_tail = False

    for fragment in _split_fragments(text):
        if _looks_like_action_fragment(fragment, speaker=speaker):
            in_action_tail = True
            overflow.append(fragment)
            continue
        if in_action_tail:
            overflow.append(fragment)
            continue
        kept.append(fragment)

    if not kept and overflow:
        return "", " ".join(overflow).strip()
    return " ".join(kept).strip(), " ".join(overflow).strip()


def build_action_excerpt(scene_or_dict: Any, max_chars: int = 720) -> str:
    parts: List[str] = []
    total = 0
    for line in _read_field(scene_or_dict, "action_lines", []) or []:
        cleaned = clean_text(line)
        if not cleaned:
            continue
        if total + len(cleaned) + 1 > max_chars:
            break
        parts.append(cleaned)
        total += len(cleaned) + 1

    for turn in _read_field(scene_or_dict, "dialogue_lines", []) or []:
        if isinstance(turn, dict):
            segments = _split_speaker_segments(
                turn.get("char", ""), turn.get("text", "")
            )
        else:
            segments = _split_speaker_segments("", turn)
        for segment_speaker, segment_text in segments:
            _, overflow = _sanitize_dialogue_text(segment_text, speaker=segment_speaker)
            cleaned = clean_text(overflow)
            if not cleaned:
                continue
            if total + len(cleaned) + 1 > max_chars:
                return " ".join(parts)
            parts.append(cleaned)
            total += len(cleaned) + 1
    return " ".join(parts)


def build_dialogue_turns(
    scene_or_dict: Any,
    max_turns: int = 12,
    max_chars: int = 980,
) -> List[str]:
    turns: List[str] = []
    total = 0
    for turn in _read_field(scene_or_dict, "dialogue_lines", []) or []:
        if isinstance(turn, dict):
            segments = _split_speaker_segments(
                turn.get("char", ""), turn.get("text", "")
            )
        else:
            segments = _split_speaker_segments("", turn)
        for speaker, segment_text in segments:
            speaker = clean_text(speaker)
            text, _ = _sanitize_dialogue_text(segment_text, speaker=speaker)
            if not text:
                continue
            formatted = f"{speaker}: {text}" if speaker else text
            if total + len(formatted) + 1 > max_chars:
                return turns
            turns.append(formatted)
            total += len(formatted) + 1
            if len(turns) >= max_turns:
                return turns
    return turns


def build_screenplay_dialogue_excerpt(
    dialogue_turns: Iterable[str], max_chars: int = 980
) -> str:
    excerpt_parts: List[str] = []
    total = 0
    for turn in dialogue_turns:
        cleaned = clean_text(turn)
        if not cleaned:
            continue
        if total + len(cleaned) + 1 > max_chars:
            break
        excerpt_parts.append(cleaned)
        total += len(cleaned) + 1
    return " ".join(excerpt_parts)


def build_screenplay_evidence(
    heading: str, action_excerpt: str, dialogue_turns: Iterable[str]
) -> str:
    parts = [f"Heading: {clean_text(heading)}"]
    if action_excerpt:
        parts.append(f"Action: {action_excerpt}")
    dialogue_turns = [clean_text(turn) for turn in dialogue_turns if clean_text(turn)]
    if dialogue_turns:
        parts.append(f"Dialogue turns: {' | '.join(dialogue_turns)}")
    return "\n".join(part for part in parts if part.strip())


def build_screenplay_context_excerpt(
    scene_or_dict: Any,
    max_blocks: int = 12,
    max_chars: int = 1600,
) -> str:
    parts: List[str] = []
    total = 0

    blocks = _read_field(scene_or_dict, "blocks", []) or []
    if not blocks:
        synthesized: List[str] = []
        action_excerpt = build_action_excerpt(scene_or_dict, max_chars=max_chars // 2)
        if action_excerpt:
            synthesized.append(action_excerpt)
        dialogue_excerpt = build_screenplay_dialogue_excerpt(
            build_dialogue_turns(scene_or_dict, max_turns=max_blocks, max_chars=max_chars)
        )
        if dialogue_excerpt:
            synthesized.append(dialogue_excerpt)
        return " ".join(synthesized).strip()

    for block in blocks:
        block_type = str(block.get("type", "") or "")
        if block_type == "SceneHeading":
            continue

        if block_type == "Dialogue":
            segments = _split_speaker_segments(
                block.get("character", ""), block.get("text", "")
            )
            rendered_parts: List[str] = []
            overflow_parts: List[str] = []
            for speaker, segment_text in segments:
                speaker = clean_text(speaker)
                text, overflow = _sanitize_dialogue_text(
                    segment_text, speaker=speaker
                )
                if text:
                    rendered_parts.append(f"{speaker}: {text}" if speaker else text)
                if overflow:
                    overflow_parts.append(clean_text(overflow))
            rendered = " ".join(part for part in rendered_parts if part)
            action_tail = " ".join(part for part in overflow_parts if part)
        else:
            rendered = clean_text(block.get("text", ""))
            action_tail = ""

        if not rendered:
            rendered = ""
        for candidate in (rendered, action_tail):
            if not candidate:
                continue
            if total + len(candidate) + 1 > max_chars:
                return " ".join(parts).strip()
            parts.append(candidate)
            total += len(candidate) + 1
            if len(parts) >= max_blocks:
                return " ".join(parts).strip()

    return " ".join(parts).strip()


def build_screenplay_payload(scene_or_dict: Any) -> Dict[str, Any]:
    heading = clean_text(_read_field(scene_or_dict, "heading", ""))
    action_excerpt = build_action_excerpt(scene_or_dict)
    dialogue_turns = build_dialogue_turns(scene_or_dict)
    context_excerpt = build_screenplay_context_excerpt(scene_or_dict)
    confidence = _read_field(scene_or_dict, "confidence_score", 0.0) or _read_field(
        scene_or_dict, "alignment_confidence", 0.0
    )

    return {
        "alignment_confidence": round(float(confidence or 0.0), 3),
        "screenplay_action_excerpt": action_excerpt,
        "screenplay_dialogue_turns": dialogue_turns,
        "screenplay_dialogue_excerpt": build_screenplay_dialogue_excerpt(
            dialogue_turns
        ),
        "screenplay_context_excerpt": context_excerpt,
        "screenplay_evidence": build_screenplay_evidence(
            heading, action_excerpt, dialogue_turns
        ),
    }
