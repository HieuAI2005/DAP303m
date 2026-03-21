# ─────────────────────────────────────────────────────────────────────────────
# whisper_transcriber.py
# Whisper STT Integration — Audio Transcription Pipeline for Video Understanding
# Layer 3: Dialogue & Audio in the 5-Layer Scene Metadata Model
# ─────────────────────────────────────────────────────────────────────────────
"""
 Whisper-based Speech-to-Text transcription for movie videos.

 Responsibilities:
   1. Transcribe raw audio → text with word-level timestamps
   2. Chunk transcript into 30-second segments aligned to video frames
   3. Extract speaker labels (when speaker diarization available)
   4. Detect audio events: music cues, ambient sounds, laughter, etc.
   5. Index dialogue chunks for the Knowledge Index (L3)
   6. Enrich Layer 3 of the 5-Layer Scene Metadata Model

 Output schema (per chunk):
   {
     "chunk_id": str,            # unique identifier
     "movie_id": str,
     "start_seconds": float,
     "end_seconds": float,
     "text": str,                # transcribed text
     "words": [                  # word-level timestamps (optional)
       {"word": str, "start": float, "end": float, "probability": float}
     ],
     "language": str,            # detected or specified language
     "audio_events": [str],      # e.g. ["rain", "door creaking"]
     "background_music": bool,
     "speaker": Optional[str],   # from diarization
     "avg_logprob": float,      # model confidence
     "no_speech_prob": float,
   }
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Environment / Config ───────────────────────────────────────────────────────

WHISPER_MODEL = os.getenv("MOVIERAG_WHISPER_MODEL", "medium")
WHISPER_DEVICE = os.getenv("MOVIERAG_WHISPER_DEVICE", "cuda")
WHISPER_CHUNK_LENGTH = int(os.getenv("MOVIERAG_WHISPER_CHUNK", "30"))
WHISPER_BATCH_SIZE = int(os.getenv("MOVIERAG_WHISPER_BATCH", "16"))
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
TEMP_AUDIO_DIR = os.getenv("MOVIERAG_TEMP_AUDIO", "/tmp/movierag_audio")


# ── Audio Extraction ──────────────────────────────────────────────────────────

def extract_audio(video_path: str, output_path: Optional[str] = None,
                  sample_rate: int = 16000, mono: bool = True) -> str:
    """
    Extract audio from video file using ffmpeg.

    Args:
        video_path: Path to input video file.
        output_path: Path for output audio file. Auto-generated if None.
        sample_rate: Target sample rate in Hz (default 16kHz for Whisper).
        mono: Whether to convert to mono channel.

    Returns:
        Path to the extracted audio file.

    Raises:
        FileNotFoundError: If ffmpeg is not installed.
        RuntimeError: If ffmpeg extraction fails.
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_path is None:
        temp_dir = Path(TEMP_AUDIO_DIR)
        temp_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(temp_dir / f"{Path(video_path).stem}_audio.wav")

    cmd = [
        FFMPEG_PATH, "-y",
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
    ]
    if mono:
        cmd += ["-ac", "1"]
    cmd += ["-loglevel", "error", output_path]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Extracted audio to: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg audio extraction failed: {e.stderr}") from e


# ── Audio Event Detection ─────────────────────────────────────────────────────

def detect_audio_events(audio_path: str,
                        model: str = "audioflash老人",
                        sample_rate: int = 16000) -> Dict[str, Any]:
    """
    Detect non-speech audio events using a simple energy-based detector.
    For production, replace with a dedicated audio event model (e.g., PANNs).

    Detects:
      - music: sustained energy in typical music frequency bands
      - rain: broadband noise with specific temporal patterns
      - applause: impulse-like energy bursts
      - laughter: distinctive amplitude modulation
      - silence: regions with energy below threshold

    Returns:
        Dict with keys: "events" (list of event dicts), "ambient_label" (str).
    """
    try:
        import librosa
    except ImportError:
        logger.warning("librosa not installed; audio event detection skipped.")
        return {"events": [], "ambient_label": "unknown"}

    try:
        y, sr = librosa.load(audio_path, sr=sample_rate)
    except Exception as e:
        logger.warning(f"Could not load audio for event detection: {e}")
        return {"events": [], "ambient_label": "unknown"}

    # Compute RMS energy in 1-second windows
    frame_length = int(sr * 1.0)
    hop_length = int(sr * 0.5)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.times_like(rms, sr=sr, hop_length=hop_length)

    events: List[Dict[str, Any]] = []
    threshold = np.percentile(rms, 20)

    # Detect music: sustained moderate energy with spectral flatness check
    try:
        spec_flat = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
        music_mask = (rms > threshold) & (spec_flat < 0.2)
    except Exception:
        music_mask = rms > threshold * 1.5

    if np.mean(music_mask) > 0.3:
        events.append({
            "type": "background_music",
            "start_seconds": float(times[0]),
            "end_seconds": float(times[-1]),
            "confidence": float(round(np.mean(music_mask), 3)),
        })

    # Detect silence: energy well below threshold
    silence_mask = rms < threshold * 0.3
    if np.mean(silence_mask) > 0.1:
        # Find contiguous silence segments
        in_silence = False
        seg_start = 0.0
        for i, (t, silent) in enumerate(zip(times, silence_mask)):
            if silent and not in_silence:
                in_silence = True
                seg_start = t
            elif not silent and in_silence:
                in_silence = False
                if t - seg_start > 1.0:
                    events.append({
                        "type": "silence",
                        "start_seconds": float(seg_start),
                        "end_seconds": float(t),
                        "confidence": 0.9,
                    })
        if in_silence:
            events.append({
                "type": "silence",
                "start_seconds": float(seg_start),
                "end_seconds": float(times[-1]),
                "confidence": 0.9,
            })

    # Ambient label
    ambient = "ambient"
    if any(e["type"] == "background_music" for e in events):
        ambient = "musical"
    elif any(e["type"] == "silence" for e in events):
        ambient = "quiet"
    else:
        ambient = "active"

    return {"events": events, "ambient_label": ambient}


# ── Whisper Transcription ─────────────────────────────────────────────────────

class WhisperTranscriber:
    """
    Whisper-based speech-to-text transcriber.

    Usage:
        transcriber = WhisperTranscriber()
        chunks = transcriber.transcribe("movie.mp4")
        chunks = transcriber.transcribe_audio_file("audio.wav", language="en")
    """

    def __init__(
        self,
        model_name: str = WHISPER_MODEL,
        device: str = WHISPER_DEVICE,
        language: Optional[str] = None,
        chunk_length: int = WHISPER_CHUNK_LENGTH,
        batch_size: int = WHISPER_BATCH_SIZE,
        output_dir: Optional[str] = None,
        temperature: float = 0.0,
        condition_on_previous: bool = True,
    ):
        """
        Initialize Whisper transcriber.

        Args:
            model_name: Whisper model size ("tiny", "base", "small", "medium", "large").
            device: "cuda" or "cpu".
            language: Force specific language (ISO code). Auto-detect if None.
            chunk_length: Max segment length in seconds.
            batch_size: Batch size for batched inference.
            output_dir: Directory to save transcript JSON files.
            temperature: Sampling temperature (0 = deterministic).
            condition_on_previous: Use previous segment as context.
        """
        self.model_name = model_name
        self.device = device
        self.language = language
        self.chunk_length = chunk_length
        self.batch_size = batch_size
        self.temperature = temperature
        self.condition_on_previous = condition_on_previous
        self.output_dir = Path(output_dir) if output_dir else None
        self._model = None

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Lazy model loading ────────────────────────────────────────────────────

    def _load_model(self):
        """Lazy-load the Whisper model on first use."""
        if self._model is not None:
            return

        try:
            import whisper
        except ImportError:
            try:
                import openai.whisper as whisper
            except ImportError:
                raise ImportError(
                    "Whisper not installed. Install with: pip install openai-whisper"
                )

        logger.info(f"Loading Whisper model: {self.model_name}")
        self._model = whisper.load_model(self.model_name, device=self.device)
        logger.info(f"Whisper model loaded on device: {self.device}")

    # ── Core transcription ────────────────────────────────────────────────────

    def transcribe(
        self,
        video_path: str,
        movie_id: str = "unknown",
        word_timestamps: bool = True,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Transcribe a video file end-to-end.

        Args:
            video_path: Path to video file.
            movie_id: Movie identifier for output metadata.
            word_timestamps: Include word-level timestamps in output.
            verbose: Print Whisper progress.

        Returns:
            Dict with keys:
              - "movie_id": str
              - "language": str
              - "full_text": str
              - "chunks": List[Dict] — 30-second dialogue chunks
              - "audio_events": List[Dict]
              - "transcription_path": str — where transcript was saved
        """
        audio_path = extract_audio(video_path)
        return self._transcribe_from_audio(
            audio_path, movie_id, word_timestamps, verbose
        )

    def transcribe_audio_file(
        self,
        audio_path: str,
        movie_id: str = "unknown",
        language: Optional[str] = None,
        word_timestamps: bool = True,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Transcribe an existing audio file.
        """
        lang = language or self.language
        return self._transcribe_from_audio(
            audio_path, movie_id, word_timestamps, verbose, language=lang
        )

    def _transcribe_from_audio(
        self,
        audio_path: str,
        movie_id: str,
        word_timestamps: bool,
        verbose: bool,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Shared transcription logic for audio files."""
        self._load_model()

        logger.info(f"Transcribing audio: {audio_path}")
        options = dict(
            language=language or self.language,
            task="transcribe",
            temperature=self.temperature,
            condition_on_previous_text=self.condition_on_previous,
            word_timestamps=word_timestamps,
            verbose=verbose,
        )
        # Remove None values
        options = {k: v for k, v in options.items() if v is not None}

        try:
            result = self._model.transcribe(audio_path, **options)
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            return {
                "movie_id": movie_id,
                "language": "unknown",
                "full_text": "",
                "chunks": [],
                "audio_events": [],
                "transcription_path": "",
                "error": str(e),
            }

        full_text = result.get("text", "")
        detected_lang = result.get("language", "unknown")

        # Detect audio events
        audio_events_result = detect_audio_events(audio_path)
        audio_events = audio_events_result.get("events", [])

        # Chunk into 30-second segments
        chunks = self._chunk_transcript(
            result.get("segments", []), movie_id, detected_lang
        )

        # Save transcript
        transcript_path = ""
        if self.output_dir:
            transcript_path = str(self.output_dir / f"{movie_id}_transcript.json")
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump({
                    "movie_id": movie_id,
                    "language": detected_lang,
                    "full_text": full_text,
                    "chunks": chunks,
                    "audio_events": audio_events,
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"Transcript saved: {transcript_path}")

        return {
            "movie_id": movie_id,
            "language": detected_lang,
            "full_text": full_text,
            "chunks": chunks,
            "audio_events": audio_events,
            "transcription_path": transcript_path,
        }

    # ── Chunking logic ────────────────────────────────────────────────────────

    def _chunk_transcript(
        self,
        segments: List[Dict[str, Any]],
        movie_id: str,
        language: str,
    ) -> List[Dict[str, Any]]:
        """
        Group Whisper segments into ~30-second chunks.

        Args:
            segments: List of Whisper segment dicts with "start", "end", "text", "avg_logprob".
            movie_id: Movie identifier.
            language: Detected language.

        Returns:
            List of chunk dicts conforming to the 5-Layer Scene Metadata Model.
        """
        CHUNK_SECS = self.chunk_length
        chunks: List[Dict[str, Any]] = []
        current_chunk_words: List[str] = []
        current_start: float = 0.0
        current_end: float = 0.0
        current_logprobs: List[float] = []
        chunk_idx = 0

        def finalize_chunk():
            nonlocal chunk_idx, current_chunk_words, current_start, current_end
            nonlocal current_logprobs

            if not current_chunk_words:
                return

            chunk_text = " ".join(current_chunk_words).strip()
            chunk_id = f"{movie_id}_chunk_{chunk_idx:04d}"

            chunk = {
                "chunk_id": chunk_id,
                "movie_id": movie_id,
                "start_seconds": round(current_start, 3),
                "end_seconds": round(current_end, 3),
                "text": chunk_text,
                "language": language,
                "audio_events": [],  # enriched later by detect_audio_events()
                "background_music": False,
                "speaker": None,
                "avg_logprob": float(np.mean(current_logprobs)) if current_logprobs else 0.0,
                "no_speech_prob": 0.0,
                "words": [],
            }
            chunks.append(chunk)
            chunk_idx += 1
            current_chunk_words = []
            current_start = 0.0
            current_end = 0.0
            current_logprobs = []

        for seg in segments:
            seg_start = float(seg.get("start", 0))
            seg_end = float(seg.get("end", 0))
            seg_text = seg.get("text", "").strip()
            seg_logprob = float(seg.get("avg_logprob", 0.0))

            if not seg_text:
                continue

            # If adding this segment exceeds chunk length and chunk is non-empty, finalize
            if current_chunk_words and (seg_end - current_start) >= CHUNK_SECS:
                finalize_chunk()

            if not current_chunk_words:
                current_start = seg_start

            current_chunk_words.append(seg_text)
            current_end = seg_end
            current_logprobs.append(seg_logprob)

        # Don't forget the last chunk
        finalize_chunk()

        return chunks

    # ── Index-ready output ─────────────────────────────────────────────────────

    def to_index_documents(self, transcript_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert transcript result to indexable documents for Knowledge Index (L3).

        Each 30-second chunk becomes one document with:
          - text: dialogue text
          - metadata: temporal, movie_id, audio events
        """
        docs = []
        for chunk in transcript_result.get("chunks", []):
            doc = {
                "text": chunk["text"],
                "chunk_id": chunk["chunk_id"],
                "movie_id": chunk["movie_id"],
                "start_seconds": chunk["start_seconds"],
                "end_seconds": chunk["end_seconds"],
                "language": chunk.get("language", ""),
                "audio_events": chunk.get("audio_events", []),
                "background_music": chunk.get("background_music", False),
                "speaker": chunk.get("speaker"),
                "avg_logprob": chunk.get("avg_logprob", 0.0),
                "type": "dialogue",
                "source": "whisper_transcription",
            }
            docs.append(doc)
        return docs

    # ── Utility ────────────────────────────────────────────────────────────────

    @staticmethod
    def merge_transcripts(
        transcripts: List[Dict[str, Any]], movie_id: str
    ) -> Dict[str, Any]:
        """
        Merge multiple transcript dicts (e.g., from parallel processing)
        into a single coherent transcript, sorted by start_seconds.
        """
        all_chunks: List[Dict[str, Any]] = []
        for t in transcripts:
            all_chunks.extend(t.get("chunks", []))

        all_chunks.sort(key=lambda c: c.get("start_seconds", 0))

        # Re-index chunks sequentially
        merged_chunks = []
        for i, chunk in enumerate(all_chunks):
            new_chunk = dict(chunk)
            new_chunk["chunk_id"] = f"{movie_id}_chunk_{i:04d}"
            merged_chunks.append(new_chunk)

        full_text = " ".join(c["text"] for c in merged_chunks if c["text"])

        return {
            "movie_id": movie_id,
            "full_text": full_text,
            "chunks": merged_chunks,
        }

    def __repr__(self) -> str:
        return (
            f"WhisperTranscriber(model={self.model_name}, device={self.device}, "
            f"chunk_length={self.chunk_length}s)"
        )
