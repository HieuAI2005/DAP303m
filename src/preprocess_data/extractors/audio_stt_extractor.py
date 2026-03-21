"""
STT Subtitle Generator — Speech-to-Text → SRT

Generates SRT subtitle files from video audio using:
  1. Groq Whisper API (whisper-large-v3-turbo) — primary, fast cloud-based
  2. Local Whisper (openai-whisper) — fallback

For new videos without pre-existing subtitle files.
"""

import os
import re
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

from preprocess_data.config import PreprocessConfig as Cfg

logger = logging.getLogger(__name__)


class STTGenerator:
    """Generate SRT subtitles from video audio using Speech-to-Text."""

    def __init__(self, language: str = "en"):
        """
        Args:
            language: Language code ('en', 'vi', etc.)
        """
        self.language = language

    def generate(self, movie_id: str, video_path: Path = None) -> Optional[Path]:
        """
        Generate SRT subtitle file from video.

        Returns: Path to generated .srt file, or None on failure
        """
        if video_path is None:
            video_path = Cfg.get_video_path(movie_id)
        if video_path is None or not video_path.exists():
            logger.error(f"  ❌ No video for {movie_id}")
            return None

        logger.info(f"  🎤 STT Generation: {movie_id}")

        # Check if SRT already exists
        Cfg.get_subtitle_dir().mkdir(parents=True, exist_ok=True)
        srt_path = Cfg.get_subtitle_dir() / f"{movie_id}.srt"
        if srt_path.exists():
            logger.info(f"  ⏩ SRT already exists: {srt_path}")
            return srt_path

        # Step 1: Extract audio
        audio_path = self._extract_audio(video_path)
        if not audio_path:
            logger.error("  ❌ Audio extraction failed")
            return None

        try:
            # Step 2: Run STT (Groq first, then local Whisper fallback)
            segments = self._transcribe(audio_path)

            if not segments:
                logger.warning("  No speech detected")
                return None

            # Step 3: Write SRT
            self._write_srt(segments, srt_path)
            logger.info(f"  ✅ STT: {len(segments)} segments → {srt_path}")
            return srt_path

        finally:
            # Cleanup temp audio
            if audio_path and audio_path.exists() and "temp" in str(audio_path).lower():
                audio_path.unlink(missing_ok=True)

    def _extract_audio(self, video_path: Path) -> Optional[Path]:
        """Extract audio track from video using ffmpeg."""
        # Use m4a for Groq (smaller than wav, supported format)
        audio_path = Path(tempfile.mktemp(suffix=".m4a"))
        try:
            cmd = [
                "ffmpeg",
                "-i",
                str(video_path),
                "-vn",  # No video
                "-acodec",
                "aac",  # AAC codec for m4a
                "-ar",
                "16000",  # 16kHz
                "-ac",
                "1",  # Mono
                "-y",
                "-v",
                "quiet",
                str(audio_path),
            ]
            subprocess.run(cmd, check=True, timeout=600)
            if audio_path.exists() and audio_path.stat().st_size > 0:
                size_mb = audio_path.stat().st_size / 1024 / 1024
                logger.info(f"  Audio extracted: {size_mb:.1f} MB")
                return audio_path
        except Exception as e:
            logger.error(f"  Audio extraction failed: {e}")
        return None

    def _transcribe(self, audio_path: Path) -> List[Dict]:
        """Transcribe audio: Groq API first, local Whisper fallback."""
        # Try Groq Whisper API first
        segments = self._transcribe_groq(audio_path)
        if segments:
            return segments

        # Fallback: local Whisper
        segments = self._transcribe_local_whisper(audio_path)
        if segments:
            return segments

        logger.warning("  No STT backend available")
        return []

    def _transcribe_groq(self, audio_path: Path) -> Optional[List[Dict]]:
        """Use Groq Whisper API (whisper-large-v3-turbo) for transcription."""
        try:
            from groq import Groq

            client = Groq()  # Uses GROQ_API_KEY env var
            logger.info(
                "  Transcribing with Groq Whisper API (whisper-large-v3-turbo)..."
            )

            # Groq has a 25MB file size limit — check and chunk if needed
            file_size = audio_path.stat().st_size / (1024 * 1024)
            if file_size > 25:
                logger.info(
                    f"  Audio is {file_size:.1f} MB > 25 MB, splitting into chunks..."
                )
                return self._transcribe_groq_chunked(audio_path)

            with open(audio_path, "rb") as f:
                transcription = client.audio.transcriptions.create(
                    file=(audio_path.name, f.read()),
                    model="whisper-large-v3-turbo",
                    temperature=0,
                    language=self.language,
                    response_format="verbose_json",
                )

            # Parse verbose_json response → segments
            segments = []
            if hasattr(transcription, "segments") and transcription.segments:
                for seg in transcription.segments:
                    segments.append(
                        {
                            "start": seg.get("start", seg.start)
                            if hasattr(seg, "start")
                            else seg["start"],
                            "end": seg.get("end", seg.end)
                            if hasattr(seg, "end")
                            else seg["end"],
                            "text": (
                                seg.get("text", seg.text)
                                if hasattr(seg, "text")
                                else seg["text"]
                            ).strip(),
                        }
                    )
            elif hasattr(transcription, "text") and transcription.text:
                # Fallback: if no segments, create one big segment
                segments.append(
                    {
                        "start": 0.0,
                        "end": 0.0,
                        "text": transcription.text.strip(),
                    }
                )

            logger.info(f"  Groq STT: {len(segments)} segments transcribed")
            return segments

        except ImportError:
            logger.info("  groq package not installed (pip install groq)")
            return None
        except Exception as e:
            logger.warning(f"  Groq STT failed: {e}")
            return None

    def _transcribe_groq_chunked(self, audio_path: Path) -> List[Dict]:
        """Split large audio into chunks and transcribe each with Groq."""
        from groq import Groq

        client = Groq()
        chunk_duration = 600  # 10 minutes per chunk
        segments = []

        # Get audio duration
        try:
            import json as _json

            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_format",
                str(audio_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = _json.loads(result.stdout)
            total_duration = float(data["format"]["duration"])
        except Exception:
            total_duration = 7200  # Default 2h

        offset = 0.0
        chunk_idx = 0
        while offset < total_duration:
            chunk_path = Path(tempfile.mktemp(suffix=f"_chunk{chunk_idx}.m4a"))
            try:
                # Extract chunk
                cmd = [
                    "ffmpeg",
                    "-ss",
                    str(offset),
                    "-i",
                    str(audio_path),
                    "-t",
                    str(chunk_duration),
                    "-acodec",
                    "aac",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-y",
                    "-v",
                    "quiet",
                    str(chunk_path),
                ]
                subprocess.run(cmd, check=True, timeout=120)

                if chunk_path.exists() and chunk_path.stat().st_size > 0:
                    with open(chunk_path, "rb") as f:
                        transcription = client.audio.transcriptions.create(
                            file=(chunk_path.name, f.read()),
                            model="whisper-large-v3-turbo",
                            temperature=0,
                            language=self.language,
                            response_format="verbose_json",
                        )

                    if hasattr(transcription, "segments") and transcription.segments:
                        for seg in transcription.segments:
                            start = (
                                seg.get("start", seg.start)
                                if hasattr(seg, "start")
                                else seg["start"]
                            )
                            end = (
                                seg.get("end", seg.end)
                                if hasattr(seg, "end")
                                else seg["end"]
                            )
                            text = (
                                seg.get("text", seg.text)
                                if hasattr(seg, "text")
                                else seg["text"]
                            )
                            segments.append(
                                {
                                    "start": start + offset,
                                    "end": end + offset,
                                    "text": text.strip(),
                                }
                            )

                    logger.info(
                        f"  Chunk {chunk_idx}: transcribed ({offset:.0f}s - {min(offset + chunk_duration, total_duration):.0f}s)"
                    )

            except Exception as e:
                logger.warning(f"  Chunk {chunk_idx} failed: {e}")
            finally:
                chunk_path.unlink(missing_ok=True)

            offset += chunk_duration
            chunk_idx += 1

        return segments

    def _transcribe_local_whisper(self, audio_path: Path) -> Optional[List[Dict]]:
        """Fallback: use local OpenAI Whisper model."""
        try:
            import whisper

            logger.info("  Loading local Whisper model (base)...")
            model = whisper.load_model("base")
            result = model.transcribe(
                str(audio_path),
                language=self.language,
                verbose=False,
            )

            segments = []
            for seg in result.get("segments", []):
                segments.append(
                    {
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"].strip(),
                    }
                )
            return segments

        except ImportError:
            logger.info("  whisper package not installed (pip install openai-whisper)")
            return None
        except Exception as e:
            logger.error(f"  Local Whisper failed: {e}")
            return None

    @staticmethod
    def _write_srt(segments: List[Dict], output_path: Path):
        """Write segments as SRT subtitle file."""
        lines = []
        for i, seg in enumerate(segments, 1):
            start_ts = _sec_to_srt(seg["start"])
            end_ts = _sec_to_srt(seg["end"])
            text = seg["text"]
            if text:
                lines.append(str(i))
                lines.append(f"{start_ts} --> {end_ts}")
                lines.append(text)
                lines.append("")

        output_path.write_text("\n".join(lines), encoding="utf-8")


def _sec_to_srt(seconds: float) -> str:
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
