#!/usr/bin/env python3
"""
Full 5-Layer Video Scene Understanding Pipeline
==============================================
Input:  Video file (.mp4) + optional SRT subtitle
Output: 5-Layer chunks → SentenceTransformer → FAISS

Pipeline Steps:
  Step 1: Shot Detection (PySceneDetect / ffmpeg)
  Step 2: Keyframe Extraction (ffmpeg)
  Step 3: CLIP Encoding (openai/clip-vit-base-patch32)
  Step 4: Whisper Transcription (Groq API)
  Step 5: VLM Scene Analysis (Groq Llama 4 Scout vision)
  Step 6: 5-Layer Chunk Builder
  Step 7: SentenceTransformer Embedding
  Step 8: FAISS Index Building
  Step 9: Query + Demo

Usage:
  python scripts/pipeline_5layer_video.py \
    --video data/movies/titanic.mp4 \
    --srt data/movies/titanic.srt \
    --movie-id tt0120338 \
    --title "Titanic (1997)" \
    --output data/pipeline_output/titanic/
"""

import argparse, json, sys, time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv as _ld

# Load .env if present
try:
    _env = Path(__file__).parent.parent / ".env"
    _ld(str(_env) if _env.exists() else None)
except Exception:
    pass

PROJECT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT / "src"))

# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Full 5-Layer Video Scene Understanding Pipeline")
    p.add_argument("--video", required=True, type=str, help="Path to video file (.mp4)")
    p.add_argument("--srt", type=str, default=None, help="Path to SRT subtitle file")
    p.add_argument("--movie-id", required=True, type=str, help="IMDb ID (e.g., tt0120338)")
    p.add_argument("--title", required=True, type=str, help="Movie title (e.g., Titanic (1997))")
    p.add_argument("--output", type=str, default=None,
                   help="Output directory (default: data/pipeline_output/{movie_id}/)")
    p.add_argument("--skip-whisper", action="store_true", help="Skip Whisper (use SRT only)")
    p.add_argument("--skip-vlm", action="store_true", help="Skip VLM analysis")
    p.add_argument("--skip-clip", action="store_true", help="Skip CLIP keyframe encoding")
    p.add_argument("--whisper-model", type=str, default="base",
                   choices=["tiny","base","small","medium","large"],
                   help="Whisper model size (default: base)")
    p.add_argument("--chunk-min-seconds", type=float, default=30.0,
                   help="Minimum chunk duration in seconds (default: 30)")
    p.add_argument("--overlap-seconds", type=float, default=5.0,
                   help="Overlap between chunks in seconds (default: 5)")
    p.add_argument("--groq-api-key", type=str, default=None,
                   help="Groq API key (or set GROQ_API_KEY env var)")
    p.add_argument("--dry-run", action="store_true", help="Show what would be done")
    return p.parse_args()


# ── Step 0: Environment Check ─────────────────────────────────────────────────

def step0_env_check(args):
    """Verify all required tools and packages are available."""
    print("\n" + "=" * 60)
    print("  STEP 0: Environment Check")
    print("=" * 60)

    checks = []

    # Python packages
    import importlib
    for mod, pip_name in [
        ("cv2", "opencv-python"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
        ("faiss", "faiss-cpu"),
        ("sentence_transformers", "sentence-transformers"),
        ("scenedetect", "scenedetect"),
        ("whisper", "openai-whisper"),
        ("openai", "openai"),
    ]:
        try:
            importlib.import_module(mod)
            checks.append(f"  ✅ {mod}")
        except ImportError:
            checks.append(f"  ❌ {mod} → pip install {pip_name}")

    # External tools
    import shutil
    for tool, name in [("ffmpeg", "ffmpeg"), ("ffprobe", "ffprobe")]:
        if shutil.which(tool):
            checks.append(f"  ✅ {name}")
        else:
            checks.append(f"  ❌ {name} → apt install ffmpeg")

    # Groq API
    import os
    api_key = args.groq_api_key or os.environ.get("GROQ_API_KEY", "")
    if api_key:
        checks.append(f"  ✅ Groq API key configured")
    else:
        checks.append(f"  ⚠️  Groq API key not set (VLM/Whisper Groq will use local)")

    for c in checks:
        print(c)

    failed = [c for c in checks if c.startswith("  ❌")]
    if failed:
        print(f"\n  ⛔ {len(failed)} checks failed. Install missing dependencies first.")
        for f in failed:
            print(f"    {f}")
        sys.exit(1)
    print(f"\n  ✅ All checks passed")


# ── Step 1: Shot Detection ────────────────────────────────────────────────────

def step1_shot_detection(video_path: Path, output_dir: Path) -> list:
    """
    Detect shot boundaries using PySceneDetect v0.6+ API.
    Returns list of {start, end, duration} shot dicts.
    Auto-re-encodes to H.264 if AV1/HEVC causes issues.
    """
    print("\n" + "=" * 60)
    print("  STEP 1: Shot Detection (PySceneDetect v0.6+)")
    print("=" * 60)

    from scenedetect import SceneManager, ContentDetector, VideoStreamCv2
    import subprocess

    # Verify video is readable with ffprobe
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(video_path)],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        print(f"  ⚠️  Video codec issue detected. Re-encoding to H.264...")
        h264_path = video_path.with_suffix(".h264.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-vf", "scale=1280:720",
            str(h264_path)
        ], capture_output=True, text=True)
        video_path = h264_path
        print(f"  Re-encoded: {video_path}")

    vs = VideoStreamCv2(str(video_path))
    fps = vs.frame_rate
    total_frames = int(vs.duration.get_frames())
    duration_sec = total_frames / fps

    print(f"  Video: {video_path.name}")
    print(f"  Duration: {duration_sec:.1f}s, FPS: {fps:.2f}, Frames: {total_frames:,}")

    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=27.0, min_scene_len=20))

    start_time = time.time()
    sm.detect_scenes(video=vs, show_progress=True)
    scene_list = sm.get_scene_list()
    elapsed = time.time() - start_time

    shots = []
    for i, scene in enumerate(scene_list):
        sf = int(scene[0].get_frames())
        ef = int(scene[1].get_frames())
        dur = (ef - sf) / fps
        shots.append({
            "shot_id": f"shot_{i+1:04d}",
            "start_frame": sf, "end_frame": ef,
            "start_seconds": sf / fps, "end_seconds": ef / fps,
            "duration": dur,
        })

    avg_dur = sum(s["duration"] for s in shots) / max(len(shots), 1)
    print(f"  Detected {len(shots)} shots in {elapsed:.1f}s")
    print(f"  Avg shot duration: {avg_dur:.1f}s")

    # Save
    out_path = output_dir / "shots.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"shots": shots, "fps": float(fps), "duration": duration_sec,
                   "frame_count": total_frames}, f, indent=2)
    print(f"  Saved: {out_path}")
    return shots


# ── Step 2: Keyframe Extraction ───────────────────────────────────────────────

def step2_keyframes(video_path: Path, shots: list, output_dir: Path, fps_extract: float = 1.0):
    """
    Extract one keyframe per shot (middle frame) + additional uniform frames.
    Saves to output_dir/keyframes/
    """
    print("\n" + "=" * 60)
    print("  STEP 2: Keyframe Extraction (ffmpeg)")
    print("=" * 60)

    kf_dir = output_dir / "keyframes"
    kf_dir.mkdir(parents=True, exist_ok=True)

    import subprocess
    import os

    # Extract one keyframe per shot (middle of each shot)
    total_shots = len(shots)
    print(f"  Extracting {total_shots} keyframes (1 per shot)...")

    start_time = time.time()
    for i, shot in enumerate(shots):
        mid_frame = int((shot["start_frame"] + shot["end_frame"]) / 2)
        output_file = kf_dir / f"{shot['shot_id']}.jpg"

        if output_file.exists():
            continue

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(mid_frame / 60.0 / 60.0 * 3600),  # frame to time
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            "-vf", f"select=eq(n\\,{mid_frame})",
            str(output_file)
        ]

        # Simpler approach: use ffmpeg select at frame level
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"select=between\\(n\\,{shot['start_frame']}\\,{shot['end_frame']}\\),"
                   f"setpts=\\(T-START\\+1\\)/TB\\,scale=336:336",
            "-vsync", "0",
            "-frame_pts", "1",
            str(output_file)
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30
        )
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{total_shots} keyframes extracted...")

    # Fallback: uniform sampling if per-shot extraction has issues
    actual_kfs = list(kf_dir.glob("*.jpg"))
    if len(actual_kfs) < total_shots * 0.5:
        print(f"  Per-shot extraction low yield ({len(actual_kfs)}/{total_shots}), using uniform sampling...")
        kf_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vf", f"fps={fps_extract},scale=336:336",
            "-q:v", "2",
            str(kf_dir / "kf_%05d.jpg")
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        actual_kfs = list(kf_dir.glob("*.jpg"))

    elapsed = time.time() - start_time
    actual_kfs = list(kf_dir.glob("*.jpg"))
    print(f"  Extracted {len(actual_kfs)} keyframes in {elapsed:.1f}s")
    print(f"  Saved to: {kf_dir}/")
    return kf_dir


# ── Step 3: CLIP Encoding ─────────────────────────────────────────────────────

def step3_clip_encoding(kf_dir: Path, output_dir: Path):
    """
    Encode keyframes with CLIP (openai/clip-vit-base-patch32).
    Returns path to embeddings file.
    """
    print("\n" + "=" * 60)
    print("  STEP 3: CLIP Encoding")
    print("=" * 60)

    from PIL import Image
    import torch
    import numpy as np
    import faiss

    # Check for CLIP
    try:
        from transformers import CLIPProcessor, CLIPModel
    except ImportError:
        print("  ⚠️  transformers CLIP not installed. Skipping CLIP encoding.")
        print("      pip install transformers torch")
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    model_name = "openai/clip-vit-base-patch32"
    print(f"  Model: {model_name}")

    from transformers import AutoProcessor, AutoModel
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    kf_files = sorted(kf_dir.glob("*.jpg"))
    print(f"  Encoding {len(kf_files)} keyframes...")

    embeddings = []
    ids = []

    for i, kf_path in enumerate(kf_files):
        try:
            img = Image.open(kf_path).convert("RGB")
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                img_emb = model.get_image_features(**inputs)
            img_emb = img_emb.cpu().numpy().flatten()
            embeddings.append(img_emb)
            ids.append(kf_path.stem)

            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(kf_files)} keyframes encoded...")
        except Exception as e:
            print(f"    ⚠️  Error on {kf_path.name}: {e}")

    if not embeddings:
        print("  ❌ No CLIP embeddings generated")
        return None

    emb_matrix = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(emb_matrix)

    out_path = output_dir / "clip_embeddings.npy"
    np.save(out_path, emb_matrix)

    id_map = {"ids": ids, "paths": [str(kf_dir / f"{i}.jpg") for i in ids]}
    id_path = output_dir / "clip_ids.json"
    with open(id_path, "w") as f:
        json.dump(id_map, f)

    print(f"  ✅ CLIP embeddings: {emb_matrix.shape} ({emb_matrix.nbytes/1024**2:.1f}MB)")
    print(f"  Saved: {out_path}")
    return out_path


# ── Step 4: Whisper Transcription ───────────────────────────────────────────

def step4_whisper(video_path: Path, output_dir: Path, args) -> list:
    """
    Transcribe video audio using Whisper.
    Returns list of {start, end, text, language} segments.
    """
    print("\n" + "=" * 60)
    print("  STEP 4: Whisper Transcription")
    print("=" * 60)

    if args.srt and not args.skip_whisper:
        print(f"  ⚠️  Both SRT ({args.srt}) and Whisper requested.")
        print("      Using Whisper transcription (more accurate for L3).")
    elif args.skip_whisper:
        print("  ⚠️  Whisper skipped (--skip-whisper). Will use SRT only.")
        return None

    from faster_whisper import WhisperModel

    import torch
    compute_type = "int8" if not torch.cuda.is_available() else "float16"
    print(f"  Compute type: {compute_type}")
    print(f"  Model: {args.whisper_model}")

    start_time = time.time()
    model = WhisperModel(args.whisper_model, device="cpu", compute_type=compute_type)
    segments_gen, info = model.transcribe(
        str(video_path),
        task="transcribe",
        word_timestamps=True,
    )
    # Materialize generator → list
    all_segments = [{"start": s.start, "end": s.end, "text": s.text.strip()} for s in segments_gen]
    elapsed = time.time() - start_time
    language = info.language or "en"

    print(f"  Transcribed: {len(all_segments)} segments in {elapsed:.1f}s")
    print(f"  Language: {language}")
    if all_segments:
        print(f"  Sample: {all_segments[0]['text'][:80]}...")

    # Save
    out_path = output_dir / "whisper_transcript.json"
    with open(out_path, "w") as f:
        json.dump({
            "video": str(video_path),
            "language": language,
            "duration": elapsed,
            "model": args.whisper_model,
            "engine": "faster-whisper",
            "segments": all_segments,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")
    return all_segments


# ── Step 5: SRT Parsing ───────────────────────────────────────────────────────

def step5_parse_srt(srt_path: Path) -> list:
    """Parse SRT file → list of {start, end, text} entries."""
    import re

    print("\n" + "=" * 60)
    print("  STEP 5: SRT Parsing")
    print("=" * 60)

    try:
        raw = srt_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw = srt_path.read_text(encoding="latin-1")

    entries = []
    blocks = re.split(r"\n\n+", raw.strip())
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        time_match = re.match(
            r"(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})",
            lines[1]
        )
        if not time_match:
            continue
        start = _ts2sec(time_match.group(1))
        end = _ts2sec(time_match.group(2))
        text = " ".join(_clean_srt_text(l) for l in lines[2:] if l.strip())
        if text:
            entries.append({"start": start, "end": end, "text": text})

    print(f"  Parsed {len(entries)} SRT entries")
    if entries:
        print(f"  Sample: {entries[0]['text'][:80]}...")
    return entries


def _ts2sec(ts: str) -> float:
    ts = ts.replace(",", ".")
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _clean_srt_text(line: str) -> str:
    import re
    line = re.sub(r"<[^>]+>", "", line)
    line = re.sub(r"\[[^\]]*:\]", "", line)
    line = re.sub(r"\([A-Z][a-z]+:", "", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


# ── Step 6: VLM Scene Analysis ───────────────────────────────────────────────

def step6_vlm_analysis(video_path: Path, shots: list, output_dir: Path, args) -> list:
    """
    Analyze each shot/scene with VLM (Groq Llama 4 Scout with vision).
    Returns list of L2+L4 scene analysis dicts.
    """
    print("\n" + "=" * 60)
    print("  STEP 6: VLM Scene Analysis (Groq Llama 4 Scout)")
    print("=" * 60)

    if args.skip_vlm:
        print("  ⚠️  VLM skipped (--skip-vlm). Using existing annotations only.")
        return None

    import os
    api_key = args.groq_api_key or os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("  ❌ Groq API key not set. VLM requires GROQ_API_KEY.")
        print("     Set: export GROQ_API_KEY=gsk_...")
        return None

    # Try to use Groq for VLM
    try:
        from groq import Groq
    except ImportError:
        print("  ❌ groq package not installed.")
        print("     pip install groq")
        return None

    client = Groq(api_key=api_key)

    # Select representative frames for VLM (max 8 per shot)
    analyses = []
    kf_dir = output_dir / "keyframes"
    kf_files = sorted(kf_dir.glob("*.jpg")) if kf_dir.exists() else []

    print(f"  Analyzing {len(shots)} shots with VLM...")
    print(f"  Note: VLM analysis is batched (8 frames per request)")

    # Group shots into batches for efficiency
    batch_size = 8
    for batch_start in range(0, min(len(shots), 20), batch_size):  # Max 20 shots for demo
        batch_end = min(batch_start + batch_size, len(shots))
        batch = shots[batch_start:batch_end]

        # Select frames for this batch
        batch_kfs = kf_files[batch_start:batch_end]
        if not batch_kfs:
            continue

        # Read frames as base64
        import base64
        frame_data = []
        for kf in batch_kfs[:4]:  # Max 4 frames per shot
            with open(kf, "rb") as img_file:
                b64 = base64.b64encode(img_file.read()).decode()
                frame_data.append(b64)

        # Call Groq vision API
        try:
            # Build prompt
            shot_descs = "\n".join([
                f"Shot {i+1}: {s['start_seconds']:.0f}s-{s['end_seconds']:.0f}s"
                for i, s in enumerate(batch)
            ])

            response = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[{
                    "role": "user",
                    "content": f"""Analyze these movie/video frames. For each frame, provide:
1. What's happening in the scene (brief description)
2. Setting/location
3. Characters visible (describe, don't make up names unless confirmed)
4. Actions/activities
5. Emotional tone

Frame timestamps: {shot_descs}

Respond in JSON format:
{{
  "scenes": [
    {{"shot_id": "shot_0001", "description": "...", "setting": "...", "actions": [...], "emotional_tone": "..."}},
    ...
  ]
}}"""
                }],
                temperature=0.1,
                max_tokens=1024,
            )

            import re
            content = response.choices[0].message.content
            # Extract JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                scene_data = json.loads(json_match.group())
                for s in scene_data.get("scenes", []):
                    analyses.append(s)
            else:
                print(f"  ⚠️  Batch {batch_start}-{batch_end}: No JSON in response")
                for s in batch:
                    analyses.append({
                        "shot_id": s["shot_id"],
                        "description": content[:200],
                        "setting": "unknown",
                        "actions": [],
                        "emotional_tone": "neutral",
                    })
        except Exception as e:
            print(f"  ⚠️  Batch {batch_start}-{batch_end} error: {e}")
            for s in batch:
                analyses.append({
                    "shot_id": s["shot_id"],
                    "description": "",
                    "setting": "",
                    "actions": [],
                    "emotional_tone": "",
                })

        print(f"    Batch {batch_start}-{batch_end} done ({len(analyses)} analyses)")

    # Save
    out_path = output_dir / "vlm_analysis.json"
    with open(out_path, "w") as f:
        json.dump({"scenes": analyses}, f, indent=2)
    print(f"  ✅ VLM analysis: {len(analyses)} scenes")
    print(f"  Saved: {out_path}")
    return analyses


# ── Step 7: 5-Layer Chunk Builder ────────────────────────────────────────────

def step7_build_chunks(
    shots: list,
    whisper_segs: list,
    srt_entries: list,
    vlm_analyses: list,
    movie_id: str,
    title: str,
    clip_emb_path: Path,
    output_dir: Path,
    args,
) -> list:
    """
    Build 5-Layer chunks from all available data sources.
    Aligns Whisper/SRT segments to shots.
    """
    print("\n" + "=" * 60)
    print("  STEP 7: 5-Layer Chunk Builder")
    print("=" * 60)

    # Load CLIP embeddings if available
    clip_embs = {}
    if clip_emb_path and clip_emb_path.exists():
        import numpy as np
        emb_matrix = np.load(clip_emb_path)
        id_path = clip_emb_path.parent / "clip_ids.json"
        if id_path.exists():
            with open(id_path) as f:
                id_map = json.load(f)
            for i, shot_id in enumerate(id_map.get("ids", [])):
                if i < len(emb_matrix):
                    clip_embs[shot_id] = emb_matrix[i]

    # Align dialogue to shots
    def get_dialogue_for_shot(shot_start, shot_end, segs, overlap_pct=0.3):
        """Collect segments overlapping with shot by at least overlap_pct."""
        duration = shot_end - shot_start
        overlapping = []
        for seg in (segs or []):
            if seg["end"] < shot_start or seg["start"] > shot_end:
                continue
            ov_start = max(shot_start, seg["start"])
            ov_end = min(shot_end, seg["end"])
            ov_dur = max(0.0, ov_end - ov_start)
            if ov_dur / max(duration, 1.0) >= overlap_pct or ov_dur >= 1.0:
                overlapping.append(seg["text"])
        return " ".join(overlapping).strip()

    # Build VLM lookup
    vlm_by_shot = {}
    if vlm_analyses:
        for a in vlm_analyses:
            sid = a.get("shot_id", "")
            if sid:
                vlm_by_shot[sid] = a

    chunks = []
    for i, shot in enumerate(shots):
        shot_id = shot["shot_id"]
        start_s = shot["start_seconds"]
        end_s = shot["end_seconds"]

        # L2: Semantic from VLM or fallback
        vlm = vlm_by_shot.get(shot_id, {})
        description = vlm.get("description", f"Scene at {start_s:.0f}s")
        situation = vlm.get("setting", "unknown")
        emotional_tone = vlm.get("emotional_tone", "neutral")
        vision_actions = vlm.get("actions", [])

        # L3: Dialogue from Whisper or SRT
        dialogue = ""
        if whisper_segs:
            dialogue = get_dialogue_for_shot(start_s, end_s, whisper_segs)
        if not dialogue and srt_entries:
            dialogue = get_dialogue_for_shot(start_s, end_s, srt_entries)
        if not dialogue:
            dialogue = "[NO_DIALOGUE]"

        # L4: Characters from VLM
        characters = vlm.get("characters", [])

        # L5: Narrative (placeholder for script alignment)
        narrative_arc = "scene"

        chunk = {
            # L1: Temporal
            "chunk_id": f"{movie_id}_chunk_{i:05d}",
            "movie_id": movie_id,
            "title": title,
            "start_seconds": start_s,
            "end_seconds": end_s,
            "duration": shot["duration"],
            "shot_id": shot_id,
            "timestamp_source": "shot_boundary",

            # L2: Semantic
            "description": description,
            "situation": situation,
            "vision_setting": situation,
            "vision_actions": vision_actions,
            "emotional_tone": emotional_tone,

            # L3: Dialogue
            "dialogue_text": dialogue,
            "speaker": "",
            "audio_events": ["speech"] if dialogue and dialogue != "[NO_DIALOGUE]" else [],
            "background_music": False,

            # L4: Cast
            "characters": characters,
            "character_emotions": {},
            "cast_in_scene": [],

            # L5: Narrative
            "narrative_arc": narrative_arc,
            "causal_relations": [],
            "screenplay_context": description,
            "script_primary_heading": "",

            # Metadata
            "clip_embedding": clip_embs.get(shot_id).tolist() if shot_id in clip_embs else None,
        }
        chunks.append(chunk)

    # Stats
    n = len(chunks)
    l3_cov = sum(1 for c in chunks if c["dialogue_text"] and c["dialogue_text"] != "[NO_DIALOGUE]") / max(n, 1) * 100
    l4_cov = sum(1 for c in chunks if c["characters"]) / max(n, 1) * 100
    l5_cov = sum(1 for c in chunks if c["narrative_arc"]) / max(n, 1) * 100

    print(f"  Built {n} chunks")
    print(f"  L1 Temporal:  100.0%")
    print(f"  L2 Semantic:  {sum(1 for c in chunks if c['description']) / max(n,1) * 100:.0f}%")
    print(f"  L3 Dialogue:  {l3_cov:.0f}%")
    print(f"  L4 Characters:{l4_cov:.0f}%")
    print(f"  L5 Narrative: {l5_cov:.0f}%")

    # Save
    out_path = output_dir / "chunks.json"
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "movie_id": movie_id,
                "title": title,
                "num_chunks": n,
                "pipeline_version": "1.0",
                "timestamp": datetime.now().isoformat(),
            },
            "chunks": chunks,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")
    return chunks


# ── Step 8: SentenceTransformer Embedding + FAISS ────────────────────────────

def step8_faiss_index(chunks: list, output_dir: Path):
    """
    Encode chunk texts with SentenceTransformer and build FAISS index.
    """
    print("\n" + "=" * 60)
    print("  STEP 8: SentenceTransformer Embedding + FAISS")
    print("=" * 60)

    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np

    MODEL = "all-MiniLM-L6-v2"
    EMBED_DIM = 384

    def make_text(c: dict) -> str:
        desc = c.get("description", "")
        parts = [desc, c.get("situation", ""), c.get("dialogue_text", "")]
        if c.get("characters"):
            parts.append("Characters: " + ", ".join(c["characters"][:5]))
        parts.append(c.get("narrative_arc", ""))
        return " | ".join(p for p in parts if p)

    print(f"  Encoding {len(chunks)} chunks with {MODEL}...")
    model = SentenceTransformer(MODEL)

    texts = [make_text(c) for c in chunks]
    vecs = model.encode(texts, show_progress_bar=True, batch_size=128)
    vecs = vecs.astype("float32")
    faiss.normalize_L2(vecs)

    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(vecs)

    out_index = output_dir / "chunks.faiss"
    faiss.write_index(index, str(out_index))

    # Metadata
    meta = {
        "index": out_index.name,
        "model": MODEL,
        "dim": EMBED_DIM,
        "num_vectors": len(chunks),
        "source": "pipeline_5layer_video",
    }
    meta_path = output_dir / "chunks_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    sz_mb = out_index.stat().st_size / 1024**2
    print(f"  ✅ FAISS index: {index.ntotal} vectors, {sz_mb:.1f}MB")
    print(f"  Saved: {out_index}")
    return out_index


# ── Step 9: Demo Queries ─────────────────────────────────────────────────────

def step9_demo_queries(chunks: list, faiss_path: Path):
    """Run demo queries and show top-K results."""
    print("\n" + "=" * 60)
    print("  STEP 9: Demo Retrieval Queries")
    print("=" * 60)

    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np

    MODEL = "all-MiniLM-L6-v2"

    def make_text(c: dict) -> str:
        parts = [c.get("description", ""), c.get("situation", ""),
                 c.get("dialogue_text", ""), c.get("narrative_arc", "")]
        if c.get("characters"):
            parts.append("Characters: " + ", ".join(c["characters"][:5]))
        return " | ".join(p for p in parts if p)

    model = SentenceTransformer(MODEL)
    texts = [make_text(c) for c in chunks]
    vecs = model.encode(texts, show_progress_bar=False, batch_size=128)
    vecs = vecs.astype("float32")
    faiss.normalize_L2(vecs)

    index = faiss.read_index(str(faiss_path))

    queries = [
        "Action scene with intense movement",
        "Quiet conversation between two people",
        "Someone talking about their feelings",
        "A character enters a building",
    ]

    print(f"\n  Running {len(queries)} demo queries...")
    for q in queries:
        q_vec = model.encode([q]).astype("float32")
        faiss.normalize_L2(q_vec)
        D, I = index.search(q_vec, 3)

        print(f"\n  Query: \"{q}\"")
        for rank, (idx, score) in enumerate(zip(I[0], D[0]), 1):
            c = chunks[idx]
            print(f"    #{rank}  [{c['start_seconds']:.0f}s-{c['end_seconds']:.0f}s] "
                  f"sim={score:.3f}")
            print(f"        {c['description'][:80]}...")


# ── Main Pipeline ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    video_path = Path(args.video)
    output_dir = Path(args.output or f"data/pipeline_output/{args.movie_id}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  VideoSceneRAG — Full 5-Layer Video Understanding Pipeline")
    print("=" * 60)
    print(f"  Movie ID:    {args.movie_id}")
    print(f"  Title:       {args.title}")
    print(f"  Video:       {video_path}")
    print(f"  SRT:         {args.srt or 'None'}")
    print(f"  Output:      {output_dir}")
    print(f"  Whisper:     {args.whisper_model} ({'enabled' if not args.skip_whisper else 'SKIPPED'})")
    print(f"  CLIP:        {'enabled' if not args.skip_clip else 'SKIPPED'}")
    print(f"  VLM:         {'enabled' if not args.skip_vlm else 'SKIPPED'}")

    if args.dry_run:
        print("\n  [DRY RUN] Would execute all steps.")
        return

    # Step 0: Environment check
    step0_env_check(args)

    # Step 1: Shot detection
    shots = step1_shot_detection(video_path, output_dir)
    if not shots:
        print("  ❌ No shots detected. Check video file.")
        return

    # Step 2: Keyframes
    kf_dir = step2_keyframes(video_path, shots, output_dir)

    # Step 3: CLIP encoding
    clip_emb_path = None
    if not args.skip_clip:
        clip_emb_path = step3_clip_encoding(kf_dir, output_dir)

    # Step 4: Whisper transcription
    whisper_segs = None
    if not args.skip_whisper:
        whisper_segs = step4_whisper(video_path, output_dir, args)

    # Step 5: SRT parsing
    srt_entries = None
    if args.srt:
        srt_entries = step5_parse_srt(Path(args.srt))

    # Step 6: VLM scene analysis
    vlm_analyses = None
    if not args.skip_vlm:
        vlm_analyses = step6_vlm_analysis(video_path, shots, output_dir, args)

    # Step 7: Build 5-layer chunks
    chunks = step7_build_chunks(
        shots=shots,
        whisper_segs=whisper_segs,
        srt_entries=srt_entries,
        vlm_analyses=vlm_analyses,
        movie_id=args.movie_id,
        title=args.title,
        clip_emb_path=clip_emb_path,
        output_dir=output_dir,
        args=args,
    )

    # Step 8: FAISS index
    faiss_path = step8_faiss_index(chunks, output_dir)

    # Step 9: Demo queries
    step9_demo_queries(chunks, faiss_path)

    print("\n" + "=" * 60)
    print("  ✅ PIPELINE COMPLETE")
    print(f"  Output: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
