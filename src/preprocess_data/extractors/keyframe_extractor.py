"""
Precision Scene Keyframe Extractor

Extracts keyframes directly from raw video files at exact scene boundary
timestamps from MovieNet annotation JSON files.

Adapted from: scripts/extract_scene_keyframes.py
"""

import json
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Set, Any

from preprocess_data.config import PreprocessConfig as Cfg
from preprocess_data.extractors.visual_pruner import VisualPruner

logger = logging.getLogger(__name__)


class KeyframeExtractor:
    """Extract keyframes at exact annotation scene boundaries."""

    def __init__(
        self,
        height: int = None,
        quality: int = None,
        prune: bool = True,
        clip_encoder=None,
    ):
        self.height = height or Cfg.KEYFRAME_HEIGHT
        self.quality = quality or Cfg.KEYFRAME_QUALITY
        self.prune = prune
        self.clip_encoder = clip_encoder
        if self.prune and self.clip_encoder is None:
            try:
                from movierag.indexing.clip_encoder import CLIPEncoder

                self.clip_encoder = CLIPEncoder()
            except Exception as e:
                logger.warning(f"Could not initialize CLIP encoder for pruning: {e}")
        self.pruner = (
            VisualPruner(clip_encoder=self.clip_encoder) if prune else None
        )

    @staticmethod
    def _scene_idx_from_id(scene_id: Any, fallback: int = 0) -> int:
        if isinstance(scene_id, int):
            return scene_id
        if isinstance(scene_id, str):
            match = re.search(r"(\d+)$", scene_id)
            if match:
                return int(match.group(1))
        return fallback

    @staticmethod
    def _img_idx_from_name(path_str: str) -> int:
        name = Path(path_str).name
        match = re.search(r"-(\d+)\.jpg$", name, re.IGNORECASE)
        if match:
            # Map 01/02/03 -> 0/1/2 so the middle frame keeps highest priority downstream.
            return max(int(match.group(1)) - 1, 0)

        match = re.search(r"_img_(\d+)\.jpg$", name, re.IGNORECASE)
        if match:
            return int(match.group(1))

        return 0

    @classmethod
    def _normalize_index_entry(cls, item: Dict[str, Any]) -> Dict[str, Any]:
        path = item.get("path") or item.get("frame_path", "")
        timestamp_sec = float(item.get("timestamp_sec", item.get("timestamp", 0.0)))
        scene_id = item.get("scene_id", "")
        scene_idx = item.get("scene_idx")
        if scene_idx is None:
            scene_idx = cls._scene_idx_from_id(scene_id, item.get("shot_id", 0))

        normalized = dict(item)
        normalized["path"] = str(path)
        normalized["timestamp_sec"] = timestamp_sec
        normalized["timestamp"] = timestamp_sec
        normalized["scene_id"] = scene_id
        normalized["scene_idx"] = int(scene_idx)
        normalized["shot_id"] = int(item.get("shot_id", item.get("shot_idx", 0)))
        normalized["img_idx"] = int(
            item.get("img_idx", cls._img_idx_from_name(str(path)))
        )
        return normalized

    # ── Video Info ──

    @staticmethod
    def get_video_info(video_path: Path) -> Dict:
        """Get FPS and duration via ffprobe."""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                str(video_path),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)

            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    fps_str = stream.get("r_frame_rate", "24/1")
                    parts = fps_str.split("/")
                    fps = (
                        float(parts[0]) / float(parts[1])
                        if len(parts) == 2
                        else float(parts[0])
                    )
                    duration = float(data.get("format", {}).get("duration", 0))
                    return {
                        "fps": fps,
                        "duration": duration,
                        "width": int(stream.get("width", 0)),
                        "height": int(stream.get("height", 0)),
                    }
        except Exception as e:
            logger.warning(f"ffprobe failed: {e}")
        return {"fps": 24.0, "duration": 0, "width": 0, "height": 0}

    # ── Scene Loading ──

    @staticmethod
    def load_scenes(movie_id: str, fps: float) -> List[Dict]:
        """Load annotation scenes and convert to timestamp-based entries."""
        ann_path = Cfg.get_annotation_dir() / f"{movie_id}.json"
        if not ann_path.exists():
            return []

        try:
            data = json.loads(ann_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to parse annotation: {e}")
            return []

        scenes = []
        for i, s in enumerate(data.get("scene", [])):
            frame_range = s.get("frame", [0, 0])
            if len(frame_range) < 2:
                continue

            start_sec = frame_range[0] / fps
            end_sec = frame_range[1] / fps
            if end_sec - start_sec < 0.5:
                continue

            scenes.append(
                {
                    "scene_idx": i,
                    "scene_id": s.get("id", f"scene_{i}"),
                    "frame_start": frame_range[0],
                    "frame_end": frame_range[1],
                    "start_sec": round(start_sec, 3),
                    "end_sec": round(end_sec, 3),
                    "mid_sec": round((start_sec + end_sec) / 2, 3),
                    "duration_sec": round(end_sec - start_sec, 3),
                }
            )
        return scenes

    # ── Single Frame Extraction ──

    def extract_frame_at_time(
        self, video_path: Path, timestamp_sec: float, output_path: Path
    ) -> bool:
        """Extract a single frame from video at exact timestamp."""
        h, m, s = (
            int(timestamp_sec // 3600),
            int((timestamp_sec % 3600) // 60),
            timestamp_sec % 60,
        )
        ts_str = f"{h:02d}:{m:02d}:{s:06.3f}"

        cmd = [
            "ffmpeg",
            "-ss",
            ts_str,
            "-i",
            str(video_path),
            "-vframes",
            "1",
            "-vf",
            f"scale=-1:{self.height}",
            "-qscale:v",
            str(self.quality),
            "-y",
            "-v",
            "quiet",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, check=True, timeout=30)
            return output_path.exists()
        except Exception:
            return False

    # ── Movie Processing ──

    def process_movie(self, movie_id: str, force: bool = False) -> Dict:
        """Extract chronological keyframes (6 frames) per Semantic Scene for MapReduce batching."""
        video_path = Cfg.get_video_path(movie_id)
        if not video_path:
            logger.warning(f"  ❌ No video for {movie_id}")
            return {"movie_id": movie_id, "status": "no_video", "keyframes": 0}

        info = self.get_video_info(video_path)
        fps, duration = info["fps"], info["duration"]
        logger.info(
            f"  📹 {video_path.name} | FPS: {fps:.3f} | Duration: {_fmt(duration)}"
        )

        scenes = self.load_scenes(movie_id, fps)
        if not scenes:
            return {"movie_id": movie_id, "status": "no_scenes", "keyframes": 0}

        out_dir = Cfg.get_shot_keyf_dir() / movie_id
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Thư mục cho 2 bộ dữ liệu mới
        vector_dir = out_dir / "vector_clean"
        vlm_dir = out_dir / "vlm_quality"
        vector_dir.mkdir(exist_ok=True)
        vlm_dir.mkdir(exist_ok=True)

        if not force:
            existing = list(vector_dir.glob("*.jpg"))
            if existing: # Nếu đã có dữ liệu trong folder mới
                logger.info(f"  ⏩ Đã có dữ liệu lọc ({len(existing)} frames).")
                return {"status": "skipped"}

        logger.info(f"  📊 Trích xuất từ {len(scenes)} scenes...")
        shot_img_dir = (Cfg.OUTPUT_DIR / movie_id / "shot_images") if Cfg.OUTPUT_DIR else None
        ann_path = Cfg.get_annotation_dir() / f"{movie_id}.json"

        try:
            ann_data = json.loads(ann_path.read_text(encoding="utf-8"))
            raw_shots = ann_data.get("raw_shots", [])
        except:
            raw_shots = []

        logger.info(f"  📊 Liên kết và lọc dữ liệu đa lớp...")
        vector_index = []
        vlm_index = []
        t0 = time.time()

        import shutil

        for scene in scenes:
            s_idx = scene["scene_idx"]
            scene_id = scene["scene_id"]
            start_frame = scene["frame_start"]
            end_frame = scene["frame_end"]
            scene_keyframes = []

            # Lấy các shot thuộc scene này
            scene_shots = [
                (i, s) for i, s in enumerate(raw_shots)
                if s["start_frame"] >= start_frame and s["end_frame"] <= end_frame
            ]
            if not scene_shots and raw_shots:
                scene_shots = [(i, s) for i, s in enumerate(raw_shots)
                               if max(s["start_frame"], start_frame) < min(s["end_frame"], end_frame)]

            for shot_idx, shot in scene_shots:
                shot_name_idx = shot_idx + 1
                for img_num in ["01", "02", "03"]:
                    fname = f"{movie_id}_shot_{shot_name_idx:03d}-{img_num}.jpg"
                    thumb_path = shot_img_dir / fname if shot_img_dir else None

                    if thumb_path and thumb_path.exists():
                        shot_dur = shot["end_sec"] - shot["start_sec"]
                        ts = shot["start_sec"] + (shot_dur * (int(img_num) - 0.5) / 3.0)
                        scene_keyframes.append({
                            "filename": fname,
                            "scene_idx": s_idx,
                            "scene_id": scene_id,
                            "shot_idx": shot_idx,
                            "path": str(thumb_path),
                            "timestamp_sec": round(ts, 3),
                        })

            # Fallback: if no pre-extracted shot images found, extract directly from video
            if not scene_keyframes and video_path:
                start_sec = scene["start_sec"]
                end_sec   = scene["end_sec"]
                dur       = end_sec - start_sec
                for k, frac in enumerate([0.25, 0.5, 0.75]):
                    ts = start_sec + dur * frac
                    fname     = f"{movie_id}_scene_{s_idx:03d}-{k+1:02d}.jpg"
                    out_path  = out_dir / fname
                    if force or not out_path.exists():
                        self.extract_frame_at_time(video_path, ts, out_path)
                    if out_path.exists():
                        scene_keyframes.append({
                            "filename": fname,
                            "scene_idx": s_idx,
                            "scene_id": scene_id,
                            "shot_idx": 0,
                            "path": str(out_path),
                            "timestamp_sec": round(ts, 3),
                        })

            # --- Lọc Đa Lớp (Advanced Visual Pruning) ---
            if self.prune and self.pruner and scene_keyframes:
                frame_paths = [kf["path"] for kf in scene_keyframes]
                shot_map = [kf["shot_idx"] for kf in scene_keyframes]
                timestamps = [kf["timestamp_sec"] for kf in scene_keyframes]
                
                l1_frames, l2_frames = self.pruner.prune_scene(
                    movie_id=movie_id,
                    scene_idx=s_idx,
                    frame_paths=frame_paths,
                    shot_map=shot_map,
                    timestamps=timestamps,
                )
                
                # Xử lý Layer 1 (Vector Clean)
                for kf in l1_frames:
                    src = Path(kf["frame_path"]) # Note: new API uses 'frame_path'
                    dst = vector_dir / src.name
                    if not dst.exists() and src.exists():
                        shutil.copy2(src, dst)
                    
                    # Store in index with legacy-friendly keys
                    kf_idx = kf.copy()
                    kf_idx["path"] = str(dst)
                    kf_idx["scene_id"] = scene_id # Restore original scene_id string
                    vector_index.append(kf_idx)
                
                # Xử lý Layer 2 (VLM Quality)
                for kf in l2_frames:
                    src = Path(kf["frame_path"])
                    dst = vlm_dir / src.name
                    if not dst.exists() and src.exists():
                        shutil.copy2(src, dst)
                    
                    kf_idx = kf.copy()
                    kf_idx["path"] = str(dst)
                    kf_idx["scene_id"] = scene_id
                    vlm_index.append(kf_idx)
            else:
                # Nếu không lọc, dùng chung cho cả 2 (không khuyến khích)
                vector_index.extend(scene_keyframes)
                vlm_index.extend(scene_keyframes)

        # Lưu 2 file index riêng biệt
        def save_idx(data_list, name):
            idx_path = out_dir / f"{name}_index.json"
            normalized_keyframes = [
                self._normalize_index_entry(item) for item in data_list
            ]
            idx_data = {
                "movie_id": movie_id,
                "total_frames": len(normalized_keyframes),
                "video_fps": fps,
                "keyframes": normalized_keyframes,
            }
            idx_path.write_text(json.dumps(idx_data, indent=2, ensure_ascii=False), encoding="utf-8")
            return idx_path

        save_idx(vector_index, "vector_clean")
        save_idx(vlm_index, "vlm_quality")
        
        # Restore legacy index for backward compatibility (CVFaceExtractor, VLMVisionExtractor, etc.)
        save_idx(vlm_index, "keyframe")

        elapsed = time.time() - t0
        logger.info(f"  ✅ Đã lọc xong: Vector={len(vector_index)}, VLM={len(vlm_index)} trong {elapsed:.1f}s")
        return {
            "movie_id": movie_id,
            "status": "ok",
            "vector_count": len(vector_index),
            "vlm_count": len(vlm_index),
            "elapsed": round(elapsed, 1),
        }

    def process_movie_fixed_interval(
        self,
        movie_id: str,
        video_path: Path,
        force: bool = False,
        interval: float = 5.0,
    ) -> Dict:
        """Fallback: Extract keyframes at fixed intervals when no annotation exists."""
        info = self.get_video_info(video_path)
        duration = info["duration"]

        out_dir = Cfg.get_shot_keyf_dir() / movie_id
        out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"  📊 Extracting at fixed {interval}s intervals...")
        extracted = 0
        t0 = time.time()

        t = 0.0
        idx = 0
        while t < duration:
            fname = f"fixed_{idx:04d}.jpg"
            out_path = out_dir / fname
            if self.extract_frame_at_time(video_path, t, out_path):
                extracted += 1
            t += interval
            idx += 1

        elapsed = time.time() - t0
        return {
            "movie_id": movie_id,
            "status": "ok",
            "keyframes": extracted,
            "elapsed": round(elapsed, 1),
        }

    def process_all(
        self, movie_ids: List[str] = None, force: bool = False
    ) -> List[Dict]:
        """Process multiple movies."""
        ids = movie_ids or self._get_processable_ids()
        logger.info(f"Processing {len(ids)} movies...")
        results = []
        for i, mid in enumerate(ids, 1):
            logger.info(f"\n[{i}/{len(ids)}] {mid}")
            results.append(self.process_movie(mid, force=force))
        return results

    @staticmethod
    def _get_processable_ids() -> List[str]:
        """Movies that have both annotation AND raw video."""
        ann_dir = Cfg.get_annotation_dir()
        ann_ids = (
            {p.stem for p in ann_dir.glob("*.json")} if ann_dir.exists() else set()
        )
        vid_ids = set()
        for d in [Cfg.RAW_VIDEOS_DIR, Cfg.RAW_MOVIES_DIR]:
            if d.exists():
                vid_ids |= {
                    p.stem
                    for p in d.glob("*.*")
                    if p.suffix in {".mp4", ".mkv", ".avi"}
                }
        return sorted(ann_ids & vid_ids)


def _fmt(seconds: float) -> str:
    h, m, s = int(seconds // 3600), int((seconds % 3600) // 60), int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
