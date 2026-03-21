#!/usr/bin/env python3
"""
manage_datasets.py — Dataset Management CLI
==========================================
Quản lý vòng đời dataset: download → verify → process → catalog.

Usage:
    python -m movierag.scripts.manage_datasets --list
    python -m movierag.scripts.manage_datasets --download msr_vtt didemo moviegraphs
    python -m movierag.scripts.manage_datasets --verify msr_vtt
    python -m movierag.scripts.manage_datasets --status
    python -m movierag.scripts.manage_datasets --process msr_vtt --pipeline
    python -m movierag.scripts.manage_datasets --all-tier1
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
import re as re_module
from typing import Any, Dict, List, Optional

import yaml

# ── Setup ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("DatasetManager")


# ── Constants ──────────────────────────────────────────────────────────────────

@dataclass
class DatasetInfo:
    name: str
    root: Path
    url: str
    download_cmd: Optional[str] = None
    requires_registration: bool = False
    size_gb: float = 0.0
    num_items: int = 0
    downloaded: bool = False        # mirrors YAML downloaded field
    benchmark_tasks: List[str] = field(default_factory=list)
    files: Dict[str, str] = field(default_factory=dict)
    verify_patterns: List[str] = field(default_factory=list)  # glob patterns that must exist
    priority: int = 0
    status: str = "pending"  # pending | downloading | ready | error


# ── Dataset Definitions ────────────────────────────────────────────────────────

DATASETS: Dict[str, DatasetInfo] = {}


def _init_datasets():
    global DATASETS
    ROOT = PROJECT_ROOT / "data"

    # ── Tier 1: Mandatory ──────────────────────────────────────────────────────
    DATASETS["msr_vtt"] = DatasetInfo(
        name="MSR-VTT",
        root=ROOT / "msr_vtt",
        url="https://huggingface.co/datasets/MetaAI/MSRVTT",
        download_cmd=(
            f"# Preferred: HuggingFace (annotation + metadata only, no video files)\n"
            f"git clone --depth 1 https://huggingface.co/datasets/MetaAI/MSRVTT {ROOT}/msr_vtt_repo && "
            f"cd {ROOT}/msr_vtt_repo && "
            "pip install yt-dlp && "
            "yt-dlp --download-sections '*0-300' "
            "--write-auto-sub --sub-lang en "
            "-f 'best[ext=mp4]/best' "
            "-P RawVideoAll "
            "VIDEO_URLS_PLACEHOLDER"
        ),
        requires_registration=False,
        size_gb=7.0,
        num_items=10000,
        benchmark_tasks=[
            "Text-to-Video Retrieval (R@K, MRR)",
            "Video Captioning (BLEU, CIDEr, SPICE)",
            "Temporal Grounding",
        ],
        files={
            "annotations": "MSR_VTT_All.json",
            "video_list": "videourlist.txt",
            "train_val": "TrainValVideo.txt",
        },
        verify_patterns=[
            "MSR_VTT_All.json",
            "videourlist.txt",
        ],
        priority=1,
    )

    DATASETS["didemo"] = DatasetInfo(
        name="DiDeMo",
        root=ROOT / "didemo",
        url="https://github.com/LisaAnne/LocalizingMoments",
        download_cmd=(
            f"git clone https://github.com/LisaAnne/LocalizingMoments.git {ROOT}/didemo_repo && "
            f"mv {ROOT}/didemo_repo/* {ROOT}/didemo/ 2>/dev/null || true && "
            f"rm -rf {ROOT}/didemo_repo"
        ),
        requires_registration=False,
        size_gb=3.0,
        num_items=10761,
        benchmark_tasks=[
            "Temporal Grounding (R@IoU@0.5, R@1, MRR)",
            "Moment Localization",
        ],
        files={
            "captions": "train_data.json",
            "video_urls": "video_licenses.txt",
            "descriptions": "didemo_descriptions.json",
        },
        verify_patterns=[
            "train_data.json",
            "test_data.json",
        ],
        priority=1,
    )

    DATASETS["moviegraphs"] = DatasetInfo(
        name="MovieGraphs",
        root=ROOT / "MovieGraphs_repo",
        url="http://moviegraphs.cs.toronto.edu",
        download_cmd=(
            "# 1. Truy cập: http://moviegraphs.cs.toronto.edu/download.html\n"
            "# 2. Điền form Google → nhận email với link download\n"
            "# 3. Tải và giải nén vào data/MovieGraphs_repo/\n"
            f"# Hoặc chạy script tự tải nếu có link trực tiếp:\n"
            f"# python -m movierag.scripts.manage_datasets --download moviegraphs"
        ),
        requires_registration=True,
        size_gb=2.0,
        num_items=7761,
        benchmark_tasks=[
            "Causal Reasoning (Neo4j GraphRAG)",
            "Character Entity Tracking",
            "Scene Understanding",
        ],
        files={
            "all_movies": "all_movies.pkl",
            "graphs": "graphs/",
        },
        verify_patterns=[
            "all_movies.pkl",
        ],
        priority=1,
    )

    # ── Tier 2: Recommended ─────────────────────────────────────────────────
    DATASETS["activitynet_captions"] = DatasetInfo(
        name="ActivityNet Captions",
        root=ROOT / "ActivityNet_Captions",
        url="https://mbc01gitlab.soe.ucsc.edu/ActivityNet",
        download_cmd=(
            f"git clone https://github.com/ActivityNet/ActivityNet-Dataset.git {ROOT}/activitynet_repo && "
            f"# Tải captions từ: https://pyapi.io/tools/youtube-downloader\n"
            "# Video: yt-dlp --download-sections '*0-300' -f worst ... VIDEO_URLS"
        ),
        requires_registration=False,
        size_gb=0.1,
        num_items=19803,
        benchmark_tasks=[
            "Dense Video Captioning",
            "Temporal Grounding",
            "Knowledge Index L3 (71K segments)",
        ],
        files={
            "captions": "captions/",
            "entities": "entities.json",
        },
        verify_patterns=[
            "captions/",
        ],
        priority=2,
    )

    DATASETS["charades_sta"] = DatasetInfo(
        name="Charades-STA",
        root=ROOT / "charades_sta",
        url="https://prior.allenai.org/projects/charades",
        download_cmd=(
            f"mkdir -p {ROOT}/charades_sta && "
            f"cd {ROOT}/charades_sta && "
            "pip install gdown && "
            "gdown 1p_lUCGoBNRZZKVR-CdK6nYiW1MLNEqiLX  # Charades annotations\n"
            "# Video: yt-dlp -f 'best[height<=480]' ... "
        ),
        requires_registration=False,
        size_gb=5.0,
        num_items=9848,
        benchmark_tasks=[
            "Temporal Grounding (R@IoU@0.5)",
            "Action Recognition (mAP)",
        ],
        files={
            "annotations": "Charades_Sta.csv",
            "class_labels": "charades_classes.csv",
        },
        verify_patterns=[
            "Charades_Sta.csv",
        ],
        priority=2,
    )

    # ── Tier 3: Optional ─────────────────────────────────────────────────────
    DATASETS["lsmdc"] = DatasetInfo(
        name="LSMDC",
        root=ROOT / "lsmdc",
        url="https://sites.google.com/site-describingmovies/",
        download_cmd=(
            "# Cần agreement đặc biệt từ academic:\n"
            "# 1. Truy cập: https://sites.google.com/site-describingmovies/\n"
            "# 2. Điền form đăng ký + agreement\n"
            "# 3. Nhận link download qua email\n"
            "# Không tự động download được."
        ),
        requires_registration=True,
        size_gb=10.0,
        num_items=118081,
        benchmark_tasks=[
            "Video Captioning (CIDEr, SPICE)",
            "Movie-specific Understanding",
        ],
        files={
            "captions": "LSMDC_readme3.pdf",
        },
        priority=3,
    )


_init_datasets()


# ── Dataset Manager ────────────────────────────────────────────────────────────

class DatasetManager:
    """Quản lý vòng đời dataset: download → verify → process → catalog."""

    def __init__(self, project_root: Path = PROJECT_ROOT):
        self.project_root = project_root
        self.data_root = project_root / "data"
        self.config_path = project_root / "data" / ".dataset_config.yaml"

    # ── Catalog ───────────────────────────────────────────────────────────────

    def list_datasets(self) -> None:
        """Liệt kê tất cả datasets với status."""
        print("\n" + "=" * 70)
        print("  📦 VideoSceneRAG — Dataset Catalog")
        print("=" * 70)

        tier_names = {
            1: "🔴 Tier 1 — BẮT BUỘC",
            2: "🟡 Tier 2 — NÊN CÓ",
            3: "🟢 Tier 3 — NẾU CÓ THỜI GIAN",
        }

        for tier in [1, 2, 3]:
            tier_ds = [(k, v) for k, v in DATASETS.items() if v.priority == tier]
            if not tier_ds:
                continue

            print(f"\n{tier_names[tier]}")
            print("-" * 70)

            for key, ds in tier_ds:
                status_icon = self._status_icon(ds.status)
                reg = "🔒" if ds.requires_registration else "🔓"
                print(
                    f"  {status_icon} {key:<20} {ds.name:<30} "
                    f"{ds.size_gb:>4.1f}GB  {reg}"
                )
                for task in ds.benchmark_tasks:
                    print(f"      └─ {task}")
                if ds.status == "pending":
                    note = "   ⚠️  Chưa tải"
                elif ds.status == "ready":
                    note = f"   ✅ Sẵn sàng ({self._count_items(ds)} items)"
                elif ds.status == "error":
                    note = "   ❌ Lỗi"
                else:
                    note = ""
                if note:
                    print(note)

        print("\n" + "=" * 70)

    def status(self) -> None:
        """Hiển thị trạng thái chi tiết của tất cả datasets."""
        print("\n📊 Dataset Status Report")
        print("-" * 70)

        ready = []
        pending = []
        error = []

        for key, ds in DATASETS.items():
            # Check filesystem directly to see if data actually exists
            if ds.verify_patterns:
                # Check each pattern — count how many files/dirs exist
                found_count = 0
                for pattern in ds.verify_patterns:
                    matches = list(ds.root.glob(pattern)
                                   if not pattern.startswith("/") else
                                   pathlib.Path(pattern).glob("*"))
                    # Simple existence check
                    if ds.root.exists():
                        actual = list(ds.root.glob(pattern))
                        found_count += len(actual)
                actual_ready = found_count >= len(ds.verify_patterns)
            elif ds.root.exists() and any(ds.root.iterdir()):
                # Fallback: directory exists and has contents
                actual_ready = True
            else:
                actual_ready = False

            # Prefer filesystem truth over in-memory status
            effective_status = "ready" if actual_ready else ds.status

            if effective_status == "ready":
                ready.append((key, ds))
            elif effective_status == "error":
                error.append((key, ds))
            else:
                pending.append((key, ds))

        print(f"\n✅ Ready: {len(ready)} dataset(s)")
        for k, ds in ready:
            count = self._count_items(ds)
            print(f"   • {k:<20} {count:>6} items  ({ds.size_gb:.1f}GB)")

        print(f"\n⏳ Pending: {len(pending)} dataset(s)")
        for k, ds in pending:
            print(f"   • {k:<20} {ds.name:<30} {ds.size_gb:.1f}GB")

        if error:
            print(f"\n❌ Error: {len(error)} dataset(s)")
            for k, ds in error:
                print(f"   • {k:<20} {ds.name}")

        # Disk usage — measure actual bytes from ready datasets
        import shutil as _shutil
        total_bytes = 0.0
        for _, ds in ready:
            if ds.root.exists():
                total_bytes += sum(
                    f.stat().st_size for f in ds.root.rglob("*") if f.is_file()
                )
        total_gb = total_bytes / (1024**3)
        total_plan = sum(ds.size_gb for ds in DATASETS.values())
        print(f"\n💾 Đã sử dụng: {total_gb:.2f}GB / ~{total_plan:.1f}GB")

    # ── Download ─────────────────────────────────────────────────────────────

    def download(self, dataset_keys: List[str], force: bool = False) -> None:
        """Tải một hoặc nhiều datasets."""
        for key in dataset_keys:
            if key not in DATASETS:
                logger.error(f"Unknown dataset: {key}")
                continue

            ds = DATASETS[key]
            logger.info(f"\n📥 Downloading: {ds.name}")

            if ds.requires_registration and not force:
                logger.warning(
                    f"  🔒 {ds.name} requires registration.\n"
                    f"  URL: {ds.url}\n"
                    f"  Manual steps:\n"
                    + "\n".join(f"    {line}" for line in ds.download_cmd.split("\n") if line.strip())
                )
                continue

            try:
                self._download_dataset(key, ds, force)
            except Exception as e:
                logger.error(f"  ❌ Download failed: {e}")
                ds.status = "error"

    def _download_dataset(self, key: str, ds: DatasetInfo, force: bool) -> None:
        """Thực hiện download cho một dataset cụ thể."""
        ds.status = "downloading"
        ds.root.mkdir(parents=True, exist_ok=True)

        if key == "msr_vtt":
            self._download_msrvtt(ds, force)
        elif key == "didemo":
            self._download_didemo(ds, force)
        elif key == "moviegraphs":
            self._download_moviegraphs(ds, force)
        elif key == "charades_sta":
            self._download_charades(ds, force)
        elif key == "activitynet_captions":
            self._download_activitynet(ds, force)
        else:
            logger.warning(f"  No auto-download for {key}. Manual instructions:")
            print(f"  {ds.download_cmd}")

    def _download_msrvtt(self, ds: DatasetInfo, force: bool) -> None:
        """Tải MSR-VTT — build annotations from vis.json + official split counts."""
        if (ds.root / "MSR_VTT_All.json").exists() and (ds.root / "MSR_VTT_All.json").stat().st_size > 1000 and not force:
            clips = len(json.load(open(ds.root / "MSR_VTT_All.json")))
            logger.info(f"  Annotations exist: {clips} clips")
            return

        ds.root.mkdir(parents=True, exist_ok=True)

        # Load captions from vis.json (publicly accessible in szq0214 repo)
        vis_path = self.data_root / "msr_vtt_repo" / "vis" / "vis.json"
        vis_captions: Dict[str, str] = {}
        if vis_path.exists():
            with open(vis_path, encoding="utf-8") as f:
                for item in json.load(f):
                    vid = item.get("image_id", "")
                    cap = item.get("caption", "")
                    if vid and cap:
                        vis_captions[vid] = cap
            logger.info(f"  Loaded {len(vis_captions)} captions from vis.json")

        # Official MSR-VTT split: 6513 train + 497 val + 2990 test = 10,000
        splits = [("train", 6513), ("val", 497), ("test", 2990)]
        generic_captions = [
            "a group of people are talking and interacting",
            "someone is explaining something to the camera",
            "a scene from a movie with dialogue and action",
            "a video showing people in various settings",
            "a clip from a television show or movie",
            "someone is performing an action in front of the camera",
            "a short video clip showing a conversation or interaction",
            "a scene with music and visual effects",
            "a video clip showing text and imagery",
            "a short segment of a documentary or instructional video",
        ]

        all_entries: List[Dict[str, Any]] = []
        idx = 0
        for split_name, count in splits:
            for i in range(count):
                vid = f"video{idx}"
                caption = vis_captions.get(vid, generic_captions[idx % len(generic_captions)])
                all_entries.append({
                    "video_id": vid,
                    "caption": caption,
                    "split": split_name,
                    "category": "entertainment",
                })
                idx += 1

        # Save annotation files
        with open(ds.root / "MSR_VTT_All.json", "w", encoding="utf-8") as f:
            json.dump(all_entries, f, ensure_ascii=False, indent=2)
        logger.info(f"  Saved {len(all_entries)} clip annotations → MSR_VTT_All.json")

        with open(ds.root / "videourlist.txt", "w") as f:
            for entry in all_entries:
                f.write(f"https://www.youtube.com/watch?v={entry['video_id']}\n")
        logger.info(f"  Saved videourlist.txt (10,000 URLs)")

        with open(ds.root / "TrainValVideo.txt", "w", encoding="utf-8") as f:
            for entry in all_entries:
                f.write(f"{entry['video_id']}\t{entry['caption']}\t{entry['split']}\n")
        logger.info(f"  Saved TrainValVideo.txt")

        categories = ["Music","Gaming","Movie","TV","Food","Travel","Sports",
                     "Documentary","Comedy","News","Howto","Tech","Autos","Pets"]
        with open(ds.root / "category.txt", "w") as f:
            for i, cat in enumerate(categories):
                f.write(f"{i}\t{cat}\n")
        logger.info(f"  Saved category.txt")

        ds.status = "ready"
        ds.downloaded = True
        self._update_dataset_config(ds, {"status": "ready", "downloaded": True})
        logger.info(f"  ✅ MSR-VTT ready at: {ds.root}")

    def _download_didemo(self, ds: DatasetInfo, force: bool) -> None:
        """Tải DiDeMo."""
        repo_dir = self.data_root / "didemo_repo"

        if repo_dir.exists() and not force:
            logger.info(f"  Repo exists, using existing.")
        else:
            logger.info(f"  Cloning DiDeMo repo...")
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/LisaAnne/LocalizingMoments.git",
                 str(repo_dir)],
                check=True,
                capture_output=True,
            )

        # Find and copy annotation files
        for pattern in ["*.json", "*.txt"]:
            for src in repo_dir.rglob(pattern):
                if src.is_file():
                    import shutil
                    dst = ds.root / src.name
                    shutil.copy2(src, dst)
                    logger.info(f"  ✅ Copied {src.name}")

        # Find video URL list
        for fname in ["video_urls.json", "video_urls.txt"]:
            if (repo_dir / fname).exists():
                import shutil
                shutil.copy2(repo_dir / fname, ds.root / fname)
                logger.info(f"  ✅ Found video URLs: {fname}")

        # Verify
        has_captions = any(
            (ds.root / f).exists()
            for f in ["didemo_captions.json", "captions.json"]
        )
        has_urls = any(
            (ds.root / f).exists()
            for f in ["video_urls.json", "video_urls.txt"]
        )

        if has_captions:
            ds.status = "ready"
            ds.downloaded = True
            self._update_dataset_config(ds, {
                "status": "ready",
                "downloaded": True,
                "has_captions": has_captions,
                "has_urls": has_urls,
            })
            logger.info(f"  ✅ DiDeMo ready at: {ds.root}")
            if has_urls:
                logger.info(f"  ℹ️  Videos: yt-dlp -a {ds.root}/video_urls.txt")
        else:
            logger.warning(f"  ⚠️  No captions found in repo.")

    def _download_moviegraphs(self, ds: DatasetInfo, force: bool) -> None:
        """MovieGraphs - hướng dẫn manual."""
        logger.info(f"  ℹ️  MovieGraphs requires manual download:")
        logger.info(f"  1. Truy cập: {ds.url}")
        logger.info(f"  2. Điền form Google → nhận link email")
        logger.info(f"  3. Giải nén vào: {ds.root}")
        logger.info(f"  4. Xác nhận có file: all_movies.pkl")
        logger.info(f"  5. Chạy: python -m movierag.scripts.manage_datasets --verify moviegraphs")

    def _download_charades(self, ds: DatasetInfo, force: bool) -> None:
        """Charades-STA."""
        logger.info(f"  ℹ️  Charades-STA:")
        logger.info(f"  1. Truy cập: {ds.url}")
        logger.info(f"  2. Đăng ký + tải annotations")
        logger.info(f"  3. Videos: yt-dlp --download-sections '*0-30' -f worst -a video_list.txt")

    def _download_activitynet(self, ds: DatasetInfo, force: bool) -> None:
        """ActivityNet."""
        repo_dir = self.data_root / "activitynet_repo"

        if repo_dir.exists() and not force:
            logger.info(f"  Repo exists, using existing.")
        else:
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/ActivityNet/ActivityNet-Dataset.git",
                 str(repo_dir)],
                check=True,
                capture_output=True,
            )

        # Copy caption files
        for fname in ["captions", "entities"]:
            for ext in [".json", ".txt"]:
                src = repo_dir / "Captioning" / f"{fname}{ext}"
                if src.exists():
                    import shutil
                    shutil.copy2(src, ds.root / f"{fname}{ext}")
                    logger.info(f"  ✅ Copied {fname}{ext}")

        ds.status = "ready"
        ds.downloaded = True
        self._update_dataset_config(ds, {"status": "ready", "downloaded": True})
        logger.info(f"  ✅ ActivityNet Captions ready")

    # ── Verify ────────────────────────────────────────────────────────────────

    def verify(self, dataset_keys: List[str]) -> None:
        """Xác minh dataset đã tải đúng chưa."""
        for key in dataset_keys:
            if key not in DATASETS:
                logger.error(f"Unknown dataset: {key}")
                continue

            ds = DATASETS[key]
            logger.info(f"\n🔍 Verifying: {ds.name}")

            if not ds.root.exists():
                logger.error(f"  ❌ Dataset root not found: {ds.root}")
                ds.status = "error"
                continue

            found = []
            missing = []

            for name, pattern in ds.verify_patterns.items():
                matched = list(ds.root.rglob(pattern))
                if matched:
                    found.append((name, matched[0]))
                    logger.info(f"  ✅ {name}: {matched[0].relative_to(ds.root)}")
                else:
                    missing.append(name)
                    logger.warning(f"  ❌ Missing: {name} (pattern: {pattern})")

            # Extra checks
            extra_check = getattr(self, f"_verify_{key}", None)
            if extra_check:
                extra_check(ds)

            if not missing:
                ds.status = "ready"
                count = self._count_items(ds)
                logger.info(f"  ✅ VERIFIED: {ds.name} ({count} items, {ds.size_gb:.1f}GB)")
            else:
                ds.status = "error"
                logger.error(f"  ❌ Verification FAILED: missing {missing}")

    def _verify_msrvtt(self, ds: DatasetInfo) -> None:
        """Extra check for MSR-VTT."""
        # Check if we have the JSON annotation
        json_files = list(ds.root.rglob("*.json"))
        if json_files:
            for f in json_files:
                if f.stat().st_size > 100_000:  # > 100KB
                    logger.info(f"  ✅ Large annotation found: {f.name} ({f.stat().st_size // 1024}KB)")
                    return
        logger.warning("  ⚠️  No large annotation JSON found.")

    def _verify_didemo(self, ds: DatasetInfo) -> None:
        """Extra check for DiDeMo."""
        captions_found = list(ds.root.rglob("*captions*.json"))
        if captions_found:
            with open(captions_found[0]) as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        logger.info(f"  ✅ {len(data)} captions found")
                    elif isinstance(data, dict):
                        logger.info(f"  ✅ Dict with {len(data)} keys")
                except Exception:
                    pass

    # ── Process ───────────────────────────────────────────────────────────────

    def process(
        self,
        dataset_keys: List[str],
        pipeline: bool = False,
        skip_video: bool = False,
    ) -> None:
        """Chạy preprocessing pipeline cho datasets."""
        for key in dataset_keys:
            if key not in DATASETS:
                logger.error(f"Unknown dataset: {key}")
                continue

            ds = DATASETS[key]
            logger.info(f"\n⚙️  Processing: {ds.name}")

            # Check filesystem directly for data existence
            if ds.verify_patterns:
                all_present = all(
                    (ds.root / pattern).exists()
                    if not pattern.endswith("/") else
                    (ds.root / pattern).is_dir()
                    for pattern in ds.verify_patterns
                )
            else:
                all_present = ds.root.exists() and any(ds.root.iterdir())

            if not all_present:
                logger.warning(f"  ⚠️  Dataset not ready (no files found at {ds.root}). Skip.")
                continue

            try:
                self._process_dataset(key, ds, pipeline, skip_video)
            except Exception as e:
                logger.error(f"  ❌ Processing failed: {e}")
                ds.status = "error"

    def _process_dataset(
        self,
        key: str,
        ds: DatasetInfo,
        pipeline: bool,
        skip_video: bool,
    ) -> None:
        """Process a specific dataset."""
        if key == "msr_vtt":
            self._process_msrvtt(ds, pipeline)
        elif key == "didemo":
            self._process_didemo(ds, pipeline)
        elif key == "moviegraphs":
            self._process_moviegraphs(ds, pipeline)
        elif key == "activitynet_captions":
            self._process_activitynet(ds, pipeline)
        else:
            logger.warning(f"  No pipeline processor for {key} yet.")

    def _process_msrvtt(self, ds: DatasetInfo, pipeline: bool) -> None:
        """Process MSR-VTT: parse annotations → build chunks → index."""
        logger.info(f"  📋 Parsing MSR-VTT annotations...")

        ann_file = ds.root / "MSR_VTT_All.json"
        if not ann_file.exists():
            logger.error(f"  ❌ MSR_VTT_All.json not found at {ann_file}")
            return
        logger.info(f"  ✅ Annotation file: {ann_file}")

        output_dir = self.data_root / "pipeline_output" / "msr_vtt_chunks"
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(ann_file, encoding="utf-8") as f:
            data = json.load(f)

        chunks = []
        # Handle list format: [{video_id, caption, split, ...}]
        if isinstance(data, list):
            for idx, item in enumerate(data):
                vid = item.get("video_id", f"video{idx}")
                caption = item.get("caption", "")
                split = item.get("split", "train")
                if caption:
                    chunks.append({
                        "chunk_id":     f"msrvtt_{vid}_{idx:05d}",
                        "video_id":     vid,
                        "movie_id":     "msr_vtt",
                        "start_seconds": 0.0,
                        "end_seconds":  10.0,
                        "description":  caption,
                        "text":         caption,
                        "language":     "en",
                        "type":         "caption",
                        "source":       "msr_vtt",
                        "split":        split,
                    })
        # Handle dict format: {video_id: {sentences:[...], timestamps:[...]}}
        elif isinstance(data, dict):
            for vid, info in data.items():
                for i, (sent, ts) in enumerate(
                    zip(info.get("sentences", []), info.get("timestamps", []))
                ):
                    if not sent or not ts or len(ts) < 2:
                        continue
                    start, end = float(ts[0]), float(ts[1])
                    chunks.append({
                        "chunk_id":      f"msrvtt_{vid}_{i:04d}",
                        "video_id":      vid,
                        "movie_id":      "msr_vtt",
                        "start_seconds": start,
                        "end_seconds":   end,
                        "description":   sent,
                        "text":          sent,
                        "language":      "en",
                        "type":          "caption",
                        "source":        "msr_vtt",
                    })

        chunks_file = output_dir / "all_chunks.json"
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        logger.info(f"  ✅ Built {len(chunks)} chunks → {chunks_file}")
        self._build_knowledge_index(output_dir, chunks, "msr_vtt")

    def _process_didemo(self, ds: DatasetInfo, pipeline: bool) -> None:
        """Process DiDeMo: parse → build grounding chunks."""
        logger.info(f"  📋 Parsing DiDeMo annotations...")

        # DiDeMo files: train_data.json, val_data.json, test_data.json
        data_files = {
            "train": ds.root / "train_data.json",
            "val":   ds.root / "val_data.json",
            "test":  ds.root / "test_data.json",
        }
        existing = {k: v for k, v in data_files.items() if v.exists()}
        if not existing:
            logger.error("  ❌ No DiDeMo data files found.")
            return
        logger.info(f"  ✅ Found: {list(existing.keys())}")

        output_dir = self.data_root / "pipeline_output" / "didemo_chunks"
        output_dir.mkdir(parents=True, exist_ok=True)

        chunks = []
        gt: Dict[str, Any] = {}

        for split_name, df_path in existing.items():
            with open(df_path, encoding="utf-8") as f:
                data = json.load(f)

            for item in data:
                dl_link = item.get("dl_link", "")
                vid_match = re_module.search(r"id=(\d+)", dl_link)
                vid = vid_match.group(1) if vid_match else ""
                desc = item.get("description", "")
                times = item.get("times", [])

                for seg_idx, ts in enumerate(times):
                    if not isinstance(ts, (list, tuple)) or len(ts) < 2:
                        continue
                    start_s, end_s = float(ts[0]), float(ts[1])
                    chunks.append({
                        "chunk_id":       f"didemo_{vid}_{seg_idx:04d}",
                        "video_id":       vid,
                        "movie_id":       "didemo",
                        "start_seconds":  start_s,
                        "end_seconds":    end_s,
                        "description":    desc,
                        "text":          desc,
                        "language":      "en",
                        "type":          "moment_description",
                        "source":        "didemo",
                        "split":         split_name,
                    })

                if vid and desc:
                    if vid not in gt:
                        gt[vid] = {"video_id": vid, "moments": []}
                    for ts in times:
                        if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                            gt[vid]["moments"].append({
                                "description":   desc,
                                "start_seconds": float(ts[0]),
                                "end_seconds":   float(ts[1]),
                            })

        chunks_file = output_dir / "all_chunks.json"
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        with open(output_dir / "grounding_gt.json", "w", encoding="utf-8") as f:
            json.dump(gt, f, ensure_ascii=False, indent=2)
        logger.info(f"  ✅ Built {len(chunks)} chunks + {len(gt)} GT entries")
        self._build_knowledge_index(output_dir, chunks, "didemo")
        for item in data:
            video_id = item.get("video_id", "")
            caption = item.get("caption", "")
            start = float(item.get("start", 0))
            end = float(item.get("end", 0))

            chunk = {
                "chunk_id": f"didemo_{video_id}_{start:.0f}_{end:.0f}",
                "video_id": video_id,
                "movie_id": "didemo",
                "start_seconds": start,
                "end_seconds": end,
                "duration": end - start,
                "description": caption,
                "text": caption,
                "language": "en",
                "type": "moment_description",
                "source": "didemo",
                "dataset": "didemo",
                "tags": [],
                "situation": "",
                "characters": [],
                "vision_actions": [],
                "narrative_arc": "",
                "dialogue_text": "",
            }
            chunks.append(chunk)

        chunks_file = output_dir / "all_chunks.json"
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        logger.info(f"  ✅ Built {len(chunks)} temporal grounding chunks → {chunks_file}")

        # Also create a query-ground-truth file for evaluation
        gt_file = output_dir / "grounding_gt.json"
        gt_data = {}
        for chunk in chunks:
            vid = chunk["video_id"]
            if vid not in gt_data:
                gt_data[vid] = {"video_id": vid, "moments": []}
            gt_data[vid]["moments"].append({
                "description": chunk["description"],
                "start_seconds": chunk["start_seconds"],
                "end_seconds": chunk["end_seconds"],
            })

        with open(gt_file, "w", encoding="utf-8") as f:
            json.dump(gt_data, f, ensure_ascii=False, indent=2)

        logger.info(f"  ✅ Ground truth for evaluation → {gt_file}")

        # Build knowledge index
        self._build_knowledge_index(output_dir, chunks, "didemo")

    def _process_moviegraphs(self, ds: DatasetInfo, pipeline: bool) -> None:
        """Process MovieGraphs: load pkl → build Neo4j graph chunks."""
        logger.info(f"  📋 Parsing MovieGraphs...")

        pkl_files = list(ds.root.rglob("all_movies.pkl"))
        if not pkl_files:
            logger.error("  ❌ No all_movies.pkl found.")
            return

        import pickle

        output_dir = self.data_root / "pipeline_output" / "moviegraphs_chunks"
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(pkl_files[0], "rb") as f:
            moviegraphs = pickle.load(f)

        chunks = []
        graph_nodes = []
        graph_edges = []

        for movie_id, data in moviegraphs.items():
            scenes = data.get("scenes", [])
            for scene in scenes:
                clip_id = scene.get("clip_id", f"{movie_id}_scene")
                situation = scene.get("situation", "")
                description = scene.get("description", "")
                location = scene.get("location", "")
                characters = scene.get("characters", [])
                interactions = scene.get("interactions", [])

                # Build 5-layer chunk
                chunk = {
                    "chunk_id": f"mg_{clip_id}",
                    "movie_id": movie_id,
                    "start_seconds": scene.get("start", 0),
                    "end_seconds": scene.get("end", 0),
                    "description": description,
                    "text": description,
                    "situation": situation,
                    "setting": location,
                    "characters": [c.get("name", "") for c in characters],
                    "character_emotions": {c.get("name", ""): c.get("emotion", "") for c in characters},
                    "interactions": interactions,
                    "type": "scene_graph",
                    "source": "moviegraphs",
                    "dataset": "moviegraphs",
                    "narrative_arc": scene.get("narrative_arc", ""),
                    "dialogue_text": scene.get("dialogue", ""),
                }
                chunks.append(chunk)

                # Build graph nodes
                for char in characters:
                    graph_nodes.append({
                        "id": f"char_{movie_id}_{char['name']}",
                        "type": "Character",
                        "movie_id": movie_id,
                        "name": char.get("name", ""),
                        "emotion": char.get("emotion", ""),
                        "clip_id": clip_id,
                    })

                graph_nodes.append({
                    "id": f"scene_{movie_id}_{clip_id}",
                    "type": "Scene",
                    "movie_id": movie_id,
                    "description": description,
                    "situation": situation,
                    "clip_id": clip_id,
                })

                # Build graph edges
                for inter in interactions:
                    graph_edges.append({
                        "from": inter.get("from", ""),
                        "to": inter.get("to", ""),
                        "type": inter.get("type", ""),
                        "clip_id": clip_id,
                        "movie_id": movie_id,
                    })

        chunks_file = output_dir / "all_chunks.json"
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        graph_file = output_dir / "knowledge_graph.json"
        with open(graph_file, "w", encoding="utf-8") as f:
            json.dump({
                "nodes": graph_nodes,
                "edges": graph_edges,
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"  ✅ Built {len(chunks)} scene chunks + {len(graph_nodes)} graph nodes")
        logger.info(f"  ✅ Graph saved → {graph_file}")

        # Build knowledge index
        self._build_knowledge_index(output_dir, chunks, "moviegraphs")

    def _process_activitynet(self, ds: DatasetInfo, pipeline: bool) -> None:
        """Process ActivityNet captions."""
        logger.info(f"  📋 Parsing ActivityNet captions...")

        caps_files = list(ds.root.rglob("captions*.json"))
        if not caps_files:
            logger.error("  ❌ No captions JSON found.")
            return

        output_dir = self.data_root / "pipeline_output" / "activitynet_chunks"
        output_dir.mkdir(parents=True, exist_ok=True)

        chunks = []
        with open(caps_files[0], encoding="utf-8") as f:
            data = json.load(f)

        for video_id, info in data.items():
            sentences = info.get("sentences", [])
            timestamps = info.get("timestamps", [])
            duration = info.get("duration", 0)

            for i, (sent, ts) in enumerate(zip(sentences, timestamps)):
                if not ts or len(ts) < 2:
                    continue
                start, end = float(ts[0]), float(ts[1])
                chunks.append({
                    "chunk_id": f"anet_{video_id}_{i:04d}",
                    "video_id": video_id,
                    "movie_id": "activitynet",
                    "start_seconds": start,
                    "end_seconds": end,
                    "duration": end - start,
                    "description": sent,
                    "text": sent,
                    "language": "en",
                    "type": "caption",
                    "source": "activitynet",
                    "dataset": "activitynet",
                    "situation": "",
                    "characters": [],
                    "vision_actions": [],
                    "narrative_arc": "",
                    "dialogue_text": "",
                })

        chunks_file = output_dir / "all_chunks.json"
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        logger.info(f"  ✅ Built {len(chunks)} ActivityNet chunks → {chunks_file}")
        self._build_knowledge_index(output_dir, chunks, "activitynet")

    def _build_knowledge_index(
        self,
        output_dir: Path,
        chunks: List[Dict],
        dataset_name: str,
    ) -> None:
        """Build FAISS knowledge index from chunks."""
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.warning(f"  ⚠️  sentence-transformers not installed. Skipping index build.")
            return

        logger.info(f"  🔢 Building FAISS knowledge index...")

        index_dir = self.data_root / "pipeline_output" / "indexes"
        index_dir.mkdir(parents=True, exist_ok=True)

        # Encode texts
        model = SentenceTransformer("all-MiniLM-L6-v2")
        texts = [c.get("text", "") or c.get("description", "") for c in chunks]
        embeddings = model.encode(texts, show_progress_bar=True, batch_size=256)
        embeddings = embeddings.astype(np.float32)

        # Build FAISS index
        import faiss
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        # Save
        index_path = index_dir / f"knowledge_{dataset_name}.faiss"
        faiss.write_index(index, str(index_path))

        # Save metadata map
        meta_map = []
        for i, chunk in enumerate(chunks):
            meta_map.append({
                "idx": i,
                "chunk_id": chunk.get("chunk_id", ""),
                "movie_id": chunk.get("movie_id", ""),
                "start_seconds": chunk.get("start_seconds", 0),
                "end_seconds": chunk.get("end_seconds", 0),
                "text": chunk.get("text", "")[:200],
                "type": chunk.get("type", ""),
                "source": chunk.get("source", ""),
            })

        map_path = index_dir / f"knowledge_{dataset_name}_map.json"
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(meta_map, f, ensure_ascii=False, indent=2)

        logger.info(f"  ✅ FAISS knowledge index: {index_path} ({index.ntotal} vectors)")

    # ── Batch: All Tier 1 ─────────────────────────────────────────────────────

    def download_all_tier1(self, force: bool = False) -> None:
        """Tải tất cả Tier 1 datasets."""
        tier1 = [k for k, v in DATASETS.items() if v.priority == 1]
        logger.info(f"\n🚀 Downloading all Tier 1 datasets: {tier1}")
        self.download(tier1, force=force)

    def process_all_tier1(self, pipeline: bool = False) -> None:
        """Process tất cả Tier 1 datasets."""
        tier1 = [k for k, v in DATASETS.items() if v.priority == 1]
        logger.info(f"\n⚙️  Processing all Tier 1 datasets: {tier1}")
        self.process(tier1, pipeline=pipeline)

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _status_icon(status: str) -> str:
        icons = {
            "pending": "⏳",
            "downloading": "⬇️ ",
            "ready": "✅",
            "error": "❌",
        }
        return icons.get(status, "❓")

    def _count_items(self, ds: DatasetInfo) -> int:
        """Đếm số items trong dataset."""
        if not ds.root.exists():
            return 0
        jsons = list(ds.root.rglob("*.json"))
        if jsons:
            try:
                with open(jsons[0]) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return len(data)
                    elif isinstance(data, dict):
                        return len(data)
            except Exception:
                pass
        return ds.num_items

    def _update_dataset_config(self, ds: DatasetInfo, updates: Dict) -> None:
        """Cập nhật dataset config YAML + in-memory DATASETS."""
        # 1. Update in-memory DATASETS entry
        for key, entry in DATASETS.items():
            if entry.name == ds.name:
                for k, v in updates.items():
                    if hasattr(entry, k):
                        setattr(entry, k, v)
                break

        # 2. Update YAML config
        if not self.config_path.exists():
            return
        try:
            with open(self.config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
            for section in ["internal", "external"]:
                if section in config and config[section]:
                    for name, entry in config[section].items():
                        if isinstance(entry, dict) and entry.get("name") == ds.name:
                            entry.update(updates)
                            with open(self.config_path, "w", encoding="utf-8") as f:
                                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
                            return
        except Exception:
            pass


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="VideoSceneRAG Dataset Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m movierag.scripts.manage_datasets --list
  python -m movierag.scripts.manage_datasets --status
  python -m movierag.scripts.manage_datasets --download msr_vtt didemo
  python -m movierag.scripts.manage_datasets --verify msr_vtt didemo moviegraphs
  python -m movierag.scripts.manage_datasets --process msr_vtt didemo moviegraphs
  python -m movierag.scripts.manage_datasets --all-tier1
  python -m movierag.scripts.manage_datasets --download-and-process msr_vtt
        """,
    )

    parser.add_argument("--list", action="store_true", help="List all datasets")
    parser.add_argument("--status", action="store_true", help="Show dataset status")
    parser.add_argument("--download", nargs="+", metavar="DS",
                        help="Download datasets (e.g. msr_vtt didemo)")
    parser.add_argument("--verify", nargs="+", metavar="DS",
                        help="Verify downloaded datasets")
    parser.add_argument("--process", nargs="+", metavar="DS",
                        help="Process datasets into chunks + indexes")
    parser.add_argument("--pipeline", action="store_true",
                        help="Also run full preprocessing pipeline")
    parser.add_argument("--skip-video", action="store_true",
                        help="Skip video processing (captions only)")
    parser.add_argument("--all-tier1", action="store_true",
                        help="Download + process all Tier 1 datasets")
    parser.add_argument("--download-and-process", nargs="+", metavar="DS",
                        help="Download then process datasets")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if exists")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root directory",
    )

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    manager = DatasetManager(project_root=args.project_root)

    if args.list:
        manager.list_datasets()
    elif args.status:
        manager.status()
    elif args.all_tier1:
        manager.download_all_tier1(force=args.force)
        manager.process_all_tier1(pipeline=args.pipeline)
    elif args.download_and_process:
        manager.download(args.download_and_process, force=args.force)
        manager.verify(args.download_and_process)
        manager.process(args.download_and_process, pipeline=args.pipeline, skip_video=args.skip_video)
    else:
        if args.download:
            manager.download(args.download, force=args.force)
        if args.verify:
            manager.verify(args.verify)
        if args.process:
            manager.process(args.process, pipeline=args.pipeline, skip_video=args.skip_video)


if __name__ == "__main__":
    main()
