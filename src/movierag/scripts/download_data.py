"""
MovieRAG Dataset Downloader & Verifier

Since MovieNet and MovieGraphs require registration/forms to download,
this script sets up the directory structure and provides direct links/instructions
for the user to acquire the data manually.

Usage:
    python -m movierag.scripts.download_data
"""

import os
import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from movierag.config import get_config

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("MovieRAG")


def print_box(text: str):
    lines = text.strip().split("\n")
    width = max(len(line) for line in lines) + 4
    print("+" + "-" * width + "+")
    for line in lines:
        print(f"|  {line:<{width - 4}}  |")
    print("+" + "-" * width + "+")


def check_movienet(config):
    """Check and guide for MovieNet"""
    data_dir = config.paths.movienet_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check annotations
    has_meta = (data_dir / "meta" / "movie_info.json").exists()
    has_scene = (data_dir / "annotations" / "scene").exists()

    print(f"\n[1] Checking MovieNet at: {data_dir}...")

    if has_meta and has_scene:
        print("✅ MovieNet metadata found.")
    else:
        print("❌ MovieNet metadata MISSING.")
        print_box("""
ACTION REQUIRED: Download MovieNet Metadata
1. Go to: https://opendatalab.com/OpenDataLab/MovieNet
2. Login and Download:
   - 'annotation.v1.zip'
   - 'meta.v1.zip'
3. Extract them into:
   d:\\Study\\School\\project_ky4\\data\\movienet\\
   
   Structure should be:
   data/movienet/
   ├── items/ (meta.v1.zip content)
   └── annotations/ (annotation.v1.zip content)
        """)

    # Check Keyframes/Videos
    has_kf = (data_dir / "keyframes").exists()
    if has_kf:
        print("✅ MovieNet keyframes found.")
    else:
        print("❌ MovieNet keyframes MISSING.")
        print_box("""
ACTION REQUIRED: Download Movie Images/Videos
Option A: Download Keyframes (Recommended for Search)
   - If available on OpenDataLab, download 'keyframes.zip'
   - Extract to: data/movienet/keyframes/

Option B: Download Videos and Extract
   - Download 'movie_files.zip'
   - Extract to: data/movienet/videos/
   - Run: python -m movierag.scripts.extract_keyframes
        """)


def check_moviegraphs(config):
    """Check and guide for MovieGraphs"""
    data_dir = config.paths.moviegraphs_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    has_pkl = (data_dir / "all_movies.pkl").exists()

    print(f"\n[2] Checking MovieGraphs at: {data_dir}...")

    if has_pkl:
        print("✅ MovieGraphs data found.")
    else:
        print("❌ MovieGraphs data MISSING.")
        print_box("""
ACTION REQUIRED: Download MovieGraphs
1. Go to: http://moviegraphs.cs.toronto.edu/download.html
2. Fill the Google Form to get the link.
3. Download the data (graphs/annotations).
4. Save 'all_movies.pkl' to:
   d:\\Study\\School\\project_ky4\\data\\MovieGraphs_repo\\
        """)


def main():
    config = get_config()
    print("=" * 60)
    print("MovieRAG Dataset Setup")
    print("=" * 60)

    check_movienet(config)
    check_moviegraphs(config)

    print("\n" + "=" * 60)
    print("After downloading, run:")
    print("python -m movierag.scripts.build_index --data-dir data/movienet --full")


if __name__ == "__main__":
    main()
