"""
Run ONLY step 4b (Semantic Segmentation) and step 4c (FFmpeg Physical Clipping)
on an existing movie that already has annotation and subtitle files.

Usage:
    conda activate videorag; python run_segmentation.py tt0120338
"""
import sys
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

sys.path.insert(0, str(Path(__file__).parent))

from preprocess_data.config import PreprocessConfig as Cfg

# Point to the standard movie_output directory where Titanic data already lives
MOVIE_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "movie_output"
Cfg.OUTPUT_DIR = MOVIE_OUTPUT_DIR

from preprocess_data.extractors.semantic_scene_segmenter import SemanticSceneSegmenter
from preprocess_data.extractors.clip_extractor import ClipExtractor

def run(movie_id: str, force: bool = False):
    video_path = Cfg.RAW_VIDEOS_DIR / f"{movie_id}.mp4"
    if not video_path.exists():
        print(f"❌ Video not found at {video_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"  [Segmentation] {movie_id}")
    print(f"  Video: {video_path} ({video_path.stat().st_size / 1e9:.2f} GB)")
    print(f"{'='*60}")

    # --- EXECUTION TOGGLES ---
    run_llm_segmentation = True
    run_ffmpeg_clipping = False

    # Step 4b: Semantic Scene Segmentation
    if run_llm_segmentation:
        print("\n[4b] Semantic Scene Segmentation (LLM + Opti-Semantic Alignment)...")
        segmenter = SemanticSceneSegmenter()
        ok = segmenter.process_movie(movie_id)
        if not ok:
            print("  [WARN] Semantic segmentation skipped/failed — check annotation/subtitle files.")
        else:
            print("  [OK] Semantic segmentation complete!")
    else:
        print("\n[4b] Semantic Scene Segmentation skipped by toggle.")

    # Step 4c: Physical FFmpeg clipping
    if run_ffmpeg_clipping:
        print("\n[4c] Physical Video Clipping (FFmpeg)...")
        extractor = ClipExtractor()
        ok_extract = extractor.extract_clips(movie_id, video_path)
        if not ok_extract:
            print("  [WARN] FFmpeg clipping failed or skipped.")
        else:
            print("  [OK] Physical clipping complete!")
    else:
        print("\n[4c] Physical Video Clipping skipped by toggle.")

if __name__ == "__main__":
    mid = sys.argv[1] if len(sys.argv) > 1 else "tt0120338"
    force_flag = "--force" in sys.argv
    run(mid, force=force_flag)
