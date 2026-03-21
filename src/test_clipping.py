import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocess_data.config import PreprocessConfig as Cfg
from preprocess_data.extractors.clip_extractor import ClipExtractor

def main():
    movie_id = "tt1193138_00-03-27_12s"
    video_path = Path(r"D:\Study\School\project_ky4\src\data\temp_clips\tt1193138_00-03-27_12s.mp4")
    
    # Override output directory logic for tests
    Cfg.OUTPUT_DIR = Path(r"D:\Study\School\project_ky4\data\pipeline_refactor_test")
    
    # 1. Create a dummy annotation file
    ann_dir = Cfg.get_annotation_dir()
    ann_dir.mkdir(parents=True, exist_ok=True)
    
    ann_file = ann_dir / f"{movie_id}.json"
    
    # Creating a synthetic scene structure: split the 12s video into three 4-second scenes
    dummy_data = {
        "movie_id": movie_id,
        "scenes": [
            {
                "id": "scene_0",
                "start_seconds": 0.0,
                "end_seconds": 4.0
            },
            {
                "id": "scene_1",
                "start_seconds": 4.0,
                "end_seconds": 8.0
            },
            {
                "id": "scene_2",
                "start_seconds": 8.0,
                "end_seconds": 12.0
            }
        ]
    }
    
    import json
    with open(ann_file, "w", encoding="utf-8") as f:
        json.dump(dummy_data, f, indent=2)
        
    print(f"Created dummy annotation at {ann_file}")
    
    # 2. Extract clips
    print("Starting clip extraction...")
    extractor = ClipExtractor()
    success = extractor.extract_clips(movie_id, video_path, force=True)
    
    if success:
        print("✅ Extraction successful!")
        clips_dir = Cfg.OUTPUT_DIR / movie_id / "clips"
        clips = list(clips_dir.glob("*.mp4"))
        print(f"Verified {len(clips)} clips in {clips_dir}:")
        for c in clips:
            print(f" - {c.name}")
    else:
        print("❌ Extraction failed.")

if __name__ == "__main__":
    main()
