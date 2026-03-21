#!/bin/bash
# ╔══════════════════════════════════════════════════════════╗
# ║  MSR-VTT Caption Acquisition Guide                   ║
# ║  Giải thích cách lấy 10,000 captions thực cho       ║
# ║  MSR-VTT khi có/không có video files.                ║
# ╚══════════════════════════════════════════════════════════╝

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MSR_DIR="$PROJECT_ROOT/data/msr_vtt"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  MSR-VTT Caption Acquisition Guide                   ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# ── Option A: Google Drive (recommended for full dataset) ──────────────────
echo -e "${GREEN}[Option A] Google Drive — Full 10K Captions${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1. Truy cập: https://drive.google.com/drive/folders/0B4cvsEOB5eUCckFvOU8zb0RVWTg"
echo "2. Đăng nhập Google account"
echo "3. Tìm file: MSR_VTT_All.json hoặc annotation.zip"
echo "4. Download về: $MSR_DIR/MSR_VTT_All.json"
echo ""
echo "Sau khi download, chạy:"
echo "  python scripts/enrich_chunks.py"
echo ""
echo ""

# ── Option B: HuggingFace datasets (if network allows) ─────────────────────
echo -e "${GREEN}[Option B] HuggingFace datasets API${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Nếu máy có thể truy cập HuggingFace:"
echo ""
echo "  pip install datasets"
echo "  python3 -c \""
echo "    from datasets import load_dataset"
echo "    ds = load_dataset('CharmingDog/msrvtt', split='train')"
echo "    import json"
echo "    with open('$MSR_DIR/MSR_VTT_All.json', 'w') as f:"
echo "        json.dump([dict(x) for x in ds], f, indent=2)"
echo '  "'
echo ""
echo "Hoặc:"
echo "  huggingface-cli download CharmingDog/msrvtt --repo-type dataset \\"
echo "    --local-dir $MSR_DIR/hf_cache"
echo ""
echo ""

# ── Option C: If you have videos, extract captions ──────────────────────────
echo -e "${GREEN}[Option C] Extract captions from downloaded videos${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Nếu đã có video files:"
echo ""
echo "  # Extract auto-subtitles with yt-dlp"
echo "  mkdir -p $MSR_DIR/subtitles"
echo "  yt-dlp --write-auto-subs --sub-lang en \\"
echo "    --skip-download --convert-subs srt \\"
echo "    -P $MSR_DIR/subtitles \\"
echo "    -a $MSR_DIR/videourlist.txt"
echo ""
echo "  # Or use Whisper if you have video files"
echo "  python3 -c \""
echo "    import whisper, glob, json"
echo "    model = whisper.load_model('base')"
echo "    chunks = []"
echo "    for video in glob.glob('$MSR_DIR/RawVideoAll/*.mp4'):"
echo "        result = model.transcribe(video)"
echo "        for seg in result['segments']:"
echo "            chunks.append({...})"
echo "    with open('$MSR_DIR/MSR_VTT_All.json', 'w') as f:"
echo "        json.dump(chunks, f)"
echo '  "'
echo ""

# ── Current status ──────────────────────────────────────────────────────────
echo -e "${YELLOW}[Current Status]${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 << 'PYEOF'
import json, pathlib
ROOT = pathlib.Path("/home/hiwe/project/DAP303m/project_ky4/data")
msr_dir = ROOT / "msr_vtt"

# Check current caption quality
ann = msr_dir / "MSR_VTT_All.json"
if ann.exists():
    with open(ann) as f:
        data = json.load(f)
    generic = {
        "a group of people are talking and interacting",
        "someone is explaining something to the camera",
        "a scene from a movie with dialogue and action",
        "a video showing people in various settings",
        "a video showing people in various indoor settings",
        "a clip from a television show or movie",
        "someone is performing an action in front of the camera",
        "a short video clip showing a conversation or interaction",
        "a scene with music and visual effects",
        "a video clip showing text and imagery",
        "a short segment of a documentary or instructional video",
    }
    real = [x for x in data if x.get("caption","") not in generic]
    gen  = [x for x in data if x.get("caption","") in generic]
    print(f"  Total chunks:     {len(data):,}")
    print(f"  Real captions:   {len(real):,} ({len(real)/len(data)*100:.1f}%)")
    print(f"  Placeholders:   {len(gen):,} ({len(gen)/len(data)*100:.1f}%)")
    if real:
        print(f"\n  Sample real captions:")
        for x in real[:3]:
            print(f"    [{x['video_id']}] {x['caption'][:60]}")
else:
    print("  MSR_VTT_All.json not found!")
PYEOF

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Next Steps After Getting Real Captions              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  1. Replace $MSR_DIR/MSR_VTT_All.json with real captions"
echo "  2. Re-run enrichment:"
echo "     python scripts/enrich_chunks.py"
echo "  3. Verify:"
echo "     python -m movierag.scripts.verify_pipeline_output --check-indexes"
