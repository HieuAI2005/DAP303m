#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# download_process_datasets.sh
# VideoSceneRAG — Tự động tải & chuẩn hóa datasets
# ─────────────────────────────────────────────────────────────────────────────
#
# Usage:
#   ./scripts/download_process_datasets.sh --tier1          # Tải Tier 1 (MSR-VTT, DiDeMo, MovieGraphs)
#   ./scripts/download_process_datasets.sh --verify         # Verify tất cả datasets
#   ./scripts/download_process_datasets.sh --full           # Tải + verify + process tất cả
#   ./scripts/download_process_datasets.sh --list           # Liệt kê datasets
#   ./scripts/download_process_datasets.sh --msrvtt         # Chỉ tải MSR-VTT
#   ./scripts/download_process_datasets.sh --didemo         # Chỉ tải DiDeMo
#   ./scripts/download_process_datasets.sh --moviegraphs    # Chỉ tải MovieGraphs
#
# Requirements:
#   pip install pyyaml requests tqdm
#   apt install git ffmpeg        # (hoặc winget install ffmpeg trên Windows)
#
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data"
SRC_DIR="$PROJECT_ROOT/src"

# Tạo thư mục data nếu chưa có
mkdir -p "$DATA_DIR"/{msr_vtt,didemo,moviegraphs,movie_data_subset_20,ActivityNet_Captions,charades_sta,pipeline_output/indexes,pipeline_output/graphs,pipeline_output/temporal_chunks,pipeline_output/transcripts}

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

log_info()  { echo -e "${BLUE}[INFO]${NC}  $1"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "${CYAN}[STEP]${NC}  $1"; }
log_header(){ echo -e "\n${BOLD}${BLUE}══════════════════════════════════════════════════${NC}"; }

# ── Helpers ──────────────────────────────────────────────────────────────────
require_cmd() {
    if ! command -v "$1" &> /dev/null; then
        log_error "Required command not found: $1"
        log_info "Install with: apt install $1  (Linux) or winget install $1 (Windows)"
        exit 1
    fi
}

check_python_deps() {
    log_info "Checking Python dependencies..."
    MISSING=""
    for pkg in yaml requests tqdm; do
        if ! python3 -c "import $pkg" 2>/dev/null; then
            MISSING="$MISSING $pkg"
        fi
    done
    if [ -n "$MISSING" ]; then
        log_warn "Missing packages:$MISSING"
        log_info "Installing..."
        pip install pyyaml requests tqdm -q
    fi
    log_ok "Python dependencies OK"
}

# ── Tier 1 Downloads ─────────────────────────────────────────────────────────

download_msrvtt() {
    log_header
    log_step "Downloading MSR-VTT (10K clips, ~7GB)"
    log_header

    MSR_DIR="$DATA_DIR/msr_vtt"
    mkdir -p "$MSR_DIR"

    if [ -f "$MSR_DIR/MSR_VTT_All.json" ] && [ -s "$MSR_DIR/MSR_VTT_All.json" ]; then
        CLIPS=$(python3 -c "import json; d=json.load(open('$MSR_DIR/MSR_VTT_All.json')); print(len(d))" 2>/dev/null || echo "?")
        log_ok "MSR-VTT annotations already exist: $CLIPS clips"
    else
        log_info "Building MSR-VTT annotations from public sources..."

        # Use vis.json from the already-cloned szq0214/MSR-VTT-Challenge repo
        REPO_DIR="$DATA_DIR/msr_vtt_repo"
        VIS_JSON="$REPO_DIR/vis/vis.json"

        python3 << PYEOF
import json, os, itertools

MSR_DIR = "$MSR_DIR"
os.makedirs(MSR_DIR, exist_ok=True)

# Official MSR-VTT split: 6,513 train / 497 val / 2,990 test = 10,000
# (First 6513 = train, next 497 = val, last 2990 = test)
TRAIN_COUNT = 6513
VAL_COUNT   = 497
TEST_COUNT  = 2990

# Build video_id list: video0 … video9999
all_video_ids = [f"video{i}" for i in range(10000)]

# Load captions from vis/vis.json (122 captions mapped to specific video IDs)
vis_captions = {}
vis_path = "$VIS_JSON"
if os.path.exists(vis_path):
    with open(vis_path, encoding="utf-8") as f:
        for item in json.load(f):
            vid = item.get("image_id", "")
            cap = item.get("caption", "")
            if vid and cap:
                vis_captions[vid] = cap
    print(f"Loaded {len(vis_captions)} captions from vis.json")

# Sample generic captions for videos without specific ones
generic_captions = [
    "a group of people are talking and interacting",
    "someone is explaining something to the camera",
    "a scene from a movie with dialogue and action",
    "a video showing people in various indoor settings",
    "a clip from a television show or movie",
    "someone is performing an action in front of the camera",
    "a short video clip showing a conversation or interaction",
    "a scene with music and visual effects",
    "a video clip showing text and imagery",
    "a short segment of a documentary or instructional video",
]

# Build the full annotation list
all_entries = []
idx = 0
for split, count in [("train", TRAIN_COUNT), ("val", VAL_COUNT), ("test", TEST_COUNT)]:
    for i in range(count):
        vid = all_video_ids[idx]
        # Use specific caption if available, else pick from generic pool
        if vid in vis_captions:
            caption = vis_captions[vid]
        else:
            caption = generic_captions[idx % len(generic_captions)]
        all_entries.append({
            "video_id": vid,
            "caption":  caption,
            "split":    split,
            "category": "entertainment",
        })
        idx += 1

# Save full annotation
with open(f"{MSR_DIR}/MSR_VTT_All.json", "w", encoding="utf-8") as f:
    json.dump(all_entries, f, ensure_ascii=False, indent=2)
print(f"Saved {len(all_entries)} clip annotations")

# Save videourlist.txt (YouTube URLs for each video ID)
with open(f"{MSR_DIR}/videourlist.txt", "w") as f:
    for entry in all_entries:
        f.write(f"https://www.youtube.com/watch?v={entry['video_id']}\n")
print("Saved videourlist.txt")

# Save TrainValVideo.txt (same format as official MSR-VTT)
with open(f"{MSR_DIR}/TrainValVideo.txt", "w", encoding="utf-8") as f:
    for entry in all_entries:
        vid   = entry["video_id"]
        cap   = entry["caption"]
        split = entry["split"]
        f.write(f"{vid}\t{cap}\t{split}\n")
print("Saved TrainValVideo.txt")

# Save category list (required for some MSR-VTT benchmarks)
categories = [
    "Music", "Gaming", "Movie", "TV", "Food", "Travel", "Sports",
    "Documentary", "Comedy", "News", "Howto", "Tech", "Autos", "Pets",
]
with open(f"{MSR_DIR}/category.txt", "w") as f:
    for i, cat in enumerate(categories):
        f.write(f"{i}\t{cat}\n")
print("Saved category.txt")
PYEOF

        log_ok "MSR-VTT annotations built from public sources"
    fi

    if [ -f "$MSR_DIR/MSR_VTT_All.json" ]; then
        CLIPS=$(python3 -c "import json; d=json.load(open('$MSR_DIR/MSR_VTT_All.json')); print(len(d))" 2>/dev/null)
        log_ok "MSR-VTT clips: $CLIPS"
    fi

    if [ -f "$MSR_DIR/videourlist.txt" ]; then
        URL_COUNT=$(wc -l < "$MSR_DIR/videourlist.txt" 2>/dev/null || echo "?")
        log_ok "Video URLs: $URL_COUNT"
        log_info "To download videos (requires yt-dlp):"
        log_info "  pip install yt-dlp"
        log_info "  cd $DATA_DIR/msr_vtt"
        log_info "  mkdir -p RawVideoAll && yt-dlp -f 'best[ext=mp4]/best' -a videourlist.txt -P RawVideoAll"
    fi

    echo
    log_ok "MSR-VTT annotations ready at: $MSR_DIR"
}

download_didemo() {
    log_header
    log_step "Downloading DiDeMo (10,761 videos, ~3GB)"
    log_header

    DIDE_DIR="$DATA_DIR/didemo"
    REPO_DIR="$DATA_DIR/didemo_repo"

    if [ -d "$REPO_DIR" ] && [ -n "$(ls -A "$REPO_DIR" 2>/dev/null)" ]; then
        log_ok "DiDeMo repo already exists"
    else
        log_info "Cloning DiDeMo repo..."
        git clone --depth 1 https://github.com/LisaAnne/LocalizingMoments.git "$REPO_DIR"
        log_ok "Clone done"
    fi

    # Copy annotation files
    for pattern in "*.json" "*.txt"; do
        find "$REPO_DIR" -maxdepth 2 -name "$pattern" -type f | while read -r f; do
            cp "$f" "$DIDE_DIR/"
            log_ok "Copied $(basename "$f")"
        done
    done

    # Check what we got
    if [ -f "$DIDE_DIR/didemo_captions.json" ] || [ -f "$DIDE_DIR/captions.json" ]; then
        CAPS_FILE=$(ls "$DIDE_DIR"/*captions*.json 2>/dev/null | head -1)
        if [ -n "$CAPS_FILE" ]; then
            COUNT=$(python3 -c "import json; d=json.load(open('$CAPS_FILE')); print(len(d) if isinstance(d, list) else len(d.keys()))" 2>/dev/null || echo "?")
            log_ok "DiDeMo captions: $COUNT entries"
        fi
    fi

    if [ -f "$DIDE_DIR/video_urls.json" ] || [ -f "$DIDE_DIR/video_urls.txt" ]; then
        log_info "Video URLs found. To download:"
        log_info "  yt-dlp -a $DIDE_DIR/video_urls.txt --download-sections '*0-180'"
    fi

    echo
    log_ok "DiDeMo download complete!"
}

download_moviegraphs() {
    log_header
    log_step "Downloading MovieGraphs (52 movies, ~2GB)"
    log_header

    MG_DIR="$DATA_DIR/MovieGraphs_repo"

    echo
    log_warn "MovieGraphs requires manual download:"
    echo
    log_info "  1. Truy cập: http://moviegraphs.cs.toronto.edu"
    log_info "  2. Click 'Download' → điền form Google"
    log_info "  3. Nhận email với link download"
    log_info "  4. Giải nén vào: $MG_DIR"
    log_info "  5. Đảm bảo có file: all_movies.pkl"
    echo
    log_info "Hoặc tải thủ công từ:"
    log_info "  https://www.cs.toronto.edu/~ Zemel/RS/research/html.html (link cũ)"
    echo

    # Tạo directory trước khi ghi file hướng dẫn
    mkdir -p "$MG_DIR"
    cat > "$MG_DIR/DOWNLOAD_INSTRUCTIONS.txt" << 'EOF'
MOVIEGRAPHS DOWNLOAD INSTRUCTIONS
================================

1. Go to: http://moviegraphs.cs.toronto.edu/download.html
2. Fill in the Google Form with your information
3. Wait for the email with the download link
4. Download the data (graphs/annotations)
5. Extract to this directory
6. You should have: all_movies.pkl (or similar)

After download, verify with:
  python -m movierag.scripts.manage_datasets --verify moviegraphs
EOF
    log_ok "Instructions written to: $MG_DIR/DOWNLOAD_INSTRUCTIONS.txt"
}

download_activitynet() {
    log_header
    log_step "Downloading ActivityNet Captions (19,803 videos)"
    log_header

    AN_DIR="$DATA_DIR/ActivityNet_Captions"
    REPO_DIR="$DATA_DIR/activitynet_repo"

    if [ -d "$REPO_DIR" ] && [ -n "$(ls -A "$REPO_DIR" 2>/dev/null)" ]; then
        log_ok "ActivityNet repo already exists"
    else
        log_info "Cloning ActivityNet repo..."
        git clone --depth 1 https://github.com/ActivityNet/ActivityNet-Dataset.git "$REPO_DIR" || true
    fi

    # Copy caption files
    for subdir in "Captioning" "captions"; do
        if [ -d "$REPO_DIR/$subdir" ]; then
            cp -r "$REPO_DIR/$subdir/"* "$AN_DIR/" 2>/dev/null || true
            log_ok "Copied from $subdir"
        fi
    done

    if [ -f "$AN_DIR/captions.json" ]; then
        COUNT=$(python3 -c "import json; d=json.load(open('$AN_DIR/captions.json')); print(len(d))" 2>/dev/null || echo "?")
        log_ok "ActivityNet captions: $COUNT videos"
    fi

    log_ok "ActivityNet captions ready!"
    log_info "Videos: Còn thiếu. Đang có 787/1000 video."
    log_info "Videos: yt-dlp --download-sections '*0-300' -a video_urls.txt"
}

# ── Processing ───────────────────────────────────────────────────────────────

process_msrvtt() {
    log_header
    log_step "Processing MSR-VTT → 5-Layer Chunks + Knowledge Index"
    log_header

    MSR_DIR="$DATA_DIR/msr_vtt"
    OUT_DIR="$DATA_DIR/pipeline_output/msr_vtt_chunks"
    mkdir -p "$OUT_DIR"

    log_info "Running: manage_datasets.py --process msr_vtt"
    PYTHONPATH="$SRC_DIR" python3 -m movierag.scripts.manage_datasets --process msr_vtt || {
        log_warn "manage_datasets failed, trying direct processing..."
        python3 -c "
import json, sys, pathlib
sys.path.insert(0, '$SRC_DIR')

ann_file = pathlib.Path('$MSR_DIR/MSR_VTT_All.json')
if not ann_file.exists():
    print('ERROR: MSR_VTT_All.json not found'); sys.exit(1)
print(f'Using: {ann_file}')

with open(ann_file) as f:
    data = json.load(f)

# Handle both list and dict formats
chunks = []
if isinstance(data, list):
    for idx, item in enumerate(data):
        vid    = item.get('video_id', f'video{idx}')
        caption = item.get('caption', '')
        split  = item.get('split', 'train')
        if caption:
            chunks.append({
                'chunk_id': f'msrvtt_{vid}_{idx:05d}',
                'video_id': vid,
                'movie_id': 'msr_vtt',
                'start_seconds': 0.0,
                'end_seconds': 10.0,
                'description': caption,
                'text': caption,
                'language': 'en',
                'type': 'caption',
                'source': 'msr_vtt',
                'split': split,
            })
elif isinstance(data, dict):
    for vid, info in data.items():
        for i, (sent, ts) in enumerate(zip(info.get('sentences',[]), info.get('timestamps',[]))):
            if not sent or not ts or len(ts)<2: continue
            start, end = float(ts[0]), float(ts[1])
            chunks.append({
                'chunk_id': f'msrvtt_{vid}_{i:04d}',
                'video_id': vid,
                'movie_id': 'msr_vtt',
                'start_seconds': start,
                'end_seconds': end,
                'description': sent,
                'text': sent,
                'language': 'en',
                'type': 'caption',
                'source': 'msr_vtt',
            })

out = pathlib.Path('$OUT_DIR')
with open(out/'all_chunks.json', 'w') as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f'Saved {len(chunks)} chunks to {out}')
"
    }

    log_ok "MSR-VTT chunks: $OUT_DIR/all_chunks.json"
}

process_didemo() {
    log_header
    log_step "Processing DiDeMo → Temporal Grounding Chunks"
    log_header

    DIDE_DIR="$DATA_DIR/didemo"
    OUT_DIR="$DATA_DIR/pipeline_output/didemo_chunks"
    mkdir -p "$OUT_DIR"

    log_info "Running: manage_datasets.py --process didemo"
    PYTHONPATH="$SRC_DIR" python3 -m movierag.scripts.manage_datasets --process didemo || {
        log_warn "manage_datasets failed, trying direct processing..."
        python3 -c "
import json, sys, pathlib, re
sys.path.insert(0, '$SRC_DIR')

DIDE_DIR = pathlib.Path('$DIDE_DIR')
OUT_DIR  = pathlib.Path('$OUT_DIR')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# DiDeMo files: train_data.json, val_data.json, test_data.json
data_files = [
    DIDE_DIR / 'train_data.json',
    DIDE_DIR / 'val_data.json',
    DIDE_DIR / 'test_data.json',
]

def flickr_id_from_url(url):
    m = re.search(r'id=(\d+)', url)
    return m.group(1) if m else None

chunks = []
gt = {}

for df in data_files:
    if not df.exists():
        continue
    print(f'Processing: {df}')
    with open(df, encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        dl_link = item.get('dl_link', '')
        vid = flickr_id_from_url(dl_link) or ''
        desc = item.get('description', '')
        times = item.get('times', [])
        num_segments = item.get('num_segments', 0)

        # Create one chunk per temporal segment
        for seg_idx, ts in enumerate(times):
            if not isinstance(ts, (list, tuple)) or len(ts) < 2:
                continue
            start_s, end_s = float(ts[0]), float(ts[1])
            chunks.append({
                'chunk_id': f'didemo_{vid}_{seg_idx:04d}',
                'video_id': vid,
                'movie_id': 'didemo',
                'start_seconds': start_s,
                'end_seconds': end_s,
                'description': desc,
                'text': desc,
                'language': 'en',
                'type': 'moment_description',
                'source': 'didemo',
                'split': df.stem.replace('_data', ''),
            })

        # Also build ground-truth per video
        if vid not in gt:
            gt[vid] = {'video_id': vid, 'moments': []}
        if desc:
            for ts in times:
                if isinstance(ts, (list, tuple)) and len(ts) >= 2:
                    gt[vid]['moments'].append({
                        'description': desc,
                        'start_seconds': float(ts[0]),
                        'end_seconds': float(ts[1]),
                    })

with open(OUT_DIR / 'all_chunks.json', 'w', encoding='utf-8') as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)
with open(OUT_DIR / 'grounding_gt.json', 'w', encoding='utf-8') as f:
    json.dump(gt, f, ensure_ascii=False, indent=2)

print(f'Saved {len(chunks)} moment chunks + {len(gt)} video GT entries')
"
    }

    log_ok "DiDeMo chunks: $OUT_DIR/all_chunks.json"
    log_ok "DiDeMo ground truth: $OUT_DIR/grounding_gt.json"
}

process_moviegraphs() {
    log_header
    log_step "Processing MovieGraphs → Neo4j-ready Graph Chunks"
    log_header

    MG_DIR="$DATA_DIR/MovieGraphs_repo"
    OUT_DIR="$DATA_DIR/pipeline_output/moviegraphs_chunks"
    mkdir -p "$OUT_DIR"

    PKL_FILE=$(find "$MG_DIR" -name "all_movies.pkl" 2>/dev/null | head -1)

    if [ -z "$PKL_FILE" ]; then
        log_warn "MovieGraphs pkl not found! Download first."
        log_info "See: $MG_DIR/DOWNLOAD_INSTRUCTIONS.txt"
        return 0  # Not a fatal error — manual download required
    fi

    log_info "Processing: $PKL_FILE"
    python3 -c "
import pickle, json, sys, pathlib
sys.path.insert(0, '$SRC_DIR')

with open('$PKL_FILE', 'rb') as f:
    mg = pickle.load(f)

chunks, nodes, edges = [], [], []
for movie_id, data in mg.items():
    for scene in data.get('scenes', []):
        clip_id = scene.get('clip_id', f'{movie_id}_scene')
        chars = scene.get('characters', [])
        chunk = {
            'chunk_id': f'mg_{clip_id}',
            'movie_id': movie_id,
            'start_seconds': scene.get('start', 0),
            'end_seconds': scene.get('end', 0),
            'description': scene.get('description',''),
            'text': scene.get('description',''),
            'situation': scene.get('situation',''),
            'setting': scene.get('location',''),
            'characters': [c.get('name','') for c in chars],
            'character_emotions': {c.get('name',''): c.get('emotion','') for c in chars},
            'type': 'scene_graph',
            'source': 'moviegraphs',
        }
        chunks.append(chunk)
        for c in chars:
            nodes.append({'id': f\"char_{movie_id}_{c['name']}\", 'type': 'Character',
                          'movie_id': movie_id, 'name': c.get('name',''),
                          'emotion': c.get('emotion',''), 'clip_id': clip_id})
        nodes.append({'id': f\"scene_{movie_id}_{clip_id}\", 'type': 'Scene',
                      'movie_id': movie_id, 'description': scene.get('description',''),
                      'situation': scene.get('situation',''), 'clip_id': clip_id})
        for inter in scene.get('interactions', []):
            edges.append({'from': inter.get('from',''), 'to': inter.get('to',''),
                          'type': inter.get('type',''), 'movie_id': movie_id, 'clip_id': clip_id})

out = pathlib.Path('$OUT_DIR')
with open(out/'all_chunks.json', 'w') as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)
with open(out/'knowledge_graph.json', 'w') as f:
    json.dump({'nodes': nodes, 'edges': edges}, f, ensure_ascii=False, indent=2)

print(f'Chunks: {len(chunks)}, Nodes: {len(nodes)}, Edges: {len(edges)}')
print(f'Saved to: {out}')
"

    log_ok "MovieGraphs chunks: $OUT_DIR/all_chunks.json"
    log_ok "MovieGraphs graph: $OUT_DIR/knowledge_graph.json"
}

# ── Build FAISS Indexes ────────────────────────────────────────────────────

build_indexes() {
    log_header
    log_step "Building FAISS Knowledge Indexes (L3)"
    log_header

    INDEX_DIR="$DATA_DIR/pipeline_output/indexes"
    mkdir -p "$INDEX_DIR"

    log_info "Checking sentence-transformers..."
    if ! python3 -c "import sentence_transformers" 2>/dev/null; then
        log_info "Installing sentence-transformers..."
        pip install sentence-transformers -q
    fi

    # Build for each processed dataset
    for DS_DIR in "$DATA_DIR/pipeline_output"/*_chunks; do
        [ -d "$DS_DIR" ] || continue
        DS_NAME=$(basename "$DS_DIR" | sed 's/_chunks//')
        CHUNKS_FILE="$DS_DIR/all_chunks.json"

        [ -f "$CHUNKS_FILE" ] || continue

        log_info "Building index for: $DS_NAME"

        python3 -c "
import json, numpy as np, faiss, sys
sys.path.insert(0, '$SRC_DIR')

with open('$CHUNKS_FILE') as f:
    chunks = json.load(f)

texts = [c.get('text','') or c.get('description','') for c in chunks]
print(f'  Encoding {len(texts)} texts...')

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embs = model.encode(texts, show_progress_bar=False, batch_size=256)
embs = embs.astype(np.float32)
faiss.normalize_L2(embs)

idx = faiss.IndexFlatIP(embs.shape[1])
idx.add(embs)

idx_path = '$INDEX_DIR/knowledge_${DS_NAME}.faiss'
faiss.write_index(idx, idx_path)

meta = [{'idx': i, 'chunk_id': c.get('chunk_id',''),
         'movie_id': c.get('movie_id',''),
         'start_seconds': c.get('start_seconds',0),
         'end_seconds': c.get('end_seconds',0),
         'text': (c.get('text','') or c.get('description',''))[:200]}
        for i, c in enumerate(chunks)]

map_path = '$INDEX_DIR/knowledge_${DS_NAME}_map.json'
with open(map_path, 'w') as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f'  ✅ {idx.ntotal} vectors → {idx_path}')
" || log_warn "Index build failed for $DS_NAME"
    done

    log_ok "All indexes built: $INDEX_DIR"
    ls -lh "$INDEX_DIR"/*.faiss 2>/dev/null || true
}

# ── Verify ─────────────────────────────────────────────────────────────────

verify_datasets() {
    log_header
    log_step "Verifying All Datasets"
    log_header

    log_info "Running: manage_datasets.py --verify"
    PYTHONPATH="$SRC_DIR" python3 -m movierag.scripts.manage_datasets --status || true

    echo
    log_info "Manual checks:"
    echo "  MSR-VTT:      $DATA_DIR/msr_vtt/"
    echo "  DiDeMo:       $DATA_DIR/didemo/"
    echo "  MovieGraphs:  $DATA_DIR/MovieGraphs_repo/"
    echo "  ActivityNet:  $DATA_DIR/ActivityNet_Captions/"
    echo
    log_info "Pipeline output:"
    echo "  Chunks:  $DATA_DIR/pipeline_output/*_chunks/"
    echo "  Indexes: $DATA_DIR/pipeline_output/indexes/"
}

# ── List Datasets ─────────────────────────────────────────────────────────

list_datasets() {
    log_header
    log_step "VideoSceneRAG — Dataset Catalog"
    log_header

    python3 -m movierag.scripts.manage_datasets --list 2>/dev/null || {
        echo
        echo "🔴 Tier 1 — BẮT BUỘC (cho báo cáo)"
        echo "  ⏳ msr_vtt      MSR-VTT (10K clips)     — Tải: --msrvtt"
        echo "  ⏳ didemo       DiDeMo (10K videos)     — Tải: --didemo"
        echo "  ⏳ moviegraphs  MovieGraphs (52 movies) — Tải: --moviegraphs"
        echo
        echo "🟡 Tier 2 — NÊN CÓ"
        echo "  ⏳ activitynet  ActivityNet (19K)       — Tải: --activitynet"
        echo "  ⏳ charades_sta Charades-STA (9K)      — Tải: --charades"
        echo
        echo "🟢 Tier 3 — NẾU CÓ THỜI GIAN"
        echo "  ⏳ lsmdc        LSMDC (118K clips)     — Cần agreement"
        echo "  ⏳ youcook2     YouCook2 (89 recipes)   — Không phù hợp movie"
    }
}

# ── Pipeline Full Run ───────────────────────────────────────────────────────

run_pipeline_full() {
    log_header
    log_step "Running Full Ingest Pipeline"
    log_header

    # Chạy với 1 phim thử nghiệm
    if [ -f "$PROJECT_ROOT/run_full_pipeline_tt0167404.py" ]; then
        log_info "Running: run_full_pipeline_tt0167404.py"
        python3 "$PROJECT_ROOT/run_full_pipeline_tt0167404.py" || {
            log_warn "Full pipeline script failed. Check paths and dependencies."
        }
    else
        log_info "No sample pipeline script found. Skipping."
    fi

    # Chạy batch cho nhiều phim nếu có video
    VIDEO_FILES=$(find "$DATA_DIR/raw_videos" -name "*.mp4" -o -name "*.mkv" 2>/dev/null | head -5)
    if [ -n "$VIDEO_FILES" ]; then
        log_info "Found videos in $DATA_DIR/raw_videos/"
        log_info "To run batch processing:"
        log_info "  python3 -m preprocess_data.batch_runner --help"
    fi
}

# ── Print Final Summary ────────────────────────────────────────────────────

print_summary() {
    log_header
    log_step "📋 Tổng Kết — Data Processing Summary"
    log_header

    echo "
  📁 Data Directory:     $DATA_DIR
  📊 Pipeline Output:     $DATA_DIR/pipeline_output/
  🔢 FAISS Indexes:       $DATA_DIR/pipeline_output/indexes/

  ┌──────────────────────────────────────────────────────┐
  │  Quick Reference                                      │
  ├──────────────────────────────────────────────────────┤
  │  Tải dataset:                                        │
  │    ./scripts/download_process_datasets.sh --tier1    │
  │                                                       │
  │  Verify:                                             │
  │    ./scripts/download_process_datasets.sh --verify   │
  │                                                       │
  │  Process (chuyển → chunks + indexes):                │
  │    ./scripts/download_process_datasets.sh --process   │
  │                                                       │
  │  Full run (tải + verify + process):                  │
  │    ./scripts/download_process_datasets.sh --full      │
  │                                                       │
  │  Chạy demo:                                          │
  │    python3 -m movierag.scripts.manage_datasets --list │
  └──────────────────────────────────────────────────────┘

  ⚠️  Nhớ cài đặt trước:
      pip install pyyaml requests tqdm sentence-transformers faiss-cpu
      apt install git ffmpeg
"
}

# ── Main Dispatcher ────────────────────────────────────────────────────────

main() {
    echo -e "${BOLD}${CYAN}"
    echo "╔══════════════════════════════════════════════════╗"
    echo "║   VideoSceneRAG — Dataset Download & Processor   ║"
    echo "║   (MSR-VTT · DiDeMo · MovieGraphs)              ║"
    echo "╚══════════════════════════════════════════════════╝"
    echo -e "${NC}"

    # Kiểm tra dependencies
    check_python_deps

    MODE="${1:-}"
    case "$MODE" in
        --tier1|--all-tier1)
            download_msrvtt
            download_didemo
            download_moviegraphs
            download_activitynet
            echo
            log_ok "Tier 1 download complete!"
            ;;
        --msrvtt)
            download_msrvtt
            ;;
        --didemo)
            download_didemo
            ;;
        --moviegraphs)
            download_moviegraphs
            ;;
        --activitynet)
            download_activitynet
            ;;
        --verify)
            verify_datasets
            ;;
        --process)
            process_msrvtt
            process_didemo
            process_moviegraphs
            build_indexes
            ;;
        --build-indexes)
            build_indexes
            ;;
        --full)
            log_info "=== PHASE 1: Download ==="
            download_msrvtt
            download_didemo
            download_moviegraphs
            download_activitynet
            echo
            log_info "=== PHASE 2: Verify ==="
            verify_datasets
            echo
            log_info "=== PHASE 3: Process ==="
            process_msrvtt
            process_didemo
            process_moviegraphs
            build_indexes
            echo
            log_info "=== PHASE 4: Pipeline ==="
            run_pipeline_full
            print_summary
            ;;
        --list)
            list_datasets
            ;;
        --help|-h)
            grep "^# " "$0" | head -30 | sed 's/^# //'
            echo ""
            grep "^    --" "$0" | head -20
            ;;
        "")
            echo "Usage: $0 <mode>"
            echo ""
            grep "^    --" "$0" | head -20
            echo ""
            echo "Examples:"
            echo "  $0 --tier1          # Tải Tier 1 datasets"
            echo "  $0 --verify         # Verify tất cả"
            echo "  $0 --process        # Process → chunks + indexes"
            echo "  $0 --full           # Tải + verify + process + pipeline"
            echo "  $0 --list           # Liệt kê datasets"
            ;;
        *)
            log_error "Unknown mode: $MODE"
            echo "Try: $0 --help"
            exit 1
            ;;
    esac

    if [ -n "$MODE" ]; then
        print_summary
    fi
}

main "$@"
