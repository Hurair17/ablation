#!/usr/bin/env bash
# =============================================================
# Multimodal YOLO11 (Camera + LiDAR Fusion) — Server Run Script
# =============================================================
# Usage:
#   bash run.sh                               # auto-detect nuScenes
#   NUSC_ROOT=/path/to/nuscenes bash run.sh   # manual override
#
# Dataset stays next to the repo (server does not delete it).
# Training results → /app/output/hurair_cross_attn/   (persistent)
#   runs_multimodal/      checkpoints + CSV log
#   results.tar.gz        packaged for download
# =============================================================
set -euo pipefail

# Where the repo lives (all .py files)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ===========================================================
# 0. Clone latest code to /tmp (WRITABLE) and use it for training
#
#    Root cause of repeated BatchNorm crash:
#      /app/scratch/input/ is READ-ONLY on this server platform.
#      No amount of git-reset or in-place patching can fix the files
#      in SCRIPT_DIR. Solution: always clone fresh into /tmp which
#      IS writable, and run all Python from there.
# ===========================================================
echo ""
echo "===== Cloning latest code to writable workspace ====="

TARGET_BRANCH="main"
REPO_URL="https://github.com/Hurair17/multi_modal_updated.git"

# /tmp is always writable; blow away any previous run's clone
CODE_DIR="/tmp/multimodal_code"
rm -rf "$CODE_DIR"
git clone --depth 1 --branch "$TARGET_BRANCH" "$REPO_URL" "$CODE_DIR"
echo "Cloned: $(git -C $CODE_DIR log -1 --oneline)"

# Verify the bottleneck CrossAttnFusion is in the fresh clone before wasting time
if grep -q "bottleneck_dim" "$CODE_DIR/multimodal_yolo11.py"; then
    echo "✓ Bottleneck CrossAttnFusion confirmed"
else
    echo "✗ FATAL: bottleneck_dim missing in fresh clone — check GitHub repo"
    exit 1
fi



# Persistent output folder on the server
OUT="/app/output/hurair_cross_attn"
mkdir -p "$OUT"

# Dataset stays local (not deleted by server)
OUTPUT_DIR="$SCRIPT_DIR/generated_dataset"

cd "$SCRIPT_DIR"

echo "===== Multimodal YOLO11 — Camera + LiDAR Fusion ====="
echo "Script dir : $SCRIPT_DIR"
echo "Output dir : $OUT"
echo "Dataset    : $OUTPUT_DIR"

# ===========================================================
# 1. System check
# ===========================================================
echo ""
echo "===== System check ====="
python --version
nvidia-smi || true

# ===========================================================
# 2. Environment
# ===========================================================
echo ""
echo "===== Environment ====="
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ===========================================================
# 3. Install dependencies
# ===========================================================
echo ""
echo "===== Install dependencies ====="

pip install -U pip --quiet

# Detect CUDA major version and install matching PyTorch
CUDA_MAJOR=$(nvidia-smi 2>/dev/null \
    | grep -oP "CUDA Version: \K[0-9]+" \
    | head -1 \
    || echo "12")
echo "Detected CUDA major version: $CUDA_MAJOR"

if [[ "$CUDA_MAJOR" -ge 13 ]]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --quiet
elif [[ "$CUDA_MAJOR" -ge 12 ]]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
else
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet
fi

# Pin ultralytics — the loss API changed across versions; code targets 8.3.x
pip install -r requirements.txt --quiet

# ===========================================================
# 4. Verify imports
# ===========================================================
echo ""
echo "===== Import verification ====="
python - <<'PY'
import torch, cv2, ultralytics, numpy, tqdm
import pyquaternion  # noqa
print("PyTorch     :", torch.__version__)
print("CUDA avail  :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU         :", torch.cuda.get_device_name(0))
    print("GPU memory  :", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
print("ultralytics :", ultralytics.__version__)
print("OpenCV      :", cv2.__version__)
print("numpy       :", numpy.__version__)
print("pyquaternion: OK")
PY

# ===========================================================
# 5. Locate and extract nuScenes dataset
#    Primary: /app/data/datasets/split_nuscenes(.zip)
#    Fallback: /app/data/nuscenes, /data/nuscenes, etc.
# ===========================================================
echo ""
echo "===== Locate nuScenes dataset ====="

# ---- Step 5a: find the zip / folder ----
# Try absolute first, then relative to script directory
RAW_DATA_PATH=""
for candidate in \
    "/app/data/datasets/split_nuscenes.zip" \
    "/app/data/datasets/split_nuscenes" \
    "/datasets/split_nuscenes.zip" \
    "/datasets/split_nuscenes" \
    "$(pwd)/datasets/split_nuscenes.zip" \
    "$(pwd)/datasets/split_nuscenes"; do
    if [[ -e "$candidate" ]]; then
        RAW_DATA_PATH="$candidate"
        echo "Found data at: $RAW_DATA_PATH"
        break
    fi
done

if [[ -z "$RAW_DATA_PATH" ]]; then
    echo "WARNING: datasets/split_nuscenes not found at expected paths."
    echo "Falling back to auto-detection..."
fi

# ---- Step 5b: extract if it is a zip ----
EXTRACT_DIR="$(pwd)/nuscenes_extracted"

if [[ "$RAW_DATA_PATH" == *.zip ]]; then
    echo "Extracting zip: $RAW_DATA_PATH -> $EXTRACT_DIR ..."
    mkdir -p "$EXTRACT_DIR"
    unzip -q "$RAW_DATA_PATH" -d "$EXTRACT_DIR"
    echo "Extraction complete."
    # After extraction the nuScenes root is either the extract dir itself
    # or a subdirectory — probe both
    NUSC_ROOT=""
    for probe in "$EXTRACT_DIR" "$EXTRACT_DIR"/*/; do
        probe="${probe%/}"   # strip trailing slash
        if [[ -d "$probe/v1.0-trainval" ]]; then
            NUSC_ROOT="$probe"
            break
        fi
    done
elif [[ -d "$RAW_DATA_PATH" ]]; then
    # Already a directory; check if it IS the root or contains the root
    NUSC_ROOT=""
    for probe in "$RAW_DATA_PATH" "$RAW_DATA_PATH"/*/; do
        probe="${probe%/}"
        if [[ -d "$probe/v1.0-trainval" ]]; then
            NUSC_ROOT="$probe"
            break
        fi
    done
fi

# ---- Step 5c: fall back to other common server locations ----
if [[ -z "${NUSC_ROOT:-}" ]]; then
    for candidate in \
        "/app/data/nuscenes" \
        "/app/data/nuscense" \
        "/data/nuscenes" \
        "/mnt/data/nuscenes"; do
        if [[ -d "${candidate}/v1.0-trainval" ]]; then
            NUSC_ROOT="$candidate"
            break
        fi
    done
fi

if [[ -z "${NUSC_ROOT:-}" ]]; then
    echo "ERROR: Cannot find nuScenes root (directory containing v1.0-trainval/)."
    echo ""
    echo "Override manually:"
    echo "  NUSC_ROOT=/path/to/nuscenes bash run.sh"
    echo ""
    echo "Searching server for 'v1.0-trainval':"
    find / -maxdepth 7 -type d -name "v1.0-trainval" 2>/dev/null | head -15 || true
    exit 1
fi

echo "nuScenes root : $NUSC_ROOT"
echo "Metadata dir  : $NUSC_ROOT/v1.0-trainval"
echo "JSON files    : $(find "$NUSC_ROOT/v1.0-trainval" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)"
echo "LiDAR samples : $(find "$NUSC_ROOT/samples/LIDAR_TOP" -maxdepth 1 -type f 2>/dev/null | wc -l)"
echo "CAM_FRONT     : $(find "$NUSC_ROOT/samples/CAM_FRONT" -maxdepth 1 -type f 2>/dev/null | wc -l)"

# ===========================================================
# 6. Generate dataset (scene-level split, seed=42)
# ===========================================================
echo ""
echo "===== Generate dataset (scene-level split, seed=42) ====="

if [[ -f "$OUTPUT_DIR/data.yaml" ]]; then
    N_TRAIN=$(find "$OUTPUT_DIR/train/camera" -name '*.jpg' 2>/dev/null | wc -l || echo 0)
    N_VAL=$(find   "$OUTPUT_DIR/val/camera"   -name '*.jpg' 2>/dev/null | wc -l || echo 0)
    echo "Dataset already exists: $N_TRAIN train / $N_VAL val images. Skipping generation."
else
    echo "Generating from $NUSC_ROOT  ->  $OUTPUT_DIR ..."
    NUSC_ROOT="$NUSC_ROOT" OUTPUT_DIR="$OUTPUT_DIR" python - <<'PY'
import sys, os
sys.path.insert(0, "data_pipeline")
import scene_level_pipeline as sl

# Pass paths explicitly — default parameter values are captured at import
# time, so overriding sl.DATASET_ROOT after import has no effect.
sl.run(
    dataset_root=os.environ["NUSC_ROOT"],
    output_root=os.environ["OUTPUT_DIR"],
)
PY
    echo "Dataset generation complete."
fi

# # ===========================================================
# # 7. Prepare camera-only YOLO dataset
# #    ultralytics finds labels via /images/ -> /labels/ substitution,
# #    so copy labels into camera/ and point yaml at camera/ directly.
# # ===========================================================
# echo ""
# echo "===== Prepare camera-only dataset for YOLO ====="

# OUTPUT_DIR_ESC="$OUTPUT_DIR" python - <<'PY'
# import shutil, os
# base = os.environ["OUTPUT_DIR_ESC"]
# for split in ["train", "val"]:
#     ldir = os.path.join(base, split, "labels")
#     cdir = os.path.join(base, split, "camera")
#     if not os.path.isdir(ldir):
#         print(f"  {split}: labels dir missing, skipping")
#         continue
#     files = [f for f in os.listdir(ldir) if f.endswith(".txt")]
#     copied = sum(
#         1 for f in files
#         if not os.path.exists(os.path.join(cdir, f))
#         and shutil.copy2(os.path.join(ldir, f), os.path.join(cdir, f)) is None
#     )
#     print(f"  {split}: {copied} label files copied into camera/ ({len(files)} total)")
# PY

# # Write camera_only.yaml with server-local absolute path
# cat > "$OUTPUT_DIR/camera_only.yaml" <<YAML
# path: $OUTPUT_DIR
# train: train/camera
# val:   val/camera
# nc: 10
# names: ['pedestrian','car','motorcycle','bicycle','bus','truck','construction','trailer','barrier','traffic_cone']
# YAML
# echo "Written: $OUTPUT_DIR/camera_only.yaml"

# # ===========================================================
# # 8. Camera-only YOLO11m baseline (reviewer ablation)
# #    Uses train_baseline.py which restricts multi-scale to 320-640
# #    — identical range to the multimodal model (ultralytics built-in
# #      multi_scale=True would go up to 960 px, unfairly wider).
# # ===========================================================
# echo ""
# echo "===== Camera-only YOLO11m baseline (320-640 multi-scale) ====="

# # Write camera_only.yaml (camera images only, same split as multimodal)
# cat > "$OUTPUT_DIR/camera_only.yaml" <<YAML
# path: $OUTPUT_DIR
# train: train/camera
# val:   val/camera
# nc: 10
# names: ['pedestrian','car','motorcycle','bicycle','bus','truck','construction','trailer','barrier','traffic_cone']
# YAML
# echo "Written: $OUTPUT_DIR/camera_only.yaml"

# python "$CODE_DIR/train_baseline.py" \
#     --data    "$OUTPUT_DIR/camera_only.yaml" \
#     --output  "$OUT/runs_baseline" \
#     --epochs  100 \
#     --batch   4 \
#     --workers 4 \
#     --device  0

# echo "Camera-only baseline done."

# ===========================================================
# 8b. Download pretrained backbone (yolo11x.pt)
#     Must happen BEFORE cd "$OUT" so the file lands in $SCRIPT_DIR
#     and we can pass an absolute path to train.py
# ===========================================================
# echo ""
# echo "===== Download pretrained backbone ====="
# PRETRAINED_PT="$SCRIPT_DIR/yolo11x.pt"
# if [[ -f "$PRETRAINED_PT" ]]; then
#     echo "yolo11x.pt already present: $PRETRAINED_PT"
# else
#     echo "Downloading yolo11x.pt -> $PRETRAINED_PT ..."
#     python -c "
# from ultralytics import YOLO
# import shutil, os
# model = YOLO('yolo11x.pt')   # downloads to ultralytics cache
# # locate the cached file and copy next to the script
# src = model.ckpt_path if hasattr(model, 'ckpt_path') else 'yolo11x.pt'
# dst = '$PRETRAINED_PT'
# if os.path.abspath(src) != os.path.abspath(dst):
#     shutil.copy2(src, dst)
# print('Saved to:', dst)
# "
#     echo "Download complete."
# fi

# ===========================================================
# 9. Train Multimodal YOLO11 (main model)
# ===========================================================
echo ""
echo "===== Train Multimodal YOLO11 (cross_attn fusion) ====="

# cd into OUT so runs_multimodal/ and CSV logs are created inside /app/output/hurair_cross_attn/
# Use CODE_DIR (writable /tmp clone) for all Python files — NOT the read-only SCRIPT_DIR
cd "$OUT"
python "$CODE_DIR/train.py" --data_root "$OUTPUT_DIR" --epochs 50 --batch_size 4 --img_size 640 --nc 10 --fusion_type cross_attn --workers 4


# python train.py \
#   --data_root "C:\Users\hurair\Desktop\MultiModel\scene_level_pipeline\temporal_dataset1" \
#   --epochs 100 \
#   --batch_size 4 \
#   --img_size 640 \
#   --nc 10 \
#   --fusion_type cross_attn \
#   --workers 4 \
#   --pretrained yolo11m.pt


cd "$SCRIPT_DIR"

echo "Multimodal training done."

# ===========================================================
# 10. Package results
# ===========================================================
echo ""
echo "===== Save results ====="

# Copy supporting files into OUT (scene report + dataset yaml)
cp "$OUTPUT_DIR/scene_split_report.txt" "$OUT/" 2>/dev/null || true
cp "$OUTPUT_DIR/data.yaml"              "$OUT/" 2>/dev/null || true

# Package only the results (not the large dataset images)
tar -czf "$OUT/results.tar.gz" \
    -C "$OUT" \
    runs_multimodal \
    scene_split_report.txt \
    data.yaml \
    2>/dev/null || true

echo ""
echo "============================================================"
echo "DONE"
echo "All outputs : $OUT/"
echo "Download    : $OUT/results.tar.gz"
echo "  runs_multimodal/        checkpoints, CSV log"
echo "  scene_split_report.txt  train/val scene list (for paper)"
echo "============================================================"
