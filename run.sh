#!/usr/bin/env bash
# =============================================================
# MBiCMF Ablation Study — Server Run Script
# =============================================================
# One server job = one ablation.  Change ABLATION_ID below and
# upload this script as a new issue.  All jobs share the same
# generated dataset but write to separate output directories
# so they never interfere with each other or with the A2 job.
#
# Ablation IDs:
#   A1 = Full-MBiCMF (reference)
#   A2 = UniCam          A3 = UniLidar
#   A4 = P5-only         A5 = NoResidual
#   A6 = NoBottleneck    A7 = NoFFN
#   A8 = NoDropout       A9 = ConcatFusion
#   A10= EarlyFusion
# For fusion use one of thefollowing as a ablation id
# choices=["gated", "cross_attn", "gated_bilinear", "adapt_gated"],
#Early Fusion 
# cam (3ch) ─┐
#            ├─ cat → [6ch] → Conv1×1+BN+SiLU → [3ch] → backbone_cam → (p3, p4, p5) → head
# lidar (3ch)┘
# =============================================================
set -euo pipefail

# ── ONE LINE TO CHANGE PER JOB ───────────────────────────────
ABLATION_ID="${ABLATION_ID:-cross_attn}"
# ─────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo ""
echo "============================================================"
echo "  ABLATION JOB : $ABLATION_ID"
echo "============================================================"

# ===========================================================
# 0. Clone code into /tmp
#    Unique CODE_DIR per ablation → two jobs on the same
#    machine never overwrite each other's clone.
# ===========================================================
echo ""
echo "===== Clone code ====="

TARGET_BRANCH="main"
REPO_URL="https://github.com/Hurair17/ablation.git"

CODE_DIR="/tmp/multimodal_code_${ABLATION_ID}"
rm -rf "$CODE_DIR"
git clone --depth 1 --branch "$TARGET_BRANCH" "$REPO_URL" "$CODE_DIR"
echo "Cloned: $(git -C "$CODE_DIR" log -1 --oneline)"

if grep -q "bottleneck_dim" "$CODE_DIR/multimodal_yolo11.py"; then
    echo "✓ CrossAttnFusion confirmed"
else
    echo "✗ FATAL: bottleneck_dim missing — check GitHub repo"
    exit 1
fi

# Persistent output — unique per ablation
OUT="/app/output/hurair_ablation_${ABLATION_ID}"
mkdir -p "$OUT"

# Dataset — same shared path as the main run.sh
# If the A2 job already generated it, this step is skipped.
OUTPUT_DIR="$SCRIPT_DIR/generated_dataset"

echo "Ablation   : $ABLATION_ID"
echo "Code dir   : $CODE_DIR"
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

CUDA_MAJOR=$(nvidia-smi 2>/dev/null \
    | grep -oP "CUDA Version: \K[0-9]+" \
    | head -1 \
    || echo "12")
echo "Detected CUDA major: $CUDA_MAJOR"

if [[ "$CUDA_MAJOR" -ge 13 ]]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --quiet
elif [[ "$CUDA_MAJOR" -ge 12 ]]; then
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
else
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet
fi

pip install -r "$CODE_DIR/requirements.txt" --quiet

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
PY

# ===========================================================
# 5. Dataset
#    Shared with all ablation jobs.  Skip if already present
#    (the A2 job will have generated it already).
# ===========================================================
echo ""
echo "===== Dataset ====="

if [[ -f "$OUTPUT_DIR/data.yaml" ]]; then
    N_TRAIN=$(find "$OUTPUT_DIR/train/camera" -name '*.jpg' 2>/dev/null | wc -l || echo 0)
    N_VAL=$(find   "$OUTPUT_DIR/val/camera"   -name '*.jpg' 2>/dev/null | wc -l || echo 0)
    echo "Dataset already present: $N_TRAIN train / $N_VAL val — skipping generation."
else
    echo "Dataset not found — locating nuScenes and generating..."

    NUSC_ROOT=""

    # zip candidates
    for zipcandidate in \
        "/app/data/datasets/split_nuscenes.zip" \
        "/datasets/split_nuscenes.zip"; do
        if [[ -f "$zipcandidate" && -z "$NUSC_ROOT" ]]; then
            EXTRACT_DIR="$(pwd)/nuscenes_extracted"
            mkdir -p "$EXTRACT_DIR"
            unzip -q "$zipcandidate" -d "$EXTRACT_DIR"
            for probe in "$EXTRACT_DIR" "$EXTRACT_DIR"/*/; do
                probe="${probe%/}"
                if [[ -d "$probe/v1.0-trainval" ]]; then
                    NUSC_ROOT="$probe"; break
                fi
            done
        fi
    done

    # directory candidates
    for candidate in \
        "/app/data/datasets/split_nuscenes" \
        "/app/data/nuscenes" \
        "/app/data/nuscense" \
        "/data/nuscenes" \
        "/mnt/data/nuscenes"; do
        if [[ -z "$NUSC_ROOT" && -d "${candidate}/v1.0-trainval" ]]; then
            NUSC_ROOT="$candidate"; break
        fi
    done

    if [[ -z "${NUSC_ROOT:-}" ]]; then
        echo "ERROR: Cannot find nuScenes root."
        echo "Override:  NUSC_ROOT=/path bash run.sh"
        find / -maxdepth 7 -type d -name "v1.0-trainval" 2>/dev/null | head -10 || true
        exit 1
    fi
    echo "nuScenes root: $NUSC_ROOT"

    cd "$SCRIPT_DIR"
    NUSC_ROOT="$NUSC_ROOT" OUTPUT_DIR="$OUTPUT_DIR" python - <<'PY'
import sys, os
sys.path.insert(0, "data_pipeline")
import scene_level_pipeline as sl
sl.run(
    dataset_root=os.environ["NUSC_ROOT"],
    output_root=os.environ["OUTPUT_DIR"],
)
PY
    echo "Dataset generation complete."
fi

# ===========================================================
# 6. Train ablation
#    train_ablation.py writes to:
#      $CODE_DIR/runs_ablation/$ABLATION_ID/runs_multimodal/
#    We copy that to $OUT afterwards.
# ===========================================================
echo ""
echo "===== Train ablation $ABLATION_ID ====="

# python "$CODE_DIR/train_ablation.py" \
#     --ablation   "$ABLATION_ID" \
#     --data_root  "$OUTPUT_DIR" \
#     --epochs     100 \
#     --batch_size 4 \
#     --img_size   640 \
#     --nc         10 \
#     --workers    4


python "$CODE_DIR/train.py" \
    --fusion_type   "$ABLATION_ID" \
    --data_root  "$OUTPUT_DIR" \
    --epochs     100 \
    --batch_size 4 \
    --img_size   640 \
    --nc         10 \
    --workers    4

echo "Training done."

# ===========================================================
# 7. Copy results to persistent output
#    /tmp is cleaned between jobs; $OUT (/app/output/...) is not.
# ===========================================================
echo ""
echo "===== Copy results to persistent storage ====="

RESULTS_SRC="$CODE_DIR/runs_ablation/${ABLATION_ID}/runs_multimodal"

if [[ -d "$RESULTS_SRC" ]]; then
    cp -r "$RESULTS_SRC" "$OUT/runs_multimodal"
    echo "Copied: $RESULTS_SRC -> $OUT/runs_multimodal"
else
    echo "WARNING: Results not found at $RESULTS_SRC"
fi

cp "$OUTPUT_DIR/scene_split_report.txt" "$OUT/" 2>/dev/null || true
cp "$OUTPUT_DIR/data.yaml"              "$OUT/" 2>/dev/null || true

# ===========================================================
# 8. Package results
# ===========================================================
echo ""
echo "===== Package results ====="

tar -czf "$OUT/results_${ABLATION_ID}.tar.gz" \
    -C "$OUT" \
    runs_multimodal \
    2>/dev/null || true

echo ""
echo "============================================================"
echo "DONE — Ablation $ABLATION_ID"
echo "Results : $OUT/"
echo "Download: $OUT/results_${ABLATION_ID}.tar.gz"
echo "  runs_multimodal/best.pt          best checkpoint"
echo "  runs_multimodal/training_log_*   CSV metrics per epoch"
echo "============================================================"

# ===========================================================
# 9. Environment versions (reproducibility / mAP debug)
# ===========================================================
echo ""
echo "============================================================"
echo "ENVIRONMENT VERSIONS"
echo "============================================================"
echo ""
echo "--- System ---"
uname -a || true
python --version
pip --version

echo ""
echo "--- GPU ---"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true

echo ""
echo "--- Python packages ---"
python - <<'PY'
import sys
print(f"Python        : {sys.version}")

def v(pkg):
    try:
        import importlib
        m = importlib.import_module(pkg)
        return getattr(m, "__version__", "installed (no __version__)")
    except ImportError:
        return "NOT INSTALLED"

for mod, label in [
    ("torch",        "PyTorch       "),
    ("torchvision",  "torchvision   "),
    ("ultralytics",  "ultralytics   "),
    ("numpy",        "numpy         "),
    ("cv2",          "OpenCV        "),
    ("PIL",          "Pillow        "),
    ("tqdm",         "tqdm          "),
    ("scipy",        "scipy         "),
    ("sklearn",      "scikit-learn  "),
    ("pyquaternion", "pyquaternion  "),
    ("nuscenes",     "nuscenes-devkit"),
    ("matplotlib",   "matplotlib    "),
]:
    print(f"{label}: {v(mod)}")

import torch
print(f"\nPyTorch CUDA build : {torch.version.cuda}")
print(f"cuDNN version      : {torch.backends.cudnn.version()}")
print(f"CUDA available     : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name           : {torch.cuda.get_device_name(0)}")
    print(f"GPU memory (GB)    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}")
PY

echo ""
echo "--- pip freeze (full list) ---"
pip freeze

echo "============================================================"
echo "END OF VERSION REPORT"
echo "============================================================"
