# train_baseline.py
# -----------------
# Camera-only YOLO11m baseline with multi-scale training restricted to
# the SAME range as the multimodal model (320-640 px, step 32).
#
# ultralytics built-in multi_scale=True uses +/-50% of imgsz, which would
# give 320-960 for imgsz=640 -- unfairly wider than the multimodal.
# This script subclasses DetectionTrainer and overrides preprocess_batch()
# to enforce 320-640.  The on_train_batch_start callback approach does NOT
# work because ultralytics stores the batch as a local variable inside the
# training loop -- trainer.batch is never set as an attribute on the trainer.
#
# Usage:
#   python train_baseline.py \
#       --data   /path/to/camera_only.yaml \
#       --output /path/to/output_dir
# python train_baseline.py --data "C:\Users\hurair\Desktop\MultiModel\scene_level_pipeline\dataset1\camera_only.yaml" --output "C:\Users\hurair\Desktop\MultiModal\multimodal_training\YOLO"


import argparse
import random
import torch
import torch.nn.functional as F
from pathlib import Path
from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer


# ── Match exactly the multimodal multi-scale range ──────────────────────────
IMG_SIZES  = list(range(320, 641, 32))   # [320, 352, 384, ..., 640]
SCALE_FREQ = 10                          # change size every N batches


class MultiScaleDetectionTrainer(DetectionTrainer):
    # Instance-level counters so multiple trainer instances don't share state
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ms_batch_count = 0
        self._ms_current_size = 640

    def preprocess_batch(self, batch):
        # Let ultralytics do its standard preprocessing (normalize, send to GPU)
        batch = super().preprocess_batch(batch)

        # Advance counter and maybe pick a new scale
        self._ms_batch_count += 1
        if self._ms_batch_count % SCALE_FREQ == 0:
            self._ms_current_size = random.choice(IMG_SIZES)

        imgs = batch["img"]
        target = self._ms_current_size

        if imgs.shape[-1] != target or imgs.shape[-2] != target:
            batch["img"] = F.interpolate(
                imgs.float(),
                size=(target, target),
                mode="bilinear",
                align_corners=False,
            ).to(imgs.dtype)

        return batch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",    required=True,  help="Path to camera_only.yaml")
    p.add_argument("--output",  default="runs_baseline",
                   help="Output directory (default: runs_baseline)")
    p.add_argument("--epochs",  type=int, default=50)
    p.add_argument("--batch",   type=int, default=4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--model", type=int, default="yolo11m.yaml")
    p.add_argument("--device",  default="0")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Camera-only YOLO11m baseline")
    print(f"  Multi-scale: {IMG_SIZES[0]}-{IMG_SIZES[-1]} px  "
          f"(step 32, every {SCALE_FREQ} batches)")
    print("  Matches multimodal multi-scale range exactly")
    print("  Method: DetectionTrainer.preprocess_batch override")
    print("=" * 60)

    # model = YOLO("yolo11m.pt")    # pretrained COCO weights (yolo11m)
    # model = YOLO("yolo11m.yaml")  # random init yolo11m (no pretrained)
    model = YOLO(args.model)   # random init yolo11x -- matches multimodal backbone capacity

    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=640,         # DataLoader loads at 640; trainer resizes per batch
        multi_scale=False, # disable ultralytics +/-50% scaling; trainer handles it
        device=args.device,
        workers=args.workers,
        project=str(out_dir),
        name="yolo11m_camera_only",
        exist_ok=True,
        trainer=MultiScaleDetectionTrainer,

        # ── Augmentation matched to multimodal_dataset.py ───────────────────
        mosaic=0.0,        # multimodal has no mosaic
        mixup=0.0,
        copy_paste=0.0,
        fliplr=0.5,        # match: horizontal flip p=0.5
        degrees=10.0,      # match: random affine +-10 deg
        translate=0.1,     # match: random affine +-10%
        scale=0.1,         # match: random affine +-10%
        shear=10.0,        # match: random affine +-10 deg
        hsv_h=0.01,        # match: weak colour jitter (multimodal: p=0.2, +-5%)
        hsv_s=0.1,         # reduced from ultralytics default 0.7
        hsv_v=0.1,         # reduced from ultralytics default 0.4

        # ── Optimiser / training (same as multimodal) ────────────────────────
        optimizer="AdamW",
        lr0=1e-3,
        weight_decay=5e-4,
        warmup_epochs=3,
        close_mosaic=0,
        amp=True,
    )

    print("\nBaseline training complete.")
    print(f"Results: {out_dir}/yolo11m_camera_only/")


if __name__ == "__main__":
    main()
