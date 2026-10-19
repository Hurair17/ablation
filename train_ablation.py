# train_ablation.py
# ------------------
# Runs any single ablation variant of MBiCMF using train.py's full training loop.
#
# Strategy:
#   1. Build the right model subclass by monkey-patching MultiModalYOLO11.
#   2. Change working directory so each ablation's outputs land in
#      runs_ablation/<ID>/runs_multimodal/  (train.py always writes to runs_multimodal/).
#   3. Call train.main(args) directly — no code duplication.
#
# Usage (single ablation):
#   python train_ablation.py --ablation A2 --data_root C:\Users\hurair\Desktop\MultiModel\scene_level_pipeline\dataset1 --epochs 2 --batch_size 4 --nc 10

# Usage (all ablations sequentially):
#   python train_ablation.py --ablation ALL --data_root C:\Users\hurair\Desktop\MultiModel\scene_level_pipeline\dataset1 --epochs 2 --batch_size 4 --nc 10

import os, sys, argparse, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import multimodal_yolo11
from ablation_fusion import (
    CrossAttnUniCam, CrossAttnUniLidar, CameraPassthrough,
    CrossAttnNoResidual, CrossAttnNoBottleneck,
    CrossAttnNoFFN, CrossAttnNoDropout, ConcatFusion,
)

# ─────────────────────────────────────────────────────────────────────────────
# Channel dimensions for each FPN scale
# ─────────────────────────────────────────────────────────────────────────────
_SCALE_ARGS = {
    "p3": (768,  512,  768),
    "p4": (768,  512,  768),
    "p5": (1536, 1024, 1536),
}

# ─────────────────────────────────────────────────────────────────────────────
# Ablation registry
# Each entry: (short_label, fuse_p3_cls, fuse_p4_cls, fuse_p5_cls)
# None for fuse_p5 means "use the original CrossAttnFusion2D" (A4 only)
# ─────────────────────────────────────────────────────────────────────────────
_REGISTRY = {
    "A1": ("Full-MBiCMF",      None,               None,               None),
    "A2": ("UniCam",           CrossAttnUniCam,    CrossAttnUniCam,    CrossAttnUniCam),
    "A3": ("UniLidar",         CrossAttnUniLidar,  CrossAttnUniLidar,  CrossAttnUniLidar),
    "A4": ("P5-only",          CameraPassthrough,  CameraPassthrough,  None),   # P5 unchanged
    "A5": ("NoResidual",       CrossAttnNoResidual,CrossAttnNoResidual,CrossAttnNoResidual),
    "A6": ("NoBottleneck",     CrossAttnNoBottleneck,CrossAttnNoBottleneck,CrossAttnNoBottleneck),
    "A7": ("NoFFN",            CrossAttnNoFFN,     CrossAttnNoFFN,     CrossAttnNoFFN),
    "A8": ("NoDropout",        CrossAttnNoDropout, CrossAttnNoDropout, CrossAttnNoDropout),
    "A9": ("ConcatFusion",     ConcatFusion,       ConcatFusion,       ConcatFusion),
}


# ─────────────────────────────────────────────────────────────────────────────
# Build model class
# ─────────────────────────────────────────────────────────────────────────────

def build_model_class(ablation_id):
    """
    Returns a MultiModalYOLO11 subclass where fuse_p3/p4/p5
    are replaced by the ablation variants.
    A1 returns the original class unchanged.
    """
    label, fp3_cls, fp4_cls, fp5_cls = _REGISTRY[ablation_id]

    if ablation_id == "A1":
        return multimodal_yolo11.MultiModalYOLO11

    # A4: P5 stays as CrossAttnFusion2D; only P3/P4 become CameraPassthrough
    if ablation_id == "A4":
        fp5_cls = multimodal_yolo11.CrossAttnFusion2D

    # Capture in closure (Python late-binding workaround)
    _fp3, _fp4, _fp5 = fp3_cls, fp4_cls, fp5_cls

    class _AblationModel(multimodal_yolo11.MultiModalYOLO11):
        def __init__(self, nc, cam_in_ch=3, lidar_in_ch=3, fusion_type="cross_attn"):
            # Initialise with cross_attn to get the correct backbone + head structure
            super().__init__(nc, cam_in_ch, lidar_in_ch, fusion_type="cross_attn")
            # Replace fusion modules with ablation variants
            self.fuse_p3 = _fp3(*_SCALE_ARGS["p3"])
            self.fuse_p4 = _fp4(*_SCALE_ARGS["p4"])
            self.fuse_p5 = _fp5(*_SCALE_ARGS["p5"])

    _AblationModel.__name__ = f"MBiCMF_{ablation_id}_{label}"
    return _AblationModel


# ─────────────────────────────────────────────────────────────────────────────
# Run one ablation
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(ablation_id, base_args):
    label = _REGISTRY[ablation_id][0]
    print(f"\n{'=' * 65}")
    print(f"  ABLATION {ablation_id}: {label}")
    print(f"{'=' * 65}\n")

    # 1. Patch MultiModalYOLO11 globally — train.py will pick it up on import
    AblationClass = build_model_class(ablation_id)
    multimodal_yolo11.MultiModalYOLO11 = AblationClass

    # 2. Create per-ablation output directory and cd into it
    #    train.py always writes to runs_multimodal/ relative to cwd
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "runs_ablation", ablation_id)
    os.makedirs(out_dir, exist_ok=True)
    prev_cwd = os.getcwd()
    os.chdir(out_dir)

    # 3. Build a Namespace matching train.parse_args() fields exactly
    import argparse
    train_args = argparse.Namespace(
        data_root=base_args.data_root,
        epochs=base_args.epochs,
        batch_size=base_args.batch_size,
        img_size=base_args.img_size,
        nc=base_args.nc,
        fusion_type="cross_attn",   # class handles the variant; train.py sees cross_attn
        workers=base_args.workers,
        pretrained=base_args.pretrained,
    )

    # 4. Run train.main() — full training loop, validation, checkpointing
    import train as _train
    try:
        _train.main(train_args)
    finally:
        os.chdir(prev_cwd)   # always restore cwd

    print(f"\n[{ablation_id}] done. Results → runs_ablation/{ablation_id}/runs_multimodal/")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="MBiCMF ablation study — wraps train.py for each variant"
    )
    p.add_argument("--ablation",   required=True,
                   choices=list(_REGISTRY.keys()) + ["ALL"],
                   help="Ablation ID (A1–A9) or ALL to run every variant sequentially")
    p.add_argument("--data_root",  required=True,
                   help="Dataset root (same as train.py --data_root)")
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--img_size",   type=int, default=640)
    p.add_argument("--nc",         type=int, default=10)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--pretrained", default=None,
                   help="Path to yolo11x.pt for camera backbone init")
    return p.parse_args()


def main():
    args = parse_args()

    if args.ablation == "ALL":
        for aid in _REGISTRY:
            run_ablation(aid, args)
        print("\n\nAll ablations complete.")
        print("Results tree:")
        for aid, (label, *_) in _REGISTRY.items():
            path = os.path.join("runs_ablation", aid, "runs_multimodal", "best.pt")
            exists = "[OK]" if os.path.exists(path) else "[missing]"
            print(f"  {aid} {label:<18} {exists}  {path}")
    else:
        run_ablation(args.ablation, args)


if __name__ == "__main__":
    main()
