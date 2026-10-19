"""
Inspect Scene-Level Split (dry run — no image/label processing).
Run this first to verify the split before the full pipeline.
Generates: scene_split_report.txt
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from scene_level_pipeline import (
    load_metadata, build_scene_info, scene_level_split,
    write_scene_report,
    DATASET_ROOT, OUTPUT_ROOT, TRAIN_RATIO, RANDOM_SEED,
)

print("Loading metadata ...")
meta = load_metadata(DATASET_ROOT)

print("Scanning LiDAR files on disk ...")
scenes_info = build_scene_info(meta, DATASET_ROOT)
on_disk = [s for s in scenes_info if s["has_lidar_data"]]
print(f"  Scenes with data on disk: {len(on_disk)} / {len(scenes_info)}")

print("Performing scene-level split ...")
train_scenes, val_scenes = scene_level_split(on_disk, TRAIN_RATIO, RANDOM_SEED)
print(f"  Train: {len(train_scenes)} scenes")
print(f"  Val  : {len(val_scenes)} scenes")

os.makedirs(OUTPUT_ROOT, exist_ok=True)
ok = write_scene_report(OUTPUT_ROOT, train_scenes, val_scenes, RANDOM_SEED)

print("\n--- SAMPLE LEVEL COUNTS ---")
train_samples = sum(len(s["sample_tokens"]) for s in train_scenes)
val_samples   = sum(len(s["sample_tokens"]) for s in val_scenes)
print(f"  Train samples : {train_samples}  (x6 cameras = {train_samples*6})")
print(f"  Val   samples : {val_samples}    (x6 cameras = {val_samples*6})")
print(f"  Total samples : {train_samples+val_samples}")
ratio = train_samples / (train_samples + val_samples) * 100
print(f"  Effective split: {ratio:.1f}% train / {100-ratio:.1f}% val (at sample level)")
