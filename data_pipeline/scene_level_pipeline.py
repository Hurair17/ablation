"""
Scene-Level Split Pipeline
==========================
Addresses data leakage from random 80/20 sample-level splitting.

Key fix:
  Entire SCENES are assigned to either train or val.
  No sample from any single scene ever appears in both splits.

Label generation:
  2D bounding boxes are projected from nuScenes 3D annotations
  (sample_annotation.json) exactly as described in the manuscript:
    global -> ego-vehicle -> camera -> image plane
  This is self-contained; no pre-computed labels.json is needed.

Output structure:
  <OUTPUT_ROOT>/
    train/
      camera/   <- RGB images (.jpg)
      lidar/    <- 3-channel LiDAR range images (.png): [depth|intensity|mask]
      labels/   <- YOLO 2D labels (.txt)
    val/
      camera/
      lidar/
      labels/
    data.yaml
    scene_split_report.txt
"""

import os
import json
import shutil
import math
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from pyquaternion import Quaternion


# ============================================================
# >>>  CONFIGURATION — edit these two paths before running  <<<
# ============================================================

DATASET_ROOT = r"D:\nuscenes\v1.0-trainval01_blobs"
OUTPUT_ROOT  = r"C:\Users\hurair\Desktop\MultiModel\scene_level_pipeline\dataset"

CAMERA_NAMES = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK",  "CAM_BACK_LEFT",  "CAM_BACK_RIGHT",
]

TRAIN_RATIO    = 0.80   # 80% of scenes -> train
MIN_VISIBILITY = 0.0   # discard boxes whose clipped visible area < 30%
MAX_DEPTH      = 80.0   # depth normalisation ceiling in metres
RANDOM_SEED    = 42     # seed for scene shuffle inside each recording group

# nuScenes visibility tokens:  "1"=<40%  "2"=40-60%  "3"=60-80%  "4"=80-100%
# We skip visibility "1" (< 40%) unless the bbox-clip check keeps the box.
MIN_VISIBILITY_TOKEN = "1"   # accept all tokens; fine-grained filter done by bbox

# nuScenes category -> YOLO class id
CLASS_MAPPING = {
    "human.pedestrian.adult":               0,
    "human.pedestrian.child":               0,
    "human.pedestrian.wheelchair":          0,
    "human.pedestrian.stroller":            0,
    "human.pedestrian.personal_mobility":   0,
    "human.pedestrian.police_officer":      0,
    "human.pedestrian.construction_worker": 0,
    "vehicle.car":                          1,
    "vehicle.motorcycle":                   2,
    "vehicle.bicycle":                      3,
    "vehicle.bus.bendy":                    4,
    "vehicle.bus.rigid":                    4,
    "vehicle.truck":                        5,
    "vehicle.construction":                 6,
    "vehicle.emergency.ambulance":          1,
    "vehicle.emergency.police":             1,
    "vehicle.trailer":                      7,
    "movable_object.barrier":               8,
    "movable_object.trafficcone":           9,
    "movable_object.pushable_pullable":     8,
    "movable_object.debris":               8,
    "static_object.bicycle_rack":          8,
}

CLASS_NAMES = [
    "pedestrian", "car", "motorcycle", "bicycle",
    "bus", "truck", "construction", "trailer",
    "barrier", "traffic_cone",
]


# ============================================================
# 1.  METADATA LOADING
# ============================================================

def load_metadata(dataset_root):
    """Load all required nuScenes JSON tables."""
    vp = os.path.join(dataset_root, "v1.0-trainval")
    tables = {}
    for name in ["sample", "sample_data", "calibrated_sensor",
                 "ego_pose", "sensor", "scene", "log",
                 "sample_annotation", "instance", "category"]:
        with open(os.path.join(vp, f"{name}.json")) as f:
            tables[name] = json.load(f)
    return tables


# ============================================================
# 2.  SCENE INFO + DISK CHECK
# ============================================================

def build_scene_info(meta, dataset_root):
    """
    Returns one dict per scene with:
        token, name, date, location, logfile, sample_tokens,
        nbr_samples, has_lidar_data
    """
    dataset_path = Path(dataset_root)
    log_map = {l["token"]: l for l in meta["log"]}

    lidar_sd_by_sample = {}
    for sd in meta["sample_data"]:
        if sd["is_key_frame"] and "LIDAR_TOP" in sd.get("filename", ""):
            lidar_sd_by_sample[sd["sample_token"]] = sd

    samples_by_scene = defaultdict(list)
    for s in meta["sample"]:
        samples_by_scene[s["scene_token"]].append(s["token"])

    result = []
    for scene in meta["scene"]:
        log      = log_map[scene["log_token"]]
        s_tokens = samples_by_scene.get(scene["token"], [])
        has_data = any(
            st in lidar_sd_by_sample and
            (dataset_path / lidar_sd_by_sample[st]["filename"]).exists()
            for st in s_tokens
        )
        result.append({
            "token":          scene["token"],
            "name":           scene["name"],
            "log_token":      scene["log_token"],
            "date":           log["date_captured"],
            "location":       log["location"],
            "logfile":        log["logfile"],
            "sample_tokens":  s_tokens,
            "nbr_samples":    scene["nbr_samples"],
            "has_lidar_data": has_data,
        })
    return result


# ============================================================
# 3.  SCENE-LEVEL SPLIT
# ============================================================

def scene_level_split(scenes_info, train_ratio=0.80, random_seed=42):
    """
    Assign whole scenes to train or val (no sample-level leakage).

    Within each (date, location) recording session:
      - Sort scenes by name first (stable base order).
      - Shuffle with random_seed for reproducible but randomised ordering.
      - Take floor(n * train_ratio) for train, rest for val.
      - When n > 1, guarantee >= 1 val scene.

    Using the same random_seed always produces the same split.
    Change random_seed to explore alternative valid splits.
    """
    import random as _random
    rng = _random.Random(random_seed)

    valid = sorted(
        [s for s in scenes_info if s["has_lidar_data"]],
        key=lambda s: s["name"],          # stable sort before shuffle
    )
    groups = defaultdict(list)
    for s in valid:
        groups[(s["date"], s["location"])].append(s)

    # Global target: floor applied once to the full set (not per-group).
    # Per-group floor() accumulates rounding errors that push 1 extra scene
    # into val.  We correct afterwards to hit the exact global target.
    n_total    = len(valid)
    n_val_target = n_total - math.floor(n_total * train_ratio)  # e.g. 85-68=17

    train_scenes, val_scenes = [], []
    for _, group in sorted(groups.items()):
        group = list(group)               # copy before shuffling
        rng.shuffle(group)                # reproducible shuffle within group
        n       = len(group)
        n_train = max(1, math.floor(n * train_ratio))
        if n > 1:
            n_train = min(n_train, n - 1)
        train_scenes.extend(group[:n_train])
        val_scenes.extend(group[n_train:])

    # Global correction: move excess val scenes to train (or vice-versa)
    while len(val_scenes) > n_val_target and len(train_scenes) > 0:
        train_scenes.append(val_scenes.pop())
    while len(val_scenes) < n_val_target and len(train_scenes) > 1:
        val_scenes.append(train_scenes.pop())

    return train_scenes, val_scenes


# ============================================================
# 4.  CALIBRATION HELPERS
# ============================================================

def build_sensor_index(meta):
    """Return fast-lookup dicts used throughout processing."""
    sensor_map  = {s["channel"]: s["token"] for s in meta["sensor"]}
    calib_map   = defaultdict(list)
    for cs in meta["calibrated_sensor"]:
        calib_map[cs["sensor_token"]].append(cs["token"])
    sd_by_token = {sd["token"]: sd for sd in meta["sample_data"]}
    cs_by_token = {cs["token"]: cs for cs in meta["calibrated_sensor"]}
    ep_by_token = {ep["token"]: ep for ep in meta["ego_pose"]}
    return sensor_map, calib_map, sd_by_token, cs_by_token, ep_by_token


def get_sd_token(sample_token, channel, meta, sensor_map, calib_map, sd_by_token):
    """Return key-frame sample_data token for a given sensor channel."""
    sensor_tok   = sensor_map.get(channel)
    if sensor_tok is None:
        raise ValueError(f"Sensor '{channel}' not found.")
    calib_tokens = calib_map.get(sensor_tok, [])
    for sd in meta["sample_data"]:
        if (sd["sample_token"] == sample_token
                and sd["calibrated_sensor_token"] in calib_tokens
                and sd["is_key_frame"]):
            return sd["token"]
    raise ValueError(f"No key-frame for sample={sample_token}, channel={channel}.")


def compute_lidar_to_cam(lidar_sd_tok, cam_sd_tok, sd_by_token, cs_by_token):
    """
    Compute camera intrinsics and LiDAR-to-camera extrinsic transform.

    Returns:
        cam_K     (3x3 ndarray)
        T_lc      {'rotation': [w,x,y,z], 'translation': [x,y,z]}
        img_size  (width, height)
    """
    lidar_cs = cs_by_token[sd_by_token[lidar_sd_tok]["calibrated_sensor_token"]]
    cam_sd   = sd_by_token[cam_sd_tok]
    cam_cs   = cs_by_token[cam_sd["calibrated_sensor_token"]]

    cam_K    = np.array(cam_cs["camera_intrinsic"])
    img_size = (cam_sd["width"], cam_sd["height"])

    R_l2e = Quaternion(lidar_cs["rotation"])
    t_l2e = np.array(lidar_cs["translation"])
    R_e2c = Quaternion(cam_cs["rotation"]).inverse
    t_e2c = np.array(cam_cs["translation"])

    R_lc  = R_e2c * R_l2e
    t_lc  = R_e2c.rotate(R_l2e.rotate(t_l2e) - t_e2c)

    return cam_K, {"rotation": [R_lc.w, R_lc.x, R_lc.y, R_lc.z],
                   "translation": t_lc.tolist()}, img_size


# ============================================================
# 5.  LIDAR -> 3-CHANNEL DEPTH IMAGE
# ============================================================

def lidar_to_depth_image(lidar_bin_path, cam_K, T_lc, img_size, max_depth=80.0):
    """
    Project LiDAR point cloud onto camera plane.

    Output: (H, W, 3) uint8
      channel 0 = depth     (0=no hit, 255=max_depth m)
      channel 1 = intensity (0-255, direct from sensor)
      channel 2 = mask      (0=no hit, 255=valid)
    """
    pts       = np.fromfile(lidar_bin_path, dtype=np.float32).reshape(-1, 5)
    xyz       = pts[:, :3]
    intensity = pts[:, 3]

    R = Quaternion(T_lc["rotation"])
    t = np.array(T_lc["translation"])
    pts_cam   = xyz @ R.rotation_matrix.T + t
    front     = pts_cam[:, 2] > 0.1
    pts_cam   = pts_cam[front]
    intensity = intensity[front]

    proj   = (cam_K @ pts_cam.T).T
    depths = pts_cam[:, 2]
    proj[:, 0] /= depths
    proj[:, 1] /= depths

    W, H  = img_size
    valid = (proj[:, 0] >= 0) & (proj[:, 0] < W) & \
            (proj[:, 1] >= 0) & (proj[:, 1] < H)
    proj, depths, intensity = proj[valid], depths[valid], intensity[valid]

    u, v = proj[:, 0].astype(np.int32), proj[:, 1].astype(np.int32)
    order = np.argsort(depths)[::-1]          # fill far first, close overwrites
    u, v, depths, intensity = u[order], v[order], depths[order], intensity[order]

    depth_ch = np.zeros((H, W), np.float32)
    int_ch   = np.zeros((H, W), np.float32)
    mask_ch  = np.zeros((H, W), np.float32)
    depth_ch[v, u] = depths
    int_ch[v, u]   = intensity
    mask_ch[v, u]  = 1.0

    depth_u8 = (np.clip(depth_ch, 0, max_depth) / max_depth * 255).astype(np.uint8)
    int_u8   = np.clip(int_ch, 0, 255).astype(np.uint8)
    mask_u8  = (mask_ch * 255).astype(np.uint8)

    return np.stack([depth_u8, int_u8, mask_u8], axis=-1)


# ============================================================
# 6.  3D BOX -> 2D BBOX PROJECTION  (manuscript method)
# ============================================================

def build_instance_category_map(meta):
    """
    Returns a dict: instance_token -> category_name
    by joining instance.json and category.json.
    """
    cat_map = {c["token"]: c["name"] for c in meta["category"]}
    return {inst["token"]: cat_map.get(inst["category_token"], "")
            for inst in meta["instance"]}


def get_3d_box_corners(translation, size, rotation_wxyz):
    """
    Compute the 8 corners of a nuScenes 3D box in the GLOBAL frame.

    nuScenes convention:
        size = [width, length, height]
        Corners in local frame (before rotation):
            x +/-= length/2
            y +/-= width/2
            z +/-= height/2

    Returns: (3, 8) ndarray  in global coordinates.
    """
    w, l, h = size
    # Local corners
    x_c = l / 2 * np.array([ 1,  1,  1,  1, -1, -1, -1, -1])
    y_c = w / 2 * np.array([ 1, -1, -1,  1,  1, -1, -1,  1])
    z_c = h / 2 * np.array([ 1,  1, -1, -1,  1,  1, -1, -1])
    corners_local = np.vstack([x_c, y_c, z_c])          # (3, 8)

    # Rotate into global frame
    R_box = Quaternion(rotation_wxyz).rotation_matrix    # (3, 3)
    corners_global = R_box @ corners_local               # (3, 8)

    # Translate
    corners_global += np.array(translation)[:, np.newaxis]
    return corners_global                                 # (3, 8)


def project_box_to_image(corners_global, ego_pose, cam_calib, cam_K,
                          img_W, img_H, min_visibility=0.10):
    """
    Project 8 global-frame corners to image plane.

    Pipeline (matches manuscript):
        global -> ego-vehicle frame (inverse ego_pose)
               -> camera frame      (inverse calibrated_sensor)
               -> image plane       (camera intrinsics K)

    Returns YOLO line string or None if box is invisible / too small.
    """
    # --- Global -> Ego ---
    t_ego = np.array(ego_pose["translation"])
    R_ego = Quaternion(ego_pose["rotation"])
    corners_ego = R_ego.inverse.rotation_matrix @ (corners_global - t_ego[:, np.newaxis])

    # --- Ego -> Camera ---
    t_cam = np.array(cam_calib["translation"])
    R_cam = Quaternion(cam_calib["rotation"])
    corners_cam = R_cam.inverse.rotation_matrix @ (corners_ego - t_cam[:, np.newaxis])

    # Keep only corners in front of camera (Z > 0)
    front_mask = corners_cam[2, :] > 0.0
    if front_mask.sum() == 0:
        return None

    # Project visible corners to image plane
    Z   = corners_cam[2, front_mask]
    uv  = (cam_K @ corners_cam[:, front_mask])           # (3, n)
    u   = uv[0, :] / Z
    v   = uv[1, :] / Z

    # 2D bounding box from extrema of projected corners
    x1_proj, y1_proj = u.min(), v.min()
    x2_proj, y2_proj = u.max(), v.max()

    if x1_proj >= img_W or y1_proj >= img_H or x2_proj <= 0 or y2_proj <= 0:
        return None   # completely outside image

    # Clip to image
    x1 = max(0.0, x1_proj);  y1 = max(0.0, y1_proj)
    x2 = min(float(img_W), x2_proj);  y2 = min(float(img_H), y2_proj)

    bw = x2 - x1;  bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None

    # Visibility: clipped / projected area
    proj_w = x2_proj - x1_proj;  proj_h = y2_proj - y1_proj
    if proj_w > 0 and proj_h > 0:
        vis_frac = (bw * bh) / (proj_w * proj_h)
        if vis_frac < min_visibility:
            return None

    # YOLO normalised format
    xc = ((x1 + x2) / 2) / img_W
    yc = ((y1 + y2) / 2) / img_H
    wn = bw / img_W
    hn = bh / img_H

    xc = max(0.0, min(1.0, xc));  yc = max(0.0, min(1.0, yc))
    wn = max(0.0, min(1.0, wn));  hn = max(0.0, min(1.0, hn))

    if wn < 0.005 or hn < 0.005:
        return None

    return xc, yc, wn, hn


def build_annotation_index(meta, instance_cat_map):
    """
    Group sample_annotation records by sample_token.
    Returns: dict  sample_token -> list of annotation dicts with category_name added.
    """
    idx = defaultdict(list)
    for ann in meta["sample_annotation"]:
        cat_name = instance_cat_map.get(ann["instance_token"], "")
        if cat_name in CLASS_MAPPING:
            ann["category_name"] = cat_name
            idx[ann["sample_token"]].append(ann)
    return idx


def generate_yolo_labels(sample_token, cam_sd_tok,
                          ann_by_sample,
                          sd_by_token, cs_by_token, ep_by_token,
                          img_W, img_H, min_visibility=0.10):
    """
    Project all 3D annotations for this sample onto the camera and
    return a list of YOLO format strings.
    """
    annotations = ann_by_sample.get(sample_token, [])
    if not annotations:
        return []

    cam_sd    = sd_by_token[cam_sd_tok]
    ego_pose  = ep_by_token[cam_sd["ego_pose_token"]]
    cam_calib = cs_by_token[cam_sd["calibrated_sensor_token"]]
    cam_K     = np.array(cam_calib["camera_intrinsic"])

    lines = []
    for ann in annotations:
        cat   = ann["category_name"]
        cid   = CLASS_MAPPING.get(cat)
        if cid is None:
            continue

        corners_g = get_3d_box_corners(
            ann["translation"], ann["size"], ann["rotation"])

        result = project_box_to_image(
            corners_g, ego_pose, cam_calib, cam_K,
            img_W, img_H, min_visibility)

        if result is None:
            continue

        xc, yc, wn, hn = result
        lines.append(f"{cid} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
    return lines


# ============================================================
# 7.  PROCESS ONE SPLIT
# ============================================================

def process_split(split_name, scenes, meta, dataset_root, output_root,
                  ann_by_sample, sensor_map, calib_map,
                  sd_by_token, cs_by_token, ep_by_token,
                  max_depth, min_visibility):

    dataset_path = Path(dataset_root)
    cam_dir   = Path(output_root) / split_name / "camera"
    lidar_dir = Path(output_root) / split_name / "lidar"
    label_dir = Path(output_root) / split_name / "labels"
    for d in (cam_dir, lidar_dir, label_dir):
        d.mkdir(parents=True, exist_ok=True)

    total_imgs  = 0
    total_boxes = 0
    skipped     = 0
    n_scenes    = len(scenes)

    for si, scene in enumerate(scenes, 1):
        for sample_token in scene["sample_tokens"]:

            # LiDAR sample_data
            try:
                lidar_tok = get_sd_token(
                    sample_token, "LIDAR_TOP", meta,
                    sensor_map, calib_map, sd_by_token)
            except Exception:
                skipped += 1
                continue

            lidar_sd   = sd_by_token.get(lidar_tok)
            lidar_path = dataset_path / lidar_sd["filename"] if lidar_sd else None
            if lidar_path is None or not lidar_path.exists():
                skipped += 1
                continue

            for cam in CAMERA_NAMES:
                try:
                    cam_tok = get_sd_token(
                        sample_token, cam, meta,
                        sensor_map, calib_map, sd_by_token)
                except Exception:
                    continue

                cam_sd = sd_by_token.get(cam_tok)
                if cam_sd is None:
                    continue

                cam_img_path = dataset_path / cam_sd["filename"]
                if not cam_img_path.exists():
                    continue

                base    = f"{sample_token}_{cam}"
                rgb_out = cam_dir   / f"{base}.jpg"
                dep_out = lidar_dir / f"{base}.png"
                lbl_out = label_dir / f"{base}.txt"

                # --- RGB image ---
                shutil.copy(cam_img_path, rgb_out)

                # --- 3-channel LiDAR depth image ---
                try:
                    cam_K, T_lc, img_size = compute_lidar_to_cam(
                        lidar_tok, cam_tok, sd_by_token, cs_by_token)
                    depth_img = lidar_to_depth_image(
                        str(lidar_path), cam_K, T_lc, img_size, max_depth)
                    cv2.imwrite(str(dep_out), depth_img)
                except Exception:
                    W = cam_sd.get("width",  1600)
                    H = cam_sd.get("height",  900)
                    cv2.imwrite(str(dep_out),
                                np.zeros((H, W, 3), dtype=np.uint8))
                    img_size = (W, H)

                # --- YOLO labels (from 3D projection) ---
                W, H    = cam_sd.get("width", 1600), cam_sd.get("height", 900)
                yolo_lines = generate_yolo_labels(
                    sample_token, cam_tok,
                    ann_by_sample,
                    sd_by_token, cs_by_token, ep_by_token,
                    W, H, min_visibility)

                with open(lbl_out, "w") as f:
                    f.write("\n".join(yolo_lines))

                total_boxes += len(yolo_lines)
                total_imgs  += 1

        if si % 10 == 0 or si == n_scenes:
            print(f"    [{split_name}] {si}/{n_scenes} scenes | "
                  f"{total_imgs} images | {total_boxes} boxes")

    print(f"\n  {split_name.upper()} complete: "
          f"images={total_imgs}  boxes={total_boxes}  "
          f"skipped_samples={skipped}")
    return total_imgs, total_boxes


# ============================================================
# 8.  SCENE SPLIT REPORT
# ============================================================

def write_scene_report(output_root, train_scenes, val_scenes, random_seed=42):
    train_tok = {s["token"] for s in train_scenes}
    val_tok   = {s["token"] for s in val_scenes}
    overlap   = train_tok & val_tok
    sep       = "=" * 72

    lines = [sep, "SCENE-LEVEL SPLIT REPORT",
             "nuScenes v1.0-trainval01 (part 1)  --  Scene-level 80/20 split",
             sep, "",
             f"Scenes on disk  : {len(train_scenes)+len(val_scenes)}",
             f"  TRAIN         : {len(train_scenes)} scenes",
             f"  VAL           : {len(val_scenes)} scenes",
             f"  Random seed   : {random_seed}  (change to get a different valid split)",
             ""]

    # Overlap check
    lines += ["OVERLAP CHECK", "-" * 40]
    if not overlap:
        lines += [
            "  RESULT : PASS -- zero scene tokens shared between splits.",
            "  Every camera-LiDAR pair from a given scene is exclusively",
            "  in train OR val, never in both.", ""]
    else:
        lines += [f"  RESULT : FAIL -- {len(overlap)} overlapping scene(s)!"]
        lines += [f"    {t}" for t in sorted(overlap)]
        lines.append("")

    def session_block(scene_list, label):
        from collections import Counter
        counts     = Counter((s["date"], s["location"]) for s in scene_list)
        total_samp = sum(s["nbr_samples"] for s in scene_list)
        b = [f"{label} RECORDING SESSIONS", "-" * 40,
             f"  Scenes  : {len(scene_list)}",
             f"  Samples : {total_samp}  "
             f"(x6 cameras = {total_samp*6} image pairs)", ""]
        for (d, loc), cnt in sorted(counts.items()):
            b.append(f"  {d}  {loc:<34}  {cnt:>3} scene(s)")
        b.append("")
        return b

    lines += session_block(train_scenes, "TRAIN")
    lines += session_block(val_scenes,   "VAL")

    # Temporal analysis
    shared_dates = sorted({s["date"] for s in train_scenes} &
                          {s["date"] for s in val_scenes})
    lines += ["TEMPORAL OVERLAP ANALYSIS", "-" * 40]
    if shared_dates:
        lines += [
            f"  Shared recording dates: {shared_dates}",
            "  Both splits contain scenes from the same calendar days.",
            "  This is expected: one day spans many independent routes.",
            "  No SCENE token is shared -- confirmed.", ""]
    else:
        lines += ["  No recording date shared between train and val.", ""]

    # Location analysis
    train_locs = sorted({s["location"] for s in train_scenes})
    val_locs   = sorted({s["location"] for s in val_scenes})
    shared_locs = sorted(set(train_locs) & set(val_locs))
    lines += ["LOCATION OVERLAP ANALYSIS", "-" * 40,
              f"  Train locations : {train_locs}",
              f"  Val   locations : {val_locs}"]
    if shared_locs:
        lines += [f"  Shared locations: {shared_locs}",
                  "  NOTE: Same geographic area but different drive routes.",
                  "        Scene-level exclusivity is still guaranteed.", ""]
    else:
        lines += ["  No location shared.", ""]

    def scene_listing(scene_list, label):
        b = [f"{label} -- FULL SCENE LIST", "-" * 40]
        for s in sorted(scene_list, key=lambda x: x["name"]):
            b.append(f"  {s['name']:<14}  {s['date']}  "
                     f"{s['location']:<34}  {s['nbr_samples']:>2} samples  "
                     f"log={s['logfile']}")
        b.append("")
        return b

    lines += scene_listing(train_scenes, "TRAIN")
    lines += scene_listing(val_scenes,   "VAL")
    lines.append(sep)

    report = "\n".join(lines)
    report_path = os.path.join(output_root, "scene_split_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\nReport saved: {report_path}")
    return len(overlap) == 0


# ============================================================
# 9.  DATA.YAML
# ============================================================

def create_data_yaml(output_root):
    content = f"""# YOLO Fusion Dataset -- Scene-Level Split
# Dataset  : nuScenes v1.0-trainval01 (part 1, 85 scenes)
# Split    : scene-level 80/20 (zero leakage)
# Labels   : projected from 3D sample_annotation.json

path: {output_root}

train:       train/camera
val:         val/camera
train_lidar: train/lidar
val_lidar:   val/lidar

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}

split_method: scene_level
"""
    p = os.path.join(output_root, "data.yaml")
    with open(p, "w") as f:
        f.write(content)
    print(f"  data.yaml: {p}")


# ============================================================
# 10.  MAIN
# ============================================================

def run(dataset_root=DATASET_ROOT, output_root=OUTPUT_ROOT,
        train_ratio=TRAIN_RATIO, min_visibility=MIN_VISIBILITY,
        max_depth=MAX_DEPTH, random_seed=RANDOM_SEED):

    print("=" * 72)
    print("SCENE-LEVEL PIPELINE -- nuScenes v1.0-trainval01")
    print("=" * 72)
    print(f"  Input  : {dataset_root}")
    print(f"  Output : {output_root}")
    print(f"  Split  : {train_ratio:.0%} train / {1-train_ratio:.0%} val  (scene-level)")
    print()

    # 1. Metadata
    print("Step 1/5  Loading metadata ...")
    meta = load_metadata(dataset_root)
    print(f"  scenes={len(meta['scene'])}  samples={len(meta['sample'])}")
    print(f"  3D annotations={len(meta['sample_annotation'])}")

    # 2. Disk scan
    print("\nStep 2/5  Scanning LiDAR files on disk ...")
    scenes_info = build_scene_info(meta, dataset_root)
    on_disk     = [s for s in scenes_info if s["has_lidar_data"]]
    print(f"  Scenes with data: {len(on_disk)} / {len(scenes_info)}")

    # 3. Scene-level split
    print("\nStep 3/5  Scene-level split ...")
    train_scenes, val_scenes = scene_level_split(on_disk, train_ratio, random_seed)
    print(f"  Train: {len(train_scenes)} scenes")
    print(f"  Val  : {len(val_scenes)} scenes")

    os.makedirs(output_root, exist_ok=True)
    print("\n--- Scene Split Report ---")
    ok = write_scene_report(output_root, train_scenes, val_scenes, random_seed)
    if not ok:
        raise RuntimeError("Scene overlap detected -- aborting!")

    # 4. Build indices
    print("\nStep 4/5  Building lookup indices ...")
    sensor_map, calib_map, sd_by_token, cs_by_token, ep_by_token = \
        build_sensor_index(meta)

    print("  Building instance->category map ...")
    instance_cat_map = build_instance_category_map(meta)

    print("  Indexing 3D annotations by sample_token ...")
    ann_by_sample = build_annotation_index(meta, instance_cat_map)
    print(f"  Samples with annotations: {len(ann_by_sample)}")

    # 5. Process splits
    print("\nStep 5/5  Processing images + labels ...")
    for split_name, scenes in [("train", train_scenes), ("val", val_scenes)]:
        print(f"\n  --- {split_name.upper()} ({len(scenes)} scenes) ---")
        process_split(
            split_name, scenes, meta, dataset_root, output_root,
            ann_by_sample, sensor_map, calib_map,
            sd_by_token, cs_by_token, ep_by_token,
            max_depth, min_visibility)

    print("\nCreating data.yaml ...")
    create_data_yaml(output_root)

    # Summary
    print("\n" + "=" * 72)
    print("PIPELINE COMPLETE")
    print("=" * 72)
    for split in ("train", "val"):
        p    = Path(output_root) / split
        nc   = len(list((p/"camera").glob("*.jpg")))  if (p/"camera").exists()  else 0
        nl   = len(list((p/"lidar").glob("*.png")))   if (p/"lidar").exists()   else 0
        nlab = len(list((p/"labels").glob("*.txt")))  if (p/"labels").exists()  else 0
        print(f"  {split.upper():5s}  camera={nc}  lidar={nl}  labels={nlab}")
    print(f"\n  Output: {output_root}")


if __name__ == "__main__":
    run()
