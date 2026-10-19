# multimodal_dataset.py
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np


class MultiModalYoloDataset(Dataset):
    def __init__(self, root, split="train", img_size=640, cache_images=False, augment=True):
        """
        root: path to dataset folder
        split: 'train' or 'val'
        img_size: target image size
        cache_images: if True, cache all images in memory (uses lots of RAM but much faster)
        augment: if True, apply data augmentation (only for training)
        """
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.cache_images = cache_images
        self.augment = augment and (split == "train")  # Only augment training data

        self.cam_dir = self.root / split / "camera"
        self.lidar_dir = self.root / split /  "lidar"
        self.label_dir = self.root /  split /"labels"

        # Pre-build all file paths during initialization
        print(f"Building file index for {split} split...")
        cam_files = {p.stem: p for p in self.cam_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]}
        lidar_files = {p.stem: p for p in self.lidar_dir.glob("*.*") if p.suffix.lower() in [".jpg", ".png", ".jpeg"]}

        # Only keep stems that have both camera and lidar
        common_stems = sorted(set(cam_files.keys()) & set(lidar_files.keys()))

        self.samples = []
        for stem in common_stems:
            self.samples.append({
                'stem': stem,
                'cam_path': cam_files[stem],
                'lidar_path': lidar_files[stem],
                'label_path': self.label_dir / f"{stem}.txt"
            })
        print(f"Found {len(self.samples)} valid samples")

        # Optional: cache all images in memory
        self.cache = {}
        if self.cache_images:
            print("Caching images in memory (this may take a while)...")
            for idx in range(len(self.samples)):
                if idx % 1000 == 0:
                    print(f"  Caching progress: {idx}/{len(self.samples)}")
                sample = self.samples[idx]
                self.cache[idx] = {
                    'cam': self.load_image(sample['cam_path']),
                    'lidar': self.load_lidar(sample['lidar_path']),
                    'labels': self.load_labels(sample['label_path'])
                }
            print(f"Cached {len(self.cache)} samples")

    def __len__(self):
        return len(self.samples)

    def apply_augmentation(self, img_cam, img_lidar, labels):
        """
        Apply data augmentation to camera, lidar images and labels.

        Args:
            img_cam: Camera image tensor [C, H, W] (already normalized 0-1)
            img_lidar: LiDAR image tensor [C, H, W] (already normalized 0-1)
            labels: Label tensor [N, 5] with [class, x, y, w, h]

        Returns:
            Augmented img_cam, img_lidar, labels
        """
        # Convert to numpy for easier manipulation [H, W, C]
        img_cam_np = img_cam.permute(1, 2, 0).numpy()
        img_lidar_np = img_lidar.permute(1, 2, 0).numpy()

        # 1. Horizontal Flip (50% chance)
        if np.random.random() < 0.5:
            img_cam_np = np.fliplr(img_cam_np).copy()
            img_lidar_np = np.fliplr(img_lidar_np).copy()
            if len(labels) > 0:
                labels = labels.clone()
                labels[:, 1] = 1 - labels[:, 1]  # Flip x coordinate

        # 2. Random Brightness & Contrast (camera only, REDUCED strength)
        if np.random.random() < 0.3:  # Reduced from 0.7 to 0.3
            brightness = np.random.uniform(0.9, 1.1)  # Reduced from 0.8-1.2
            contrast = np.random.uniform(0.9, 1.1)    # Reduced from 0.8-1.2
            img_cam_np = np.clip(img_cam_np * contrast + (brightness - 1.0) * 0.1, 0, 1)

        # 3. Random Color Jitter (camera only, REDUCED probability)
        if np.random.random() < 0.2:  # Reduced from 0.5 to 0.2
            # Slightly adjust each channel independently
            for c in range(3):
                factor = np.random.uniform(0.95, 1.05)  # Reduced from 0.9-1.1
                img_cam_np[:, :, c] = np.clip(img_cam_np[:, :, c] * factor, 0, 1)

        # 4. Add Gaussian Noise - DISABLED (was too aggressive for multimodal fusion)
        # if np.random.random() < 0.1:  # Reduced from 0.3, now disabled
        #     noise_cam = np.random.normal(0, 0.01, img_cam_np.shape)  # Reduced from 0.02
        #     img_cam_np = np.clip(img_cam_np + noise_cam, 0, 1)
        #     # LiDAR noise removed - corrupts sparse point clouds

        # Convert back to tensor [C, H, W]
        img_cam = torch.from_numpy(np.transpose(img_cam_np, (2, 0, 1))).float()
        img_lidar = torch.from_numpy(np.transpose(img_lidar_np, (2, 0, 1))).float()

        return img_cam, img_lidar, labels

    def random_affine(self, img_cam, img_lidar, labels=(), degrees=10, translate=.1, scale=.1, shear=10):
        # Applies random affine transformations to camera, lidar images and labels
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10))
        # https://github.com/ultralytics/yolov5/blob/master/utils/augmentations.py

        height = self.img_size
        width = self.img_size

        # Center
        C = np.eye(3)
        C[0, 2] = -width / 2  # x translation (pixels)
        C[1, 2] = -height / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)

        # Rotation and Scale
        R = np.eye(3)
        a = np.random.uniform(-degrees, degrees)
        s = np.random.uniform(1 - scale, 1 + scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = np.math.tan(np.random.uniform(-shear, shear) * np.math.pi / 180)  # x shear (deg)
        S[1, 0] = np.math.tan(np.random.uniform(-shear, shear) * np.math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = np.random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation (pixels)
        T[1, 2] = np.random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT

        # Apply the transformation
        if M[2, :].any(): # For perspective
             img_cam = cv2.warpPerspective(img_cam, M, dsize=(width, height), borderValue=(114, 114, 114))
             img_lidar = cv2.warpPerspective(img_lidar, M, dsize=(width, height), borderValue=(0, 0, 0)) # Black border for lidar
        else: # Affine
             img_cam = cv2.warpAffine(img_cam, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
             img_lidar = cv2.warpAffine(img_lidar, M[:2], dsize=(width, height), borderValue=(0, 0, 0))

        # Transform labels
        n = len(labels)
        if n:
            new_labels = labels.clone()
            xy = np.ones((n * 4, 3))
            xy[:, :2] = new_labels[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if M[2, :].any() else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

            # filter candidates
            i = ((new[:, 2] - new[:, 0]) > 4) & ((new[:, 3] - new[:, 1]) > 4)
            new_labels = new_labels[i]
            new_labels[:, 1:5] = new[i]

            # Convert from xyxy to xywh
            new_labels[:, 1] = (new_labels[:, 1] + new_labels[:, 3]) / 2
            new_labels[:, 2] = (new_labels[:, 2] + new_labels[:, 4]) / 2
            new_labels[:, 3] = new_labels[:, 3] - new_labels[:, 1]
            new_labels[:, 4] = new_labels[:, 4] - new_labels[:, 2]
            labels = new_labels

        return img_cam, img_lidar, labels


    def load_image(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # [C,H,W]
        return torch.from_numpy(img)


    def load_lidar(self, path):
        # treat lidar as 3-channel RGB-like image (Visualized dots on black background)
        lidar = cv2.imread(str(path), cv2.IMREAD_COLOR)  # Load BGR, defaults to 8-bit
        if lidar is None:
            raise FileNotFoundError(path)
            
        lidar = cv2.cvtColor(lidar, cv2.COLOR_BGR2RGB)
        lidar = cv2.resize(lidar, (self.img_size, self.img_size))
        
        # Standard 8-bit normalization
        lidar = lidar.astype(np.float32) / 255.0
        
        lidar = np.transpose(lidar, (2, 0, 1))  # [3, H, W]
        return torch.from_numpy(lidar)



    def load_labels(self, path):
        # returns tensor [N, 5]: [class, x, y, w, h] (normalized)
        if not path.exists():
            return torch.zeros((0, 5), dtype=torch.float32)
        labels = []
        with open(path, "r") as f:
            for line in f.readlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                cls = int(parts[0])
                x, y, w, h = map(float, parts[1:])
                labels.append([cls, x, y, w, h])
        if len(labels) == 0:
            return torch.zeros((0, 5), dtype=torch.float32)
        return torch.tensor(labels, dtype=torch.float32)

    def __getitem__(self, idx):
        # Use cached data if available
        if self.cache_images and idx in self.cache:
            cached = self.cache[idx]
            img_cam, img_lidar, labels = cached['cam'], cached['lidar'], cached['labels']

            # Apply augmentation if enabled (even for cached data)
            if self.augment:
                img_cam, img_lidar, labels = self.apply_augmentation(img_cam, img_lidar, labels)

            return img_cam, img_lidar, labels, self.samples[idx]['stem']

        # Otherwise load from disk
        sample = self.samples[idx]

        try:
            img_cam = self.load_image(sample['cam_path'])     # [3, H, W]
            img_lidar = self.load_lidar(sample['lidar_path']) # [3, H, W]
            labels = self.load_labels(sample['label_path'])   # [N,5]
        except Exception as e:
            print(f"Error loading sample {idx} ({sample['stem']}): {e}")
            raise

        # Apply augmentation if enabled
        if self.augment:
            img_cam, img_lidar, labels = self.apply_augmentation(img_cam, img_lidar, labels)

        return img_cam, img_lidar, labels, sample['stem']
