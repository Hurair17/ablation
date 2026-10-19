# multimodal_batch_augmentation.py
"""
Batch-level augmentations (Mosaic, Mixup, CopyPaste) for multimodal YOLO training.
These augmentations require combining multiple samples, so they're applied during collation.
"""

import torch
import numpy as np
import random
import cv2
from typing import List, Tuple


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Resize and pad image to new_shape while maintaining aspect ratio.

    Args:
        img: Image array [H, W, C]
        new_shape: Target size (height, width)
        color: Padding color

    Returns:
        Resized and padded image
    """
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, r, (dw, dh)


def mosaic_augmentation(samples: List[Tuple], img_size=640, enable=True, prob=1.0):
    """
    Mosaic augmentation: combine 4 images into one mosaic.
    Works for multimodal data (camera + LiDAR + labels).

    Args:
        samples: List of (img_cam, img_lidar, labels, stem) tuples
        img_size: Target image size
        enable: Whether mosaic is enabled
        prob: Probability of applying mosaic

    Returns:
        Mosaic image (cam), mosaic image (lidar), combined labels, stem
    """
    if not enable or random.random() > prob or len(samples) < 4:
        # Return first sample without mosaic
        return samples[0]

    # Select 4 random samples
    indices = random.sample(range(len(samples)), 4)
    mosaic_samples = [samples[i] for i in indices]

    # Create mosaic center point
    yc, xc = [int(random.uniform(img_size * 0.5, img_size * 1.5)) for _ in range(2)]

    # Initialize mosaic images
    mosaic_cam = np.full((img_size * 2, img_size * 2, 3), 114, dtype=np.float32)
    mosaic_lidar = np.full((img_size * 2, img_size * 2, 3), 0, dtype=np.float32)  # Black for LiDAR
    mosaic_labels = []

    # Place 4 images in mosaic
    for i, (img_cam, img_lidar, labels, stem) in enumerate(mosaic_samples):
        # Convert tensors to numpy [C, H, W] -> [H, W, C]
        img_cam_np = img_cam.permute(1, 2, 0).numpy()
        img_lidar_np = img_lidar.permute(1, 2, 0).numpy()
        h, w = img_cam_np.shape[:2]

        # Define placement (top-left, top-right, bottom-left, bottom-right)
        if i == 0:  # top-left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top-right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, img_size * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom-left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(img_size * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom-right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, img_size * 2), min(img_size * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        # Place image patches
        mosaic_cam[y1a:y2a, x1a:x2a] = img_cam_np[y1b:y2b, x1b:x2b]
        mosaic_lidar[y1a:y2a, x1a:x2a] = img_lidar_np[y1b:y2b, x1b:x2b]

        # Adjust labels
        if len(labels) > 0:
            labels_copy = labels.clone()
            # Convert normalized coords to mosaic coords
            labels_copy[:, 1] = labels[:, 1] * w + (x1a - x1b)  # x center
            labels_copy[:, 2] = labels[:, 2] * h + (y1a - y1b)  # y center
            labels_copy[:, 3] = labels[:, 3] * w  # width
            labels_copy[:, 4] = labels[:, 4] * h  # height
            mosaic_labels.append(labels_copy)

    # Crop mosaic to img_size and ensure exact dimensions
    mosaic_cam = mosaic_cam[0:img_size, 0:img_size]
    mosaic_lidar = mosaic_lidar[0:img_size, 0:img_size]

    # Ensure exact dimensions with resize (fixes C2PSA dimension mismatch)
    if mosaic_cam.shape[0] != img_size or mosaic_cam.shape[1] != img_size:
        mosaic_cam = cv2.resize(mosaic_cam, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    if mosaic_lidar.shape[0] != img_size or mosaic_lidar.shape[1] != img_size:
        mosaic_lidar = cv2.resize(mosaic_lidar, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    # Normalize labels back to [0, 1]
    if len(mosaic_labels) > 0:
        mosaic_labels = torch.cat(mosaic_labels, 0)
        # Clip to image bounds
        mosaic_labels[:, 1] = torch.clamp(mosaic_labels[:, 1], 0, img_size)
        mosaic_labels[:, 2] = torch.clamp(mosaic_labels[:, 2], 0, img_size)

        # Normalize to [0, 1]
        mosaic_labels[:, 1] = mosaic_labels[:, 1] / img_size  # x
        mosaic_labels[:, 2] = mosaic_labels[:, 2] / img_size  # y
        mosaic_labels[:, 3] = mosaic_labels[:, 3] / img_size  # w
        mosaic_labels[:, 4] = mosaic_labels[:, 4] / img_size  # h

        # Filter out invalid boxes (too small or out of bounds)
        valid_mask = (mosaic_labels[:, 3] > 0.01) & (mosaic_labels[:, 4] > 0.01)
        mosaic_labels = mosaic_labels[valid_mask]
    else:
        mosaic_labels = torch.zeros((0, 5), dtype=torch.float32)

    # Convert back to tensor [H, W, C] -> [C, H, W]
    mosaic_cam_tensor = torch.from_numpy(np.transpose(mosaic_cam, (2, 0, 1))).float()
    mosaic_lidar_tensor = torch.from_numpy(np.transpose(mosaic_lidar, (2, 0, 1))).float()

    return mosaic_cam_tensor, mosaic_lidar_tensor, mosaic_labels, "mosaic_" + mosaic_samples[0][3]


def mixup_augmentation(sample1: Tuple, sample2: Tuple, alpha=0.5, enable=True, prob=0.15):
    """
    Mixup augmentation: blend two images together.

    Args:
        sample1: First sample (img_cam, img_lidar, labels, stem)
        sample2: Second sample
        alpha: Blending factor (0.5 = 50/50 mix)
        enable: Whether mixup is enabled
        prob: Probability of applying mixup

    Returns:
        Mixed sample
    """
    if not enable or random.random() > prob:
        return sample1

    img_cam1, img_lidar1, labels1, stem1 = sample1
    img_cam2, img_lidar2, labels2, stem2 = sample2

    # Random mixing ratio
    r = np.random.beta(alpha, alpha)

    # Mix images
    mixed_cam = img_cam1 * r + img_cam2 * (1 - r)
    mixed_lidar = img_lidar1 * r + img_lidar2 * (1 - r)

    # Combine labels
    if len(labels1) > 0 and len(labels2) > 0:
        mixed_labels = torch.cat([labels1, labels2], 0)
    elif len(labels1) > 0:
        mixed_labels = labels1
    elif len(labels2) > 0:
        mixed_labels = labels2
    else:
        mixed_labels = torch.zeros((0, 5), dtype=torch.float32)

    return mixed_cam, mixed_lidar, mixed_labels, f"mixup_{stem1}_{stem2}"


def copy_paste_augmentation(sample: Tuple, paste_samples: List[Tuple], enable=True, prob=0.1, max_paste=3):
    """
    Copy-Paste augmentation: paste objects from other images onto current image.

    Args:
        sample: Main sample (img_cam, img_lidar, labels, stem)
        paste_samples: List of samples to paste from
        enable: Whether copy-paste is enabled
        prob: Probability of applying copy-paste
        max_paste: Maximum number of objects to paste

    Returns:
        Augmented sample
    """
    if not enable or random.random() > prob or len(paste_samples) == 0:
        return sample

    img_cam, img_lidar, labels, stem = sample
    img_cam_np = img_cam.permute(1, 2, 0).numpy()
    img_lidar_np = img_lidar.permute(1, 2, 0).numpy()
    h, w = img_cam_np.shape[:2]

    new_labels = [labels] if len(labels) > 0 else []

    # Randomly select objects to paste
    num_paste = random.randint(1, min(max_paste, len(paste_samples)))
    paste_indices = random.sample(range(len(paste_samples)), num_paste)

    for paste_idx in paste_indices:
        paste_cam, paste_lidar, paste_labels, _ = paste_samples[paste_idx]

        if len(paste_labels) == 0:
            continue

        # Convert to numpy
        paste_cam_np = paste_cam.permute(1, 2, 0).numpy()
        paste_lidar_np = paste_lidar.permute(1, 2, 0).numpy()
        paste_h, paste_w = paste_cam_np.shape[:2]

        # Randomly select one object from paste image
        obj_idx = random.randint(0, len(paste_labels) - 1)
        obj_label = paste_labels[obj_idx]

        # Extract object bounding box in paste image
        cx, cy, bw, bh = obj_label[1:].numpy()
        x1 = max(0, int((cx - bw/2) * paste_w))
        y1 = max(0, int((cy - bh/2) * paste_h))
        x2 = min(paste_w, int((cx + bw/2) * paste_w))
        y2 = min(paste_h, int((cy + bh/2) * paste_h))

        if x2 <= x1 or y2 <= y1:
            continue

        # Extract object patches
        obj_cam = paste_cam_np[y1:y2, x1:x2]
        obj_lidar = paste_lidar_np[y1:y2, x1:x2]
        obj_h, obj_w = obj_cam.shape[:2]

        # Random position in target image
        paste_x = random.randint(0, max(0, w - obj_w))
        paste_y = random.randint(0, max(0, h - obj_h))

        # Paste object (simple overlay, no blending)
        img_cam_np[paste_y:paste_y+obj_h, paste_x:paste_x+obj_w] = obj_cam
        img_lidar_np[paste_y:paste_y+obj_h, paste_x:paste_x+obj_w] = obj_lidar

        # Create new label for pasted object
        new_cx = (paste_x + obj_w/2) / w
        new_cy = (paste_y + obj_h/2) / h
        new_w = obj_w / w
        new_h = obj_h / h

        new_label = torch.tensor([[obj_label[0].item(), new_cx, new_cy, new_w, new_h]], dtype=torch.float32)
        new_labels.append(new_label)

    # Combine all labels
    if len(new_labels) > 0:
        combined_labels = torch.cat(new_labels, 0)
    else:
        combined_labels = torch.zeros((0, 5), dtype=torch.float32)

    # Convert back to tensor
    img_cam_tensor = torch.from_numpy(np.transpose(img_cam_np, (2, 0, 1))).float()
    img_lidar_tensor = torch.from_numpy(np.transpose(img_lidar_np, (2, 0, 1))).float()

    return img_cam_tensor, img_lidar_tensor, combined_labels, f"copypaste_{stem}"


class MultimodalBatchAugmentedCollate:
    """
    Custom collate function that applies batch-level augmentations.
    """
    def __init__(self, img_size=640, mosaic_prob=1.0, mixup_prob=0.15,
                 copy_paste_prob=0.1, enable_mosaic=True, enable_mixup=True,
                 enable_copy_paste=True, close_mosaic_epoch=None, current_epoch=0):
        """
        Initialize batch augmentation collate function.

        Args:
            img_size: Target image size
            mosaic_prob: Probability of mosaic augmentation
            mixup_prob: Probability of mixup augmentation
            copy_paste_prob: Probability of copy-paste augmentation
            enable_mosaic: Enable mosaic
            enable_mixup: Enable mixup
            enable_copy_paste: Enable copy-paste
            close_mosaic_epoch: Disable mosaic after this epoch
            current_epoch: Current training epoch
        """
        self.img_size = img_size
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.copy_paste_prob = copy_paste_prob
        self.enable_mosaic = enable_mosaic
        self.enable_mixup = enable_mixup
        self.enable_copy_paste = enable_copy_paste
        self.close_mosaic_epoch = close_mosaic_epoch
        self.current_epoch = current_epoch

    def update_epoch(self, epoch):
        """Update current epoch (used to disable mosaic at end of training)"""
        self.current_epoch = epoch

    def __call__(self, batch):
        """
        Apply batch-level augmentations and collate.

        Args:
            batch: List of (img_cam, img_lidar, labels, stem) tuples

        Returns:
            Collated batch with augmentations applied
        """
        # Check if we should disable mosaic
        mosaic_enabled = self.enable_mosaic
        if self.close_mosaic_epoch is not None and self.current_epoch >= self.close_mosaic_epoch:
            mosaic_enabled = False

        augmented_batch = []

        for i, sample in enumerate(batch):
            # Apply mosaic (requires 4 samples)
            if mosaic_enabled and len(batch) >= 4:
                sample = mosaic_augmentation(
                    batch,
                    img_size=self.img_size,
                    enable=True,
                    prob=self.mosaic_prob
                )

            # Apply mixup (blend with another random sample)
            if self.enable_mixup and len(batch) > 1:
                other_idx = random.randint(0, len(batch) - 1)
                if other_idx != i:
                    sample = mixup_augmentation(
                        sample,
                        batch[other_idx],
                        enable=True,
                        prob=self.mixup_prob
                    )

            # Apply copy-paste (paste objects from other samples)
            if self.enable_copy_paste and len(batch) > 1:
                paste_samples = [batch[j] for j in range(len(batch)) if j != i]
                sample = copy_paste_augmentation(
                    sample,
                    paste_samples,
                    enable=True,
                    prob=self.copy_paste_prob
                )

            augmented_batch.append(sample)

        # Standard collation
        imgs_cam, imgs_lidar, labels, stems = zip(*augmented_batch)
        return list(imgs_cam), list(imgs_lidar), list(labels), list(stems)


def create_augmented_dataloader(dataset, batch_size, img_size=640, mosaic_prob=1.0,
                                mixup_prob=0.15, copy_paste_prob=0.1,
                                enable_mosaic=True, enable_mixup=True, enable_copy_paste=True,
                                close_mosaic_epoch=None, current_epoch=0, **loader_kwargs):
    """
    Create DataLoader with batch-level augmentations.

    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        img_size: Image size
        mosaic_prob: Mosaic probability
        mixup_prob: Mixup probability
        copy_paste_prob: Copy-paste probability
        enable_mosaic: Enable mosaic
        enable_mixup: Enable mixup
        enable_copy_paste: Enable copy-paste
        close_mosaic_epoch: Epoch to disable mosaic
        current_epoch: Current epoch
        **loader_kwargs: Additional DataLoader arguments

    Returns:
        DataLoader with augmented collate function
    """
    from torch.utils.data import DataLoader

    collate_fn = MultimodalBatchAugmentedCollate(
        img_size=img_size,
        mosaic_prob=mosaic_prob,
        mixup_prob=mixup_prob,
        copy_paste_prob=copy_paste_prob,
        enable_mosaic=enable_mosaic,
        enable_mixup=enable_mixup,
        enable_copy_paste=enable_copy_paste,
        close_mosaic_epoch=close_mosaic_epoch,
        current_epoch=current_epoch
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        **loader_kwargs
    ), collate_fn
