"""
Custom sampler for oversampling minority classes in object detection.
"""
import torch
from torch.utils.data import Sampler
from collections import defaultdict
import numpy as np
from pathlib import Path


class ClassBalancedSampler(Sampler):
    """
    Sampler that oversamples images containing minority classes.

    Strategy:
    - Images with minority classes get sampled more frequently
    - Balances class distribution without creating duplicate images in memory
    - Works with YOLO-style datasets where one image can have multiple classes

    Args:
        dataset: Dataset object with __getitem__ that returns labels
        oversample_threshold: Classes with < threshold * mean_count are oversampled
        balance_mode: 'sqrt' (gentle) or 'full' (aggressive)
    """

    def __init__(self, dataset, data_root, split="train", oversample_threshold=0.5, balance_mode='sqrt'):
        self.dataset = dataset
        self.data_root = Path(data_root)
        self.split = split
        self.oversample_threshold = oversample_threshold
        self.balance_mode = balance_mode

        # Analyze class distribution from label files
        self.class_distribution = self._analyze_labels()

        # Compute sample weights
        self.sample_weights = self._compute_sample_weights()

        # Total number of samples to draw (increased by oversampling)
        # Multiply by average weight to increase epoch size
        avg_weight = np.mean(self.sample_weights)
        self.num_samples = int(len(self.dataset) * avg_weight)

        print(f"\n{'='*70}")
        print(f"ClassBalancedSampler initialized")
        print(f"{'='*70}")
        print(f"Original dataset size: {len(self.dataset)}")
        print(f"Effective epoch size: {self.num_samples}")
        print(f"Oversampling factor: {self.num_samples / len(self.dataset):.2f}x")
        print(f"Balance mode: {self.balance_mode}")
        print(f"{'='*70}\n")

    def _analyze_labels(self):
        """Analyze class distribution from label files."""
        label_dir = self.data_root / self.split / "labels"

        class_counts = defaultdict(int)
        images_with_class = defaultdict(list)  # image_idx -> list of classes

        # If dataset is a Subset, get original indices
        if hasattr(self.dataset, 'indices'):
            # Dataset is a Subset
            original_dataset = self.dataset.dataset
            indices = self.dataset.indices
        else:
            # Dataset is the full dataset
            original_dataset = self.dataset
            indices = range(len(self.dataset))

        # Read labels for each sample
        for idx in indices:
            # Get the stem from the original dataset
            sample = original_dataset.samples[idx]
            stem = sample['stem']
            label_path = label_dir / f"{stem}.txt"

            if label_path.exists():
                with open(label_path, 'r') as f:
                    classes_in_image = set()
                    for line in f.readlines():
                        line = line.strip()
                        if not line:
                            continue
                        cls = int(line.split()[0])
                        class_counts[cls] += 1
                        classes_in_image.add(cls)

                    # Store which classes are in this image
                    for cls in classes_in_image:
                        images_with_class[cls].append(idx)

        return {
            'class_counts': class_counts,
            'images_with_class': images_with_class
        }

    def _compute_sample_weights(self):
        """
        Compute sampling weight for each image based on minority classes it contains.

        Images containing minority classes get higher weights.
        """
        class_counts = self.class_distribution['class_counts']
        images_with_class = self.class_distribution['images_with_class']

        if not class_counts:
            # No labels found, use uniform sampling
            return [1.0] * len(self.dataset)

        # Calculate mean and identify minority classes
        mean_count = np.mean(list(class_counts.values()))
        minority_threshold = mean_count * self.oversample_threshold

        minority_classes = {
            cls: count for cls, count in class_counts.items()
            if count < minority_threshold
        }

        print(f"Mean instances per class: {mean_count:.1f}")
        print(f"Minority threshold: {minority_threshold:.1f}")
        print(f"Minority classes: {list(minority_classes.keys())}")

        # Compute oversample factor for each minority class
        oversample_factors = {}
        for cls, count in minority_classes.items():
            if self.balance_mode == 'full':
                # Fully balance to mean
                oversample_factors[cls] = mean_count / count
            elif self.balance_mode == 'sqrt':
                # Gentle balancing (square root)
                oversample_factors[cls] = np.sqrt(mean_count / count)
            else:
                raise ValueError(f"Unknown balance_mode: {self.balance_mode}")

        print("\nOversample factors:")
        for cls, factor in oversample_factors.items():
            print(f"  Class {cls}: {factor:.2f}x (instances: {class_counts[cls]})")

        # Assign weight to each image
        sample_weights = []
        for idx in range(len(self.dataset)):
            weight = 1.0  # Base weight

            # Increase weight if image contains minority classes
            for cls, factor in oversample_factors.items():
                if idx in images_with_class[cls]:
                    # Use max factor if image has multiple minority classes
                    weight = max(weight, factor)

            sample_weights.append(weight)

        return sample_weights

    def __iter__(self):
        """
        Generate indices for sampling with replacement based on weights.
        """
        # Sample indices with replacement based on weights
        indices = torch.multinomial(
            torch.tensor(self.sample_weights, dtype=torch.float),
            num_samples=self.num_samples,
            replacement=True
        ).tolist()

        return iter(indices)

    def __len__(self):
        return self.num_samples


class InstanceBalancedSampler(Sampler):
    """
    Alternative strategy: Balance by repeating samples to match target instance count.

    Simpler but more aggressive than ClassBalancedSampler.
    """

    def __init__(self, dataset, data_root, split="train", target_balance_ratio=1.0):
        """
        Args:
            target_balance_ratio: Ratio of min_class/max_class to achieve (0.5 = half balanced)
        """
        self.dataset = dataset
        self.data_root = Path(data_root)
        self.split = split
        self.target_balance_ratio = target_balance_ratio

        # Analyze distribution
        self.class_distribution = self._analyze_labels()

        # Create sampling indices
        self.indices = self._create_balanced_indices()

        print(f"\n{'='*70}")
        print(f"InstanceBalancedSampler initialized")
        print(f"{'='*70}")
        print(f"Original dataset size: {len(self.dataset)}")
        print(f"Balanced epoch size: {len(self.indices)}")
        print(f"Repeat factor: {len(self.indices) / len(self.dataset):.2f}x")
        print(f"Target balance ratio: {self.target_balance_ratio}")
        print(f"{'='*70}\n")

    def _analyze_labels(self):
        """Same as ClassBalancedSampler."""
        label_dir = self.data_root / self.split / "labels"

        class_counts = defaultdict(int)
        images_with_class = defaultdict(list)

        if hasattr(self.dataset, 'indices'):
            original_dataset = self.dataset.dataset
            indices = self.dataset.indices
        else:
            original_dataset = self.dataset
            indices = range(len(self.dataset))

        for idx in indices:
            sample = original_dataset.samples[idx]
            stem = sample['stem']
            label_path = label_dir / f"{stem}.txt"

            if label_path.exists():
                with open(label_path, 'r') as f:
                    classes_in_image = set()
                    for line in f.readlines():
                        line = line.strip()
                        if not line:
                            continue
                        cls = int(line.split()[0])
                        class_counts[cls] += 1
                        classes_in_image.add(cls)

                    for cls in classes_in_image:
                        images_with_class[cls].append(idx)

        return {
            'class_counts': class_counts,
            'images_with_class': images_with_class
        }

    def _create_balanced_indices(self):
        """Create list of indices with repetitions for minority classes."""
        class_counts = self.class_distribution['class_counts']
        images_with_class = self.class_distribution['images_with_class']

        if not class_counts:
            return list(range(len(self.dataset)))

        # Calculate target count
        max_count = max(class_counts.values())
        target_count = max_count * self.target_balance_ratio

        # Determine repeat factor for each class
        repeat_factors = {}
        for cls, count in class_counts.items():
            if count < target_count:
                repeat_factors[cls] = int(np.ceil(target_count / count))
            else:
                repeat_factors[cls] = 1

        print("Repeat factors per class:")
        for cls, factor in repeat_factors.items():
            print(f"  Class {cls}: {factor}x repeats (instances: {class_counts[cls]})")

        # Build index list
        indices = []
        image_repeat_count = defaultdict(int)

        # First pass: add all images once
        indices.extend(range(len(self.dataset)))

        # Second pass: repeat images with minority classes
        for cls, repeat_factor in repeat_factors.items():
            if repeat_factor > 1:
                # Repeat images containing this class
                for img_idx in images_with_class[cls]:
                    repeats_needed = repeat_factor - 1  # -1 because already added once
                    for _ in range(repeats_needed):
                        indices.append(img_idx)
                        image_repeat_count[img_idx] += 1

        # Shuffle to avoid sequential duplicates
        np.random.shuffle(indices)

        return indices

    def __iter__(self):
        # Shuffle indices each epoch
        indices = self.indices.copy()
        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.indices)
