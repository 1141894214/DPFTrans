"""
Data augmentation transforms for MS-LarchPest.

Includes:
- Random horizontal/vertical flipping
- Random scale jittering
- Color jittering
- Mosaic augmentation (early training stage)
"""

import cv2
import random
import numpy as np
from typing import Dict, List, Tuple
from copy import deepcopy


class Compose:
    """Compose multiple transforms."""

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, image: np.ndarray, bboxes: list = None,
                 class_labels: list = None) -> dict:
        result = {'image': image, 'bboxes': bboxes or [], 'class_labels': class_labels or []}
        for t in self.transforms:
            result = t(result)
        return result


class RandomHorizontalFlip:
    """Random horizontal flip."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, result: dict) -> dict:
        if random.random() < self.p:
            image = result['image']
            result['image'] = cv2.flip(image, 1)

            if result['bboxes']:
                boxes = np.array(result['bboxes'])
                boxes[:, [0, 2]] = image.shape[1] - boxes[:, [2, 0]]
                result['bboxes'] = boxes.tolist()

        return result


class RandomVerticalFlip:
    """Random vertical flip."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, result: dict) -> dict:
        if random.random() < self.p:
            image = result['image']
            result['image'] = cv2.flip(image, 0)

            if result['bboxes']:
                boxes = np.array(result['bboxes'])
                boxes[:, [1, 3]] = image.shape[0] - boxes[:, [3, 1]]
                result['bboxes'] = boxes.tolist()

        return result


class RandomScaleJitter:
    """Random scale jitter."""

    def __init__(self, scale_range: Tuple[float, float] = (0.5, 1.5)):
        self.scale_range = scale_range

    def __call__(self, result: dict) -> dict:
        scale = random.uniform(*self.scale_range)

        image = result['image']
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        if result['bboxes']:
            boxes = np.array(result['bboxes'])
            boxes = boxes * scale
            result['bboxes'] = boxes.tolist()

        result['image'] = image
        return result


class ColorJitter:
    """Color jittering for robustness."""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2,
                 saturation: float = 0.2, hue: float = 0.1,
                 p: float = 0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.p = p

    def __call__(self, result: dict) -> dict:
        if random.random() < self.p:
            image = result['image'].astype(np.float32)

            # Brightness
            if self.brightness > 0:
                delta = random.uniform(-self.brightness, self.brightness)
                image += delta * 255

            # Contrast
            if self.contrast > 0:
                alpha = random.uniform(1 - self.contrast, 1 + self.contrast)
                gray = image.mean(axis=-1, keepdims=True)
                image = gray + alpha * (image - gray)

            # Convert to HSV for saturation and hue
            image = image.astype(np.uint8)
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)

            # Saturation
            if self.saturation > 0:
                alpha = random.uniform(1 - self.saturation, 1 + self.saturation)
                hsv[:, :, 1] *= alpha

            # Hue
            if self.hue > 0:
                delta = random.uniform(-self.hue, self.hue) * 180
                hsv[:, :, 0] += delta

            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            result['image'] = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return result


class RandomResizedCrop:
    """Random resized crop to target size."""

    def __init__(self, size: Tuple[int, int] = (640, 640),
                 scale: Tuple[float, float] = (0.6, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, result: dict) -> dict:
        image = result['image']
        h, w = image.shape[:2]

        scale = random.uniform(*self.scale)
        crop_h = int(h * scale)
        crop_w = int(w * scale)

        top = random.randint(0, max(0, h - crop_h))
        left = random.randint(0, max(0, w - crop_w))

        image = image[top:top + crop_h, left:left + crop_w]

        if result['bboxes']:
            boxes = np.array(result['bboxes'])
            boxes[:, [0, 2]] -= left
            boxes[:, [1, 3]] -= top

            # Filter out boxes that are mostly outside
            keep = (boxes[:, 2] > 0) & (boxes[:, 0] < crop_w) & \
                   (boxes[:, 3] > 0) & (boxes[:, 1] < crop_h)
            boxes = boxes[keep]
            labels = np.array(result['class_labels'])[keep]

            # Clip boxes
            boxes[:, 0] = boxes[:, 0].clip(0, crop_w)
            boxes[:, 1] = boxes[:, 1].clip(0, crop_h)
            boxes[:, 2] = boxes[:, 2].clip(0, crop_w)
            boxes[:, 3] = boxes[:, 3].clip(0, crop_h)

            result['bboxes'] = boxes.tolist()
            result['class_labels'] = labels.tolist()

        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        result['image'] = image
        return result


class MosaicAugmentation:
    """
    Mosaic augmentation: combines 4 images into one.

    Used in early training stage, disabled in final epochs.
    """

    def __init__(self, size: Tuple[int, int] = (640, 640), p: float = 1.0):
        self.size = size
        self.p = p

    def __call__(self, images: List[np.ndarray],
                 targets: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """
        Args:
            images: list of 4 images
            targets: list of 4 target dicts
        Returns:
            mosaic image and combined targets
        """
        if len(images) < 4 or random.random() > self.p:
            # Return first image unchanged
            return images[0], targets[0]

        h, w = self.size
        center_x = random.randint(int(w * 0.3), int(w * 0.7))
        center_y = random.randint(int(h * 0.3), int(h * 0.7))

        mosaic_img = np.zeros((h, w, 3), dtype=np.uint8)
        all_boxes = []
        all_labels = []

        # Place 4 images at 4 corners
        placements = [
            (0, 0, center_x, center_y),              # top-left
            (center_x, 0, w, center_y),               # top-right
            (0, center_y, center_x, h),               # bottom-left
            (center_x, center_y, w, h),               # bottom-right
        ]

        for i, (img, target) in enumerate(zip(images[:4], targets[:4])):
            if i >= len(placements):
                break

            x1, y1, x2, y2 = placements[i]
            pw, ph = x2 - x1, y2 - y1

            # Resize image to fit placement
            img_h, img_w = img.shape[:2]
            scale = min(pw / img_w, ph / img_h)
            new_w = int(img_w * scale)
            new_h = int(img_h * scale)

            img_resized = cv2.resize(img, (new_w, new_h))
            offset_x = x1 + (pw - new_w) // 2
            offset_y = y1 + (ph - new_h) // 2

            mosaic_img[offset_y:offset_y + new_h,
                       offset_x:offset_x + new_w] = img_resized

            # Adjust boxes
            if 'boxes' in target and len(target['boxes']) > 0:
                boxes = target['boxes'].clone() if hasattr(target['boxes'], 'clone') \
                    else target['boxes'].copy()

                if hasattr(boxes, 'numpy'):
                    boxes = boxes.numpy()

                boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale * img_w + offset_x
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale * img_h + offset_y

                # Clip
                boxes[:, 0] = boxes[:, 0].clip(x1, x2)
                boxes[:, 1] = boxes[:, 1].clip(y1, y2)
                boxes[:, 2] = boxes[:, 2].clip(x1, x2)
                boxes[:, 3] = boxes[:, 3].clip(y1, y2)

                # Remove degenerate boxes
                valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

                all_boxes.extend(boxes[valid].tolist())
                labels = target['labels']
                if hasattr(labels, 'numpy'):
                    labels = labels.numpy()
                all_labels.extend(np.array(labels)[valid].tolist())

        mosaic_target = {
            'boxes': np.array(all_boxes, dtype=np.float32) if all_boxes else np.zeros((0, 4), dtype=np.float32),
            'labels': np.array(all_labels, dtype=np.int64) if all_labels else np.zeros(0, dtype=np.int64),
        }

        return mosaic_img, mosaic_target


def build_augmentation(image_size: Tuple[int, int] = (640, 640),
                       use_mosaic: bool = True) -> Compose:
    """
    Build standard augmentation pipeline.

    Mosaic augmentation is applied at the batch level (in the training loop),
    not here.
    """
    transforms = [
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomScaleJitter(scale_range=(0.7, 1.3)),
        ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        RandomResizedCrop(size=image_size, scale=(0.6, 1.0)),
    ]
    return Compose(transforms)
