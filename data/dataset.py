"""
MS-LarchPest: Harmonized multi-source UAV forestry pest dataset.

Contains 12,658 images and 41,035 annotated tree-crown instances with
three severity-oriented categories:
- H: Healthy
- LD: Light Damage
- HD: Heavy Damage

Supports multiple annotation formats (VOC XML, YOLO txt) from the
three source datasets:
1. Forest Damages-Larch Casebearer (VOC XML)
2. PWD-MFS (custom format)
3. PWD UAV Multispectral Imagery Dataset (YOLO format)
"""

import os
import cv2
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET


class MSLarchPestDataset(Dataset):
    """
    MS-LarchPest dataset loader.

    Supports VOC XML, YOLO txt, and JSON annotation formats.
    """

    # Category mapping: original labels → unified label space
    CATEGORY_MAP = {
        'healthy': 0, 'H': 0, 'Healthy': 0,
        'healthy tree crown': 0,
        'light_damage': 1, 'LD': 1, 'Light Damage': 1,
        'early': 1, 'middle': 1,
        'lightly damaged tree crown': 1,
        'high_damage': 2, 'HD': 2, 'High Damage': 2,
        'late': 2,
        'heavily damaged tree crown': 2,
        'partially discolored': 1, 'fully discolored': 2,
    }

    NUM_CLASSES = 3
    CLASS_NAMES = ['H', 'LD', 'HD']

    def __init__(self, root: str, split: str = 'train',
                 input_size: Tuple[int, int] = (640, 640),
                 transforms=None, anno_format: str = 'auto'):
        """
        Args:
            root: dataset root directory
            split: 'train', 'val', or 'test'
            input_size: target image size
            transforms: optional albumentations transforms
            anno_format: 'voc', 'yolo', 'json', or 'auto'
        """
        self.root = root
        self.split = split
        self.input_size = input_size
        self.transforms = transforms
        self.anno_format = anno_format

        # Load split file
        self.samples = self._load_split()

    def _load_split(self) -> List[Dict]:
        """Load image paths and annotation paths from split file."""
        split_file = os.path.join(self.root, 'splits', f'{self.split}.txt')
        samples = []

        if os.path.exists(split_file):
            # Use pre-defined split file
            with open(split_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 2:
                            img_path, anno_path = parts
                        else:
                            img_path = parts[0]
                            anno_path = self._infer_anno_path(img_path)
                        samples.append({
                            'image': os.path.join(self.root, img_path),
                            'annotation': os.path.join(self.root, anno_path),
                        })
        else:
            # Auto-discover from directory structure
            img_dir = os.path.join(self.root, 'images', self.split)
            if not os.path.exists(img_dir):
                img_dir = os.path.join(self.root, self.split, 'images')

            for img_name in sorted(os.listdir(img_dir)):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif')):
                    img_path = os.path.join(img_dir, img_name)
                    anno_path = self._infer_anno_path(img_path)
                    if os.path.exists(anno_path):
                        samples.append({
                            'image': img_path,
                            'annotation': anno_path,
                        })

        return samples

    def _infer_anno_path(self, img_path: str) -> str:
        """Infer annotation path from image path."""
        base = os.path.splitext(img_path)[0]

        # Try different formats
        for ext in ['.xml', '.txt', '.json']:
            anno_path = base + ext
            if os.path.exists(anno_path):
                return anno_path

        # Try replacing 'images' with 'annotations' or 'labels'
        for anno_dir in ['annotations', 'labels', 'Annotations', 'Labels']:
            for ext in ['.xml', '.txt', '.json']:
                anno_path = img_path.replace('images', anno_dir)
                anno_path = os.path.splitext(anno_path)[0] + ext
                if os.path.exists(anno_path):
                    return anno_path

        return base + '.xml'

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        # Load image
        image = cv2.imread(sample['image'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]

        # Load annotations
        boxes, labels = self._load_annotation(sample['annotation'], orig_w, orig_h)

        # Apply transforms
        if self.transforms:
            transformed = self.transforms(
                image=image, bboxes=boxes, class_labels=labels)
            image = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']

        # Resize
        image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)

        # Normalize boxes if not already
        if boxes and len(boxes) > 0:
            boxes = np.array(boxes, dtype=np.float32)
            if boxes.max() > 1.0:
                boxes[:, [0, 2]] /= orig_w
                boxes[:, [1, 3]] /= orig_h
            # Convert to cxcywh
            boxes_cxcywh = self._xyxy_to_cxcywh(boxes)
        else:
            boxes_cxcywh = np.zeros((0, 4), dtype=np.float32)
            labels = []

        # Convert to tensor
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        target = {
            'boxes': torch.from_numpy(boxes_cxcywh),
            'labels': torch.tensor(labels, dtype=torch.long),
            'orig_size': torch.tensor([orig_h, orig_w]),
            'image_id': torch.tensor([idx]),
        }

        return {'image': image, 'target': target}

    def _load_annotation(self, anno_path: str, img_w: int, img_h: int
                         ) -> Tuple[List, List]:
        """Load annotation from file, auto-detecting format."""
        if anno_path.endswith('.xml'):
            return self._load_voc_annotation(anno_path, img_w, img_h)
        elif anno_path.endswith('.json'):
            return self._load_json_annotation(anno_path, img_w, img_h)
        else:
            return self._load_yolo_annotation(anno_path, img_w, img_h)

    def _load_voc_annotation(self, anno_path: str, img_w: int, img_h: int
                              ) -> Tuple[List, List]:
        """Load VOC XML annotation."""
        boxes = []
        labels = []

        try:
            tree = ET.parse(anno_path)
            root = tree.getroot()

            for obj in root.iter('object'):
                name = obj.find('name').text
                label = self._map_category(name)

                bndbox = obj.find('bndbox')
                x1 = float(bndbox.find('xmin').text)
                y1 = float(bndbox.find('ymin').text)
                x2 = float(bndbox.find('xmax').text)
                y2 = float(bndbox.find('ymax').text)

                boxes.append([x1, y1, x2, y2])
                labels.append(label)

        except Exception as e:
            print(f"Warning: Failed to parse {anno_path}: {e}")

        return boxes, labels

    def _load_yolo_annotation(self, anno_path: str, img_w: int, img_h: int
                               ) -> Tuple[List, List]:
        """Load YOLO format annotation (normalized cxcywh)."""
        boxes = []
        labels = []

        try:
            with open(anno_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        cx = float(parts[1])
                        cy = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])

                        # YOLO format: normalize to absolute pixels
                        x1 = (cx - w / 2) * img_w
                        y1 = (cy - h / 2) * img_h
                        x2 = (cx + w / 2) * img_w
                        y2 = (cy + h / 2) * img_h

                        # Remap class id
                        label = self._remap_class_id(cls_id)

                        boxes.append([x1, y1, x2, y2])
                        labels.append(label)
        except Exception as e:
            print(f"Warning: Failed to parse {anno_path}: {e}")

        return boxes, labels

    def _load_json_annotation(self, anno_path: str, img_w: int, img_h: int
                               ) -> Tuple[List, List]:
        """Load COCO-style JSON annotation."""
        boxes = []
        labels = []

        try:
            with open(anno_path, 'r') as f:
                data = json.load(f)

            for ann in data.get('annotations', data if isinstance(data, list) else []):
                if isinstance(ann, dict):
                    label = ann.get('category_id', ann.get('class', 0))
                    label = self._remap_class_id(label)

                    bbox = ann.get('bbox', None)
                    if bbox:
                        if len(bbox) == 4:
                            x, y, w, h = bbox
                            x1, y1, x2, y2 = x, y, x + w, y + h
                            boxes.append([x1, y1, x2, y2])
                            labels.append(label)
        except Exception as e:
            print(f"Warning: Failed to parse {anno_path}: {e}")

        return boxes, labels

    def _map_category(self, name: str) -> int:
        """Map category name to unified label."""
        return self.CATEGORY_MAP.get(name, 0)

    def _remap_class_id(self, cls_id: int) -> int:
        """Remap source-specific class id to unified label."""
        # Clamp to valid range
        return min(max(int(cls_id), 0), self.NUM_CLASSES - 1)

    @staticmethod
    def _xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [cx, cy, w, h]."""
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        return np.stack([cx, cy, w, h], axis=1)


def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for variable-size targets."""
    images = torch.stack([item['image'] for item in batch])
    targets = [item['target'] for item in batch]

    return {'image': images, 'targets': targets}


def build_dataloader(root: str, split: str = 'train',
                     batch_size: int = 16, num_workers: int = 4,
                     input_size: Tuple[int, int] = (640, 640),
                     transforms=None) -> DataLoader:
    """Build a DataLoader for MS-LarchPest."""
    dataset = MSLarchPestDataset(
        root=root,
        split=split,
        input_size=input_size,
        transforms=transforms,
    )

    shuffle = split == 'train'
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=split == 'train',
    )
