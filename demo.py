"""
DPF-Trans inference demo script.

Loads a trained DPF-Trans model and runs inference on UAV forest images.
Reports mAP@0.5, FPS, parameters, and FLOPs.

Usage:
    python demo.py --checkpoint output/best_model.pth --image_dir ./test_images
    python demo.py --checkpoint output/best_model.pth --benchmark
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dpf_trans.models import DPFTrans
from dpf_trans.configs.config import DPFTransConfig


def parse_args():
    parser = argparse.ArgumentParser(description='DPF-Trans Demo')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Model checkpoint path')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Directory of images to process')
    parser.add_argument('--image_path', type=str, default=None,
                        help='Single image to process')
    parser.add_argument('--output_dir', type=str, default='./demo_output',
                        help='Output directory for visualizations')
    parser.add_argument('--input_size', type=int, nargs=2, default=[640, 640],
                        help='Input resolution (H W)')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run speed benchmark')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')

    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> DPFTrans:
    """Load trained DPF-Trans model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Build model from config or defaults
    cfg = DPFTransConfig()
    if 'config' in checkpoint:
        cfg = checkpoint['config']

    model = DPFTrans(
        num_classes=3,
        input_size=cfg.input_size,
    ).to(device)

    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    model.set_gumbel_temperature(0.1)  # Inference temperature

    return model


def preprocess_image(image: np.ndarray, input_size: tuple) -> torch.Tensor:
    """Preprocess image for DPF-Trans."""
    h, w = image.shape[:2]
    image_resized = cv2.resize(image, input_size, interpolation=cv2.INTER_LINEAR)
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor, (h, w)


def postprocess_predictions(class_logits, pred_boxes, orig_size,
                             conf_threshold: float = 0.5):
    """Convert model outputs to detection results."""
    scores, labels = class_logits.sigmoid().max(dim=-1)
    mask = scores > conf_threshold

    boxes = pred_boxes[mask]
    scores = scores[mask]
    labels = labels[mask]

    if len(boxes) == 0:
        return [], [], []

    # Convert boxes to original image coordinates
    orig_h, orig_w = orig_size
    boxes[:, [0, 2]] *= orig_w
    boxes[:, [1, 3]] *= orig_h

    # cxcywh → xyxy
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

    return boxes_xyxy.tolist(), scores.tolist(), labels.tolist()


def draw_detections(image: np.ndarray, boxes, scores, labels):
    """Draw bounding boxes on image."""
    CLASS_NAMES = ['H', 'LD', 'HD']
    COLORS = [(0, 255, 0), (255, 255, 0), (255, 0, 0)]

    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        color = COLORS[label % 3]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        text = f"{CLASS_NAMES[label]}: {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(image, text, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return image


def run_benchmark(model, device, input_size=(640, 640), num_runs=100):
    """Run inference speed benchmark."""
    print(f"\nSpeed Benchmark ({num_runs} runs)...")

    dummy = torch.randn(1, 3, *input_size).to(device)

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy)

    # Measure
    torch.cuda.synchronize()
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg_time = np.mean(times) * 1000
    fps = 1000 / avg_time

    n_params = model.count_parameters() / 1e6
    flops = model.count_flops()

    print(f"  Average latency: {avg_time:.2f} ms")
    print(f"  FPS: {fps:.1f}")
    print(f"  Parameters: {n_params:.1f}M")
    print(f"  FLOPs: {flops:.1f}G")
    print(f"\n  Paper reference: 132.5 FPS, 28.5M params, 84.5G FLOPs")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, device)
    print(f"Model loaded. Parameters: {model.count_parameters() / 1e6:.1f}M")

    # Benchmark
    if args.benchmark:
        run_benchmark(model, device, tuple(args.input_size))
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process images
    if args.image_path:
        image_paths = [args.image_path]
    elif args.image_dir:
        image_paths = [
            os.path.join(args.image_dir, f)
            for f in sorted(os.listdir(args.image_dir))
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif'))
        ]
    else:
        print("Please provide --image_dir or --image_path")
        return

    print(f"Processing {len(image_paths)} images...")

    for img_path in image_paths:
        print(f"  {os.path.basename(img_path)}...")

        image = cv2.imread(img_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]

        tensor, _ = preprocess_image(image_rgb, tuple(args.input_size))
        tensor = tensor.to(device)

        with torch.no_grad():
            outputs = model(tensor)

        boxes, scores, labels = postprocess_predictions(
            outputs['pred_logits'][0],
            outputs['pred_boxes'][0],
            (orig_h, orig_w),
            args.conf_threshold,
        )

        # Draw and save
        result = draw_detections(image.copy(), boxes, scores, labels)
        out_path = os.path.join(args.output_dir,
                                f"det_{os.path.basename(img_path)}")
        cv2.imwrite(out_path, result)

        # Print pruning stats
        s3_keep = outputs['keep_masks']['s3'].mean().item()
        s4_keep = outputs['keep_masks']['s4'].mean().item()
        print(f"    Detections: {len(boxes)} | "
              f"S3 keep ratio: {s3_keep:.2f} | S4 keep ratio: {s4_keep:.2f}")
        print(f"    Saved to: {out_path}")

    print("\nDone!")


if __name__ == '__main__':
    main()
