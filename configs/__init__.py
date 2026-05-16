"""
DPF-Trans: Dynamic Pathology-Aware Feature Pruning Transformer
for Efficient UAV-Based Forest Pest Detection.

Reference: IEEE JSTARS

Project Structure:
    dpf_trans/
    ├── configs/          Configuration files
    │   └── config.py     Model and training config
    ├── data/             Dataset and augmentation
    │   ├── dataset.py    MS-LarchPest dataset loader
    │   └── transforms.py Data augmentation pipeline
    ├── models/           Model components
    │   ├── pa_hgnet.py   PA-HGNet backbone
    │   ├── fgdp.py       Feature-Guided Dynamic Pruning
    │   ├── asfi.py       Asymmetric Sparse Feature Interaction
    │   ├── ccff.py       Cross-Scale Contextual Feature Fusion
    │   ├── vcbl.py       Vision-to-Class/Box Contrastive Learning decoder
    │   ├── dpf_trans.py  Full DPF-Trans model
    │   └── losses.py     Loss functions
    ├── utils/            Utilities
    │   └── matcher.py    Hungarian matcher
    ├── train.py          Training script
    └── demo.py           Inference demo

Usage:
    # Training
    python train.py --data_root ./data/MS-LarchPest --output_dir ./output

    # Inference
    python demo.py --checkpoint output/best_model.pth --image_dir ./test_images

    # Benchmark
    python demo.py --checkpoint output/best_model.pth --benchmark

Performance (from paper):
    - mAP@0.5: 89.2%
    - FPS: 132.5
    - Parameters: 28.5M
    - FLOPs: 84.5G
"""

__version__ = "1.0.0"
