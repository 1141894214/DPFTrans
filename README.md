# DPF-Trans: Dynamic Pathology-Aware Feature Pruning Transformer for Efficient UAV-Based Larch Health-Status Detection

Official implementation of **DPF-Trans**, a dynamic pathology-aware feature pruning Transformer for efficient UAV-based larch health-status detection.

DPF-Trans is designed for detecting and distinguishing **healthy**, **diseased**, and **dead** larch crowns from high-resolution RGB UAV imagery. The method addresses three key challenges in UAV forestry scenes: weak pathological cues, redundant background computation, and visual confusion among fine-grained health categories.

> Paper: *DPF-Trans: Dynamic Pathology-Aware Feature Pruning Transformer for Efficient UAV-Based Larch Health-Status Detection*
> Code: https://github.com/1141894214/DPFTrans

---

## Overview

UAV-based larch pest monitoring is challenging because early pest symptoms are often subtle, tree crowns are sparsely distributed in complex forest backgrounds, and different health states may look visually similar.

To address these problems, DPF-Trans introduces a pathology-aware sparse detection framework that enhances weak disease-related features, dynamically prunes redundant background regions before Transformer interaction, and improves the separability of visually confusing health-status categories.

The overall framework consists of three main components:

1. **PA-HGNet**: a pathology-aware feature enhancement backbone.
2. **Feature-Guided Sparse Encoder**: including FGDP, ASFI, and CCFF.
3. **Language-Guided Multimodal Decoder with VCBL**: improving fine-grained health-status discrimination.

---

## Main Contributions

* **Dynamic pathology-aware feature pruning framework**
  DPF-Trans performs larch-crown-aware sparse modeling before expensive Transformer interaction, reducing redundant background computation while preserving informative crown regions.

* **PA-HGNet for weak pathological cue enhancement**
  PA-HGNet enhances lesion textures, canopy boundaries, and structural abnormalities through pathology-aware feature recalibration and morphology-preserving downsampling.

* **Feature-Guided Dynamic Pruning module**
  FGDP uses pathology-aware priors and morphological fragmentation cues to adaptively retain informative larch crown windows and suppress non-target background regions.

* **Sparse topology reconstruction and cross-scale fusion**
  ASFI reconstructs semantic and geometric dependencies among retained sparse tokens, while CCFF fuses multi-scale lesion, canopy, and semantic features into a compact visual memory.

* **Visual-Confusion Boundary Learning**
  VCBL aligns visual queries with health-status class prototypes and introduces box-aware constraints to improve discrimination among healthy, diseased, and dead crowns.

---

## Performance

### Forest Damages–Larch Casebearer Dataset

| Method    |    mAP@50 |       FPS |    Params |    GFLOPs |
| --------- | --------: | --------: | --------: | --------: |
| DPF-Trans | **89.2%** | **132.3** | **28.4M** | **84.1G** |

Category-wise AP:

| Category |    AP |
| -------- | ----: |
| Healthy  | 91.8% |
| Diseased | 86.2% |
| Dead     | 88.3% |

DPF-Trans achieves a strong balance between detection accuracy and inference efficiency, making it suitable for UAV-based forest pest monitoring scenarios.

---

## Framework

The DPF-Trans pipeline can be summarized as:

```text
Input UAV RGB Image
        |
        v
PA-HGNet
  - PA-GhostBlock
  - MP-Down
        |
        v
Multi-scale Features: C3, C4, C5
Pathology-aware Priors: T3, T4
        |
        v
Feature-Guided Sparse Encoder
  - FGDP: Feature-Guided Dynamic Pruning
  - ASFI: Asymmetric Sparse Feature Interaction
  - CCFF: Cross-Scale Contextual Feature Fusion
        |
        v
Sparse Visual Memory
        |
        v
Language-Guided Multimodal Decoder
  - Query Initialization
  - Image Cross-Attention
  - Text Cross-Attention
  - VCBL
        |
        v
Health-Status Detection Results
```

---

## Dataset

### Forest Damages–Larch Casebearer

The main dataset used in this project is the **Forest Damages–Larch Casebearer** dataset, which contains UAV RGB images of larch forests affected by larch casebearer.

Dataset characteristics:

* Image resolution: `640 × 640`
* Number of images: `1,543`
* Number of annotated tree instances: `10,187`
* Categories:

  * `healthy`
  * `diseased`
  * `dead`

The dataset is split at the image level with a ratio of `7:2:1` for training, validation, and testing. The random seed is fixed to `42`.

Expected dataset structure:

```text
datasets/
└── larch_casebearer/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── labels/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── data.yaml
```

Example `data.yaml`:

```yaml
path: datasets/larch_casebearer
train: images/train
val: images/val
test: images/test

names:
  0: healthy
  1: diseased
  2: dead
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/1141894214/DPFTrans.git
cd DPFTrans
```

### 2. Create environment

```bash
conda create -n dpftrans python=3.8 -y
conda activate dpftrans
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Recommended environment:

```text
Python 3.8.18
PyTorch 1.13.1
CUDA 11.7
GPU: NVIDIA RTX 4090 24GB
```

---

## Training

Train DPF-Trans on the larch health-status detection dataset:

```bash
python train.py \
  --config configs/dpf_trans_larch.yaml \
  --data datasets/larch_casebearer/data.yaml \
  --epochs 300 \
  --batch-size 8 \
  --img-size 640
```

Main training settings:

```text
Optimizer: AdamW
Initial learning rate: 1e-4
Weight decay: 1e-4
Batch size: 8
Epochs: 300
Input size: 640 × 640
Number of decoder queries: 300
Decoder layers: 3
FGDP window size: 8
Gumbel-Softmax temperature: 1.0 -> 0.1
```

---

## Evaluation

Evaluate a trained checkpoint:

```bash
python val.py \
  --config configs/dpf_trans_larch.yaml \
  --data datasets/larch_casebearer/data.yaml \
  --weights runs/train/dpf_trans/weights/best.pt \
  --img-size 640
```

Expected metrics include:

```text
mAP@50
AP for healthy crowns
AP for diseased crowns
AP for dead crowns
Params
GFLOPs
FPS
Latency
```

---

## Inference

Run inference on UAV images:

```bash
python detect.py \
  --weights runs/train/dpf_trans/weights/best.pt \
  --source demo/images \
  --img-size 640 \
  --conf-thres 0.25
```

The detection results will be saved to:

```text
runs/detect/
```

---

## Project Structure

```text
DPFTrans/
├── configs/
│   └── dpf_trans_larch.yaml
├── datasets/
│   └── larch_casebearer/
├── models/
│   ├── dpf_trans.py
│   ├── pa_hgnet.py
│   ├── fgdp.py
│   ├── asfi.py
│   ├── ccff.py
│   └── vcbl.py
├── utils/
├── train.py
├── val.py
├── detect.py
├── requirements.txt
└── README.md
```

---

## Key Modules

### PA-HGNet

PA-HGNet is a pathology-aware feature enhancement backbone. It is designed to enhance weak disease-related cues before pruning and detection.

Main components:

* **PA-GhostBlock**: enhances pathology-sensitive spatial responses.
* **MP-Down**: preserves both high-frequency lesion boundaries and low-frequency canopy structures during downsampling.

### FGDP

FGDP performs feature-guided dynamic pruning. It partitions feature maps into local windows and estimates the importance of each window using:

* pathology-aware texture prior
* morphological fragmentation prior
* window-level visual representation

Only informative windows are retained for sparse Transformer interaction.

### ASFI

ASFI reconstructs sparse feature topology after pruning. It combines:

* global semantic interaction
* local deformable geometric modeling

This helps recover semantic and spatial dependencies among discontinuously distributed larch crown tokens.

### CCFF

CCFF performs cross-scale contextual fusion. It aggregates:

* fine-grained lesion textures
* canopy-level structures
* high-level pathological semantics

The output is a compact sparse visual memory for the decoder.

### VCBL

VCBL improves fine-grained category discrimination by aligning visual queries with health-status class prototypes and applying box-aware localization constraints.

---

## Citation

If you find this project useful, please cite our work:

```bibtex
@article{zhang2026dpftrans,
  title={DPF-Trans: Dynamic Pathology-Aware Feature Pruning Transformer for Efficient UAV-Based Larch Health-Status Detection},
  author={Zhang, Hailin and Wang, Shaopeng},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  year={2026}
}
```

---

## Acknowledgements

This work was supported by the National Natural Science Foundation of China under Grant 62066034.

We thank the providers of the public UAV forestry datasets used in this study.

---

## Contact

For questions, please contact:

```text
Hailin Zhang
College of Computer Science, Inner Mongolia University
Email: please add your email here
```

---

## License

This project is released for academic research purposes only. Please refer to the LICENSE file for more details.
