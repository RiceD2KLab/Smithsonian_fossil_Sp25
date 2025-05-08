# Faster R-CNN: Object Detection Model for Palynomorphs

This directory contains the implementation of a Faster R-CNN object detection pipeline for localizing and classifying fossil palynomorphs in NDPI microscopy tiles. Unlike the segmentation-based baseline model, Faster R-CNN outputs bounding boxes and class predictions for each detection.

## Model Overview

Faster R-CNN is a two-stage detector:

1. A Region Proposal Network (RPN) generates object proposals.
2. These proposals are classified and refined using a convolutional backbone (ResNet50 in our case).

We fine-tune a pretrained Faster R-CNN model (originally trained on COCO) to detect 7 palynomorph categories. Each input is a single focal plane crop of size 1024×1024, extracted from the center plane of each NDPI tile. During inference, Non-Maximum Suppression (NMS) is applied to remove overlapping detections.

---

## Setup Instructions

### 1. Environment Setup

Recommended Python version: 3.9

```bash
python3.10 -m venv venv
source venv/bin/activate
cd src/modeling/faster_rcnn
pip install -r requirements.txt
```

---

### 2. Directory Structure

```
Smithsonian_fossil_Sp25/
├── scripts/
│   ├── faster_rcnn_modeling.py       # Run training
│   ├── faster_rcnn_eval.py           # Evaluate model (mAP)
│   └── setup/
│       └── setup_faster_rcnn_config.py  # Interactive setup script
├── src/
│   ├── modeling/
│   │   └── faster_rcnn.py            # Model definition and configuration
│   └── evaluation/
│       └── faster_rcnn_eval.py       # mAP computation logic
```

---

### 3. Configuration Setup

Before training or evaluation, configure paths and parameters:

```bash
cd scripts
python setup/setup_faster_rcnn_config.py
```

This script prompts for the master annotation CSV path, output directory, and other settings.

---

### 4. Run the Model

```bash
cd scripts
python faster_rcnn_modeling.py
```

---

### 5. Evaluate the Model

```bash
python faster_rcnn_eval.py
```

This computes both class-specific and class-agnostic mean Average Precision (mAP) using predictions saved in:

```
src/evaluation/faster_rcnn/tmp/raw_predictions.csv
```

---

## Outputs

* **Prediction NDPAs**: Bounding boxes, confidence scores, and class labels per tile
* **Evaluation Report**: mAP metrics per class and overall

---
