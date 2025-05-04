# Faster R-CNN: Object Detection Model for Palynomorphs

This directory contains the implementation of a Faster R-CNN object detection pipeline for localizing and classifying fossil palynomorphs in NDPI microscopy tiles. Unlike the segmentation-based baseline model, Faster R-CNN outputs bounding boxes and class predictions for each detection.

## Model Overview

Faster R-CNN is a two-stage detector:

1. A Region Proposal Network (RPN) first generates object proposals.
2. These proposals are then classified and refined using a convolutional backbone (ResNet50 in our case).

We fine-tune a pretrained Faster R-CNN model (originally trained on COCO) to detect 7 palynomorph categories in our dataset. All model inputs are single focal plane crops of size 1024×1024, extracted from the center plane of each NDPI tile. During inference, Non-Maximum Suppression (NMS) is applied to remove overlapping detections.

## Setup Instructions

### 1. Environment Setup

Recommended Python version: 3.9

```bash
conda deactivate  # or 'deactivate' if using virtualenv
conda create -n faster_rcnn_env python=3.9
conda activate faster_rcnn_env
pip install -r requirements.txt  # Make sure to include torchvision, torchmetrics, pandas, etc.
```

### 2. Directory Structure

```
Smithsonian_fossil_Sp25/
├── scripts/
│   └── faster_rcnn_modeling.py       # runs training
│   └── faster_rcnn_eval.py           # runs evaluation and prints mAP
├── src/
│   └── modeling/
│       └── faster_rcnn.py            # model setup and config
│   └── evaluation/
│       └── faster_rcnn_eval.py       # mAP computation logic
```

### 3. Running the Model

```bash
cd scripts
python faster_rcnn_modeling.py
```

### 4. Evaluation

Run this to compute mean Average Precision (mAP):

```bash
python faster_rcnn_eval.py
```

This will compute both class-agnostic and class-specific mAP using predictions in `src/evaluation/faster_rcnn/tmp/raw_predictions.csv`.

## Notes on Data

* Input images are extracted from NDPI tiles at a single focal plane (middle of 25).
* Ground truth annotations come from curated Smithsonian fossil pollen datasets.
* The model downsamples all inputs to 1333×1333 pixels automatically (PyTorch default behavior).

## Outputs

* Predictions saved as CSVs with bounding boxes, confidence scores, and predicted classes.
* Evaluation reports include class-wise and mean mAP scores.

## Limitations

* No hyperparameter tuning was performed beyond epoch count.
* Class imbalance in training data remains an issue and may affect precision for underrepresented classes.
* Data augmentation (e.g., flips, jittering) was used for DETR but not for Faster R-CNN.
