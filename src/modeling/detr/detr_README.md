# DETR Pipeline Quick Start

A minimal guide to run the full DETR workflow from preprocessing CSV annotations to exporting NDPA files.

## 1. Prerequisites

* Python 3.8+
* pip install -r requirements.txt

## 2. Preprocessing (CSV → COCO → Train/Val)

In this folder, run:

```bash
python coco_preprocessing.py
```

This generates the following files in output_dir:

* pollen_dataset.json – full dataset in COCO format
* pollen_dataset_no_indet.json – dataset excluding "indeterminate" class
* pollen_train.json, pollen_val.json – stratified train/val split
These files are ready to be used with DETR fine-tuning.


## 3. Fine-tuning

We use the existing script at `../fine_tuning/detr_finetune.py`:

```bash
python ../fine_tuning/detr_finetune.py 
```

This will:

* Load a pretrained facebook/detr-resnet-50 model
* Fine-tune on your custom dataset
* Save model checkpoints and training logs to your specified output directory
* Make sure to configure paths and hyperparameters inside the script or via a config file.

## 4. Evaluation

```bash
python ../../evaluation/detr/evaluate.py 
```

This reports:

* mAP (mean Average Precision)
* Per-class AP
* Precision/recall at different IoU thresholds

## 5. Inference & NDPA Export

Finally, run:

```bash
python inference.py
```

This performs:

* Object detection on each tile image
* Tile-level non-maximum suppression (NMS)
* Export of one .ndpa XML file per slide
* Output files will be written to the NDPA output directory defined in your project configuration
