# DETR Pipeline Quick Start

A minimal guide to run the full DETR workflow from preprocessing CSV annotations to exporting NDPA files.

## 1. Prerequisites

* Python 3.8+

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

Load a pretrained facebook/detr-resnet-50 model
Fine-tune on your custom dataset
Save model checkpoints and training logs to your specified output directory
Make sure to configure paths and hyperparameters inside the script or via a config file.

## 4. Inference & NDPA Export

Finally, run:

```bash
python inference.py
```

This runs detection, applies per-tile NMS, and writes one `.ndpi.ndpa` file per slide into the NDPA output directory.
