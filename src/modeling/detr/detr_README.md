# DETR Pipeline Quick Start

A minimal guide to run the full DETR workflow from preprocessing CSV annotations to exporting NDPA files.

## 1. Prerequisites

* Python 3.8+

## 2. Preprocessing (CSV → COCO → Train/Val)

In this folder, run:

```bash
python coco_preprocessing.py
```

This will produce under `output_dir`:

* `pollen_dataset.json`
* `pollen_dataset_no_indet.json`
* `pollen_train.json` & `pollen_val.json`

## 3. Fine-tuning

We use the existing script at `../fine_tuning/detr_finetune.py`:

```bash
python ../fine_tuning/detr_finetune.py \
  --detr-config detr_config.json \
  --project-config ../../config.json
```

It will train the model, logging to the output directory and saving weights as specified.

## 4. Inference & NDPA Export

Finally, run:

```bash
python inference.py \
  --detr-config detr_config.json \
  --project-config ../../config.json
```

This runs detection, applies per-tile NMS, and writes one `.ndpa` file per slide into the NDPA output directory.
