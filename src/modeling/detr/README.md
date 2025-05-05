# DETR Pipeline Quick Start

A minimal guide to run the full DETR workflow from preprocessing CSV annotations to exporting NDPA files.

## 0. Prerequisites

* Python 3.8+
* pip install -r requirements.txt

## 1. DETR Configuration

Before running the following scripts, you may use the configuration file that's specific to DETR located at 
```bash
./detr_config.json
```
to fit your needs. Here's a description of fields you may need to edit:
* model_name: CNN Backbone for your model (e.g., "facebook/detr-resnet-50")
* weights_path: Absolute path to pretrained or fine-tuned weights
* num_labels: Number of classes
* output_dir: Absolute path of where you'd like all your output files stored for the scripts
* focal_length: Number of focal planes at which the slides were taken at.
* All other fields are for hyperparameter tuning purposes.


## 2. Preprocessing (CSV → COCO → Train/Val)

In this folder, run:

```bash
python coco_preprocessing.py
```

This generates the following files in output_dir:

* pollen_dataset.json – full dataset in COCO format
* pollen_dataset_no_indet.json – dataset excluding "indeterminate" class also in COCO format
* pollen_train.json, pollen_val.json – stratified train/val split
These files are needed DETR fine-tuning and evaluation.


## 3. Fine-tuning

We use the existing script at `../fine_tuning/detr_finetune.py`:

```bash
python ../fine_tuning/detr_finetune.py 
```

This will:

* Load a pretrained facebook/detr-resnet-50 model
* Fine-tune on your custom dataset
* Save model weights to your specified output directory
* Make sure to configure paths and hyperparameters inside the script or via the config file.

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
Note: Inference only requires tiled images and model weights to perform this.
