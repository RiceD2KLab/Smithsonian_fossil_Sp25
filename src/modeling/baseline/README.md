# Baseline: Segmentation model
This model is a segmentation model that has been taken from ____________________

## Setup Virtual Environment
Recommended Python version: 3.9
If not already done so, deactivate any current conda or python virtual environments. (conda: `conda deactivate` Python virtual environment: `deactivate`)
Next, run the following commands:
```
cd src/modeling/baseline/pollen-detection-cli
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install openslide-python, pycocotools, torchmetrics
```

## Run Baseline Modeling Script Via Command Line Interface
To run the baseline model, run the `baseline_modeling.py` script located in `Smithsonian_fossil_Sp25/scripts`. See below
```
cd scripts
python baseline_modeling.py
```
This script will first prompt the user to input some configurations:
- directory location for the baseline outputs: The baseline model will generate a directory called "baseline_outputs". User will input a directory where the model can store this "baseline_outputs" folder
- prediction confidence threshold: This represents the confidence threshold at which to include predictions in the outputs and evaluation

After the model is finished running, the "baseline_outputs" folder will have the following structure:
```
baseline_outputs/
--> reformatted_tiles/
--> prediction_ndpas/
--> model_outputs/
--> baseline_eval_results.txt
```

### Description of Outputs
reformatted tiles: This subdirectory stores the tiles in the format that the model expects. Particularly, the data is transformed from 25, 2048x2048 focal planes for each tile, to 9, 1024x1024 focal planes per tile. The focal planes chosen were even spaced from the original 25 focal planes. 

prediction_ndpas: This subdirectory stores all generated ndpa files that store predictions. The naming convention of the ndpa files is 'sample_name.ndpi_predictions.ndpa'.
- ex. D3094_1_L_2024_02_20_11_32_26_Colorado.ndpi_predictions.ndpa
baseline_eval_results.txt: this is a .txt file that stores the mean average precision results of evaluation for the model.

model_outputs: This subdirectory stores all the direct outputs of the model. 
```
Explain model_outputs structure here 
```

baseline_eval_results.txt: 



