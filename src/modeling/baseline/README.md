# Baseline: Segmentation model
This is a pretrained segmentation model from the [Punyasena Paleoecology and Palynology Lab](https://github.com/paleopollen) (github repo for model [here](https://github.com/paleopollen/pollen-detection-cli)). The model takes, as input, multi-focal tile crops of an ndpi image and outputs bounding boxes of pollen detections within each tile, irrespective of the focal plane. It is important to note that this model is trained particularly on fossil pollen images to detect fossil pollen, whereas the image data and ground truth annotations provided by the Smithsonian include fossil palynomorphs, which fossil pollen are a sub category of. 

## Setup Virtual Environment
Recommended Python version: 3.9
If not already done so, deactivate any current conda or python virtual environments. (conda: `conda deactivate` or Python virtual environment: `deactivate`)
Next, run the following commands:
```
cd src/modeling/baseline/pollen-detection-cli
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install openslide-python, pycocotools, torchmetrics
```

## Manual Tweak to Code (IMPORTANT!)
in line 71 of pollen-detection-cli/src/pollen_detector.py, change `self.conf_thresh = 0.20` to be `self.conf_thresh = 0.0`. This allows for the detector to initially find ALL possible predictions. Afterwards, this project will filter resulting predictions based on user input prediction confidence threshold (see below). 

## Run Baseline Modeling Script Via Command Line Interface
To run the baseline model, run the `baseline_modeling.py` script located in `Smithsonian_fossil_Sp25/scripts`. See below
```
cd scripts
python baseline_modeling.py
```
This script will first prompt the user to input some configurations:
- directory location for the baseline outputs: The baseline model will generate a directory called "baseline_outputs". User will input text representing the absolute path to the directory where the model can store this "baseline_outputs" folder
- prediction confidence threshold: This represents the confidence threshold at which to include predictions in the outputs and evaluation. User will input a decimal value between 0.0 and 1.0
- pretrained model weights: User will input text representing the absolute path to the pretrained model weights

After the model is finished running, the "baseline_outputs" folder will have the following structure:
```
baseline_outputs/
--> baseline_eval_results.txt
--> reformatted_tiles/
--> prediction_ndpas/
--> model_outputs/
```

## Description of Outputs
#### `baseline_eval_results.txt`
This text file contains the mean average precision results of the model. See top level README for more information. 

#### `reformatted tiles/`
This subdirectory stores the tiles in the format that the model expects for input. Particularly, the data is transformed from 25, 2048x2048 focal planes for each tile, to 9, 1024x1024 focal planes per tile. The focal planes chosen were even spaced from the original 25 focal planes. The structure of this subdirectory is the same structure as the ndpi_tiles directory in the image preprocessing step

#### `prediction_ndpas/`
This subdirectory stores all generated ndpa files that store predictions. The naming convention of the ndpa files is 'sample_name.ndpi_predictions.ndpa'.
- ex. `D3094_1_L_2024_02_20_11_32_26_Colorado.ndpi_predictions.ndpa`

#### `model_outputs/`
This subdirectory stores all the direct outputs of the model. The structure of the model_outputs directory is as follows:
```
model_outputs/
--> <ndpi_filename>_<tile_id>_<detection number>/ # detection by ndpi, tile_id, and detection number
  --> 0z.png # detection crop at focal plane 1
  --> ...
  --> 8z.png # detection crop at focal plane 9
  --> mask_1.png # detection mask
  --> metadata_1.json # details regarding the detection
...
```
For example,
```
model_outputs/
--> D3094_1_L_2024_02_20_11_32_26_Colorado_51767x_36884y_1/ 
  --> ...
--> D3094_1_L_2024_02_20_11_32_26_Colorado_51767x_36884y_2/
  --> ...
--> D3094_1_L_2024_02_20_11_32_26_Colorado_55931x_38272y_1/
  --> ...
--> D5151-B-1_L_2024_02_02_15_31_05_Tennessee_58014x_48900y_1/
  --> ...
--> D5151-B-1_L_2024_02_02_15_31_05_Tennessee_59402x_50288y_1/
...
```
What the above structure means is that for the Colorado file, there were 2 detections in the 51767x_36884y tile, and only 1 detection in the 55931x_38272y tile. For the Tennessee file, there was 1 detection for the 58013x_48900 tile, and 1 detection for the 59402x_50288y tile. 

## Run Evaluation of Baseline Model
To calculate the mean average precision for the baseline model, run the `baseline_eval.py` script located in `Smithsonian_fossil_Sp25/scripts`. See below. __NOTE__ eval must be run AFTER the modeling is finished. 
```
cd scripts
python baseline_eval.py
```

After the evaluation script is finished running, the mean average precision outputs will be in the "baseline_outputs" folder. See below:
```
baseline_outputs/
**--> baseline_eval_results.txt**
--> reformatted_tiles/
--> prediction_ndpas/
--> model_outputs/
```

There are many lines that the mean average precision outputs. They are aligned with torch metrics mean average precision library. However, we are more focused on the line that begins with mAP_50, which stands for mean average precision at IoU = 0.5
