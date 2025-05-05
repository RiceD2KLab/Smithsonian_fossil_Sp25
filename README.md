# Smithsonian_fossil_Sp25
## Table of Contents
- [Introduction](#introduction)
- [Repo Structure](#repo-structure)
- [Installation](#installation)
- [Data](#data)
- [Setup](#setup)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling](#modeling)
- [Acknowledgments/Citations](#acknowledgmentscitations)
  
## Introduction
### Team Members

#### Students:
- **Aaeisha Baharun** (she/her)
- **Audrey (AJ) Kim** (they/them)
- **Jonathan Lee** (he/him)
- **Andrew Ondara** (he/him)
- **Izzy Sanchez** (he/him)
- **Yuhan (Bruce) Wu** (he/him)

#### Faculty Mentor:
- **Dr. Arko Barman**

#### PhD Mentor:
- **Krish Kabra**

#### Sponsor:
- **Dr. Ingrid Romero**

This repository hosts the code for an automated machine-learning model designed to detect and classify palynomorphs (fossil pollen, spores, and other organic-walled microorganisms) from high-resolution NDPI (NanoZoomer Digital Pathology Image) files. 
There are two contexts in which this repo can be used. 
- running the model on an unannotated region. In other words, running these models in a production like context
- running the models over an annotated region for evaluation of models purposes.
Unless otherwise mentioned, run all steps for both use cases. 


## Repo Structure


- `src/`
  - `data_preprocessing/` Scripts relevant to pre-processing the data and exploratory data analysis
  - `entities/` Classes to represent data structures
  - `modeling/` Scripts relevant for modeling
  - `evaluation/` Scripts relevant to model evaluation
  - `tools/` project wide utilities/tools
- `README.md` Project documentation
- `requirements.txt` Package requirements


## Installation
### Python and Conda Installation
1. Clone the repository:
   ```sh
   git clone --recurse-submodules https://github.com/RiceD2KLab/Smithsonian_fossil_Sp25.git

   cd /Smithsonian_fossil_Sp25
   ```
2. Install Python (recommended: version 3.9 or greater)
3. Install Conda (if using virtual environments)
4. Set up a virtual environment:
   ```sh
   conda env create -f environment.yml -n palynomorph_detection
   conda activate palynomorph_detection
   ```

## Data

**ndpi images (.ndpi)** : this project expects all ndpi files to be stored in a single directory. Nothing other than the ndpi files should exist in this directory. NOTE: ndpi files are usually very large (20-50GB), so sufficient space for the ndpi img files directory is essential. 

**ndpa annotation files (.ndpa)** : this project expects all ndpa files to be stored in a single directory, different from the ndpi images. Nothing other than the ndpa files should exist in this directory

Each ndpi img file has a corresponding ndpa annotation file that has information about the palynomorphs and where they are in the image. The names of each file for a particular specimen are the same, save their file extensions. 

For example: for D3283-2_2024_02_06_15_37_28_Kentucky,
- NDPI: D3283-2_2024_02_06_15_37_28_Kentucky.ndpi,
- NDPA: D3283-2_2024_02_06_15_37_28_Kentucky.ndpi.ndpa (the '.ndpi.ndpa' is intentional)

## SetUp
To set up the code, please run the following commands. The user will then be prompted to fill out the fields below to configure locations, and other settings:
- abs_path_to_ndpi_dir : Input the absolute path to the directory that holds all the ndpi images
- abs_path_to_ndpa_dir : Input the absolute path to the directory that holds all the ndpa annotation files
- abs_path_to_location_for_master_annotation_csv : Input the absolute path to the directory holding the master_annotation_csv to be created at the annotation preprocessing step. **NOTE: if running on a region with no annotations (in other words, running in a production like context), then user can leave this blank**
- abs_path_to_ndpi_tiles_dir : Input the absolute path to the directory that holds all tiled ndpi images
- tile_size : this is the tile size in pixels. Input only a single integer that represents the side length of the tile. Currently, only square tiles are supported.
- tile_overlap : Input an integer representing the pixel amount of overlap for the tiles
```
cd scripts
python setup.py
```
## Data Preprocessing

The order of the data preprocessing steps is important. The user must preprocess the images first before preprocessing the annotations. 
1. **Image preprocessing**: run the code below to tile a directory of ndpi images, with tiles stored at the directory specified by the user at setup.
```
cd scripts
python image_preprocessing.py
```
The file structure of the tiled ndpi images will look as follows:
```
all tiles directory/ # this is the directory for all tiles specified by the user at setup.
|     |------> filename1/ # the ndpi image that was tiled (i.e. D3283-2_2024_02_06_15_37_28_Kentucky)
|     |            |------> 0x_0y/ # the particular tile (the folder name is the top left coordinate of the tile
|     |            |           |------> 0z.png # a particular focal plane of the tile
|     |            |           |------> 1z.png
|     |            |           |------> ...
|     |            |           |------> 24z.png
|     |            |-------> 0x_2048y/
|     |            |           |------> 0z.png
|     |            |           |------> ...
|     |            |-------> ...
|     |------> filename2/
|     |            |------> ...
|     |------> ...
```
2. **Annotation preprocessing**: Now that all ndpi images have been tiled, the annoation preprocessing can be run. Run the code below to save all annotations of all the ndpa files into one master_annotation_csv file stored at the location specified by the user at setup. **NOTE: if running on a region with no annotations (in other words, running in a production like context), then user can ignore this step**
```
cd scripts
python annnotation_preprocessing.py
```
In particular, the csv saved from annotation preprocessing has the following structure
| File_name | annot_id | Palynomorph_type | Center | Radius | TL | BL | TR | BR | Tile_id |
|-----------|----------|------------------|--------|--------|----|----|----|----|---------|
|           |          |                  |        |        |    |    |    |    |         | 

Where TL, BL, TR, BR are the top-left, bottom-left, top-right, and bottom-right corners of the bounding box for the annotation, and tile_id is the name of the tile that the annotation exists in. 


## Exploratory Data Analysis
NOTE: python eda.py is being heavily edited right now so it does not quite up to the standards expected for the software check. Please forgive us. 

In order to run the script generating visualizations and summarizations of our data, run the following commands:
```
cd src/data_preprocessing
python eda.py
```

This step in the data science pipeline is to gain an understanding of the data we have, for example by understanding the distribution of categories and areas. 

The script will output in the console the following information and visualizations:
- Number of annotations
- Pie chart of the proportion of annotations per palynomorph type
- Histogram of the distribution of annotations per file
- Histogram of the distribution of average area of palynomorph type
- Box plot of the distribution of annotation areas per palynomorph type


## Modeling:
This project compares the performance of 3 models for the palynomorph detection task. To learn more about each model and how to run them, click on each of the links below to navigate to their respective ReadMe files nested within the src/modeling subdirectories. 
- DEtection TRansformer (DETR)
- [Faster R-CNN](https://github.com/RiceD2KLab/Smithsonian_fossil_Sp25/blob/main/src/modeling/faster_rcnn/README.md)
- [Pretrained U-Net Segmentation model](https://github.com/RiceD2KLab/Smithsonian_fossil_Sp25/tree/main/src/modeling/baseline), provided by [Punyasena Paleoecology and Palynology Lab](https://github.com/paleopollen)

## Evaluation:
To evaluate the performance of each of our models against ground truth annotations, we use Mean Average Prediction. Each model has its own methods of performing evaluation, so please see above links to learn more about how to conduct such evaluation.  

## Acknowledgments/Citations

This project uses the [ndpi-tile-cropper-cli](https://github.com/paleopollen/ndpi-tile-cropper-cli) repository, which provides a command-line interface for cropping NDPI tiles, and the [pollen-detection-cli](https://github.com/paleopollen/pollen-detection-cli) repository, which provides a segmentation model to detect pollens from cropped NDPI tiles. Special thanks to **Punyasena, S. W.**, & **Satheesan, S. P.** and the [Punyasena Paleoecology and Palynology Lab](https://github.com/paleopollen) for their work in these projects.

.


