# Smithsonian_fossil_Sp25

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

This repository hosts the code for an automated machine-learning pipeline designed to detect and classify palynomorphs (fossil pollen, spores, and other organic-walled microorganisms) from high-resolution NDPI (NanoZoomer Digital Pathology Image) files. 

## Table of Contents
- [Introduction](#introduction)
- [File Structure](#file-structure)
- [Installation](#installation)
- [Data](#data)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Acknowledgments/Citations](#acknowledgmentscitations)

## Repo Structure


- `src/`
  - `data_preprocessing/` Scripts relevant to pre-processing the data
    - [`ndpi_tile_cropper/`](https://github.com/Jonathanwxlee/ndpi-tile-cropper-cli) Forked submodule of ndpi_tile_cropper (see acknowledgments)
  - `modeling/` Scripts relevant for modeling
  - `evaluation/` Scripts relevant to model evaluation
  
- `scripts/` Misc. scripts
- `README.md` Project documentation
- `requirements.txt` Package requirements


## Installation
### Python and Conda Installation
1. Clone the repository:
   ```sh
   git clone --recursive https://github.com/RiceD2KLab/Smithsonian_fossil_Sp25.git
   cd /Smithsonian_fossil_Sp25
   ```
2. Install Python (recommended: version 3.9 or greater)
3. Install Conda (if using virtual environments)
4. Set up a virtual environment:
   ```sh
   conda create --name myenv python=3.9
   conda activate myenv
   ```
5. Install dependencies from `requirements.txt`:
   ```sh
   pip install -r requirements.txt
   ```

## Data

Data received should come in two forms: ndpi img files, and ndpa annotation files. This project expects each set of files to be stored in their own directory. Specifically, all ndpi files stored in a directory, and all ndpa files stored in a different directory. The locations of the directories does not matter, as long as neither directory exists within the other. It is also important to note that ndpi files are very large, so sufficient space for the ndpi img files directory is essential. 

Per scanned and annotated slide, there are two files, ndpi which has the image, and an ndpa file that corresponds to the ndpi, which has information about the annotations and where they are in the image. 

At this current state in the project:
- ndpi files are stored in `/storage/hpc/work/smithsonian/ndpi_files`
- ndpa annotation files are stored in `/projects/dsci435/smithsonian_sp25/data/annotations`

Add info on EDA here. 


## Data Preprocessing (Still in progress)

### Creating a master "raw" annotations csv file from ndpa annotation file data

In order to convert all ndpa annotation data into a master raw annotation csv file, run the `annotations_to_csv.py` script using the following commands:
```
cd src/data_preprocessing
python annotations_to_csv.py
```
The current version of this script will access the ndpa files in the ndpa annotation file directory and create a csv with the following structure:
| file name |    id    | pol_type  |     x     |     y     |  radius   |
|-----------|----------|-----------|-----------|-----------|-----------|
|           |          |           |           |           |           |

Currently, the `annotations_to_csv.py` script outputs the resulting csv file in both the github directory as `projects/dsci435/smithsonian_sp25/data/all_annotations.csv` and in NOTS as `/home/ak136/Smithsonian_fossil_Sp25/src/data_preprocessing/all_annotations.csv`

Many of the annotations of the palynomorphs in our data are very specific. Because we are interested in only 7 broad categories of palynomorphs, we have been provided a dictionary of different specimen names and the category they belong to. The dictionary was provided to us by the Smithsonian as a csv file named `Dictionary_categories_NDPI_files.csv`. Using this, the `cleaning.py` script replaces the instances of specimen names to one of the seven categories, and stores them in the Github (and currently in NOTS as well). 

The script also adds and populates a `radius` column to the csv. 

To run this script, use the following command:
```
cd src/data_preprocessing
python cleaning.py
```

### Tiling the ndpi image(s)
To tile an ndpi image, run `/src/data_preprocessing/cropper_runner.py`. cropper_runner.py takes several command line arguments as defined below:
```
    --dir : a flag for running the cropper tool on a whole directory of ndpi imgs 
    --tile_coord_extract : a flag for only extracting jsons of the pixel-wise tile coordinates. No cropping is performed. 
    --input_file_path : The absolute path to the input ndpi img. IMPORTANT: Only use this if cropping a single ndpi img. 
    --ndpi_directory : The absolute path to a directory of ndpi imgs. IMPORTANT: Only use this if cropping all ndpi imgs in a directory.
    --annotations_directory : The absolute path to a directory of all the ndpa annotation files.
    --output_dir : output directory for results of tiles
    --tile_overlap : pixel-wise overlap amount of tiles
    --tile_size : pixel-wise size of the tiles
```

Given that the ndpi imgs and the ndpa annotation files are stored in the structure specified in the data section, run following commands below to tile a directory of ndpi imgs. 

```
cd src/data_preprocessing
python cropper_runner.py --dir --ndpi_directory PATH_TO_NDPI_DIRECTORY --annotations_directory PATH_TO_NDPA_DIRECTORY --output_dir PATH_TO_OUTPUT_TILES_DIRECTORY --tile_overlap 660 --tile_size 2048
```

### Creating a transformed master annotations csv. 
This step involves taking the annotation csv made previously, and adding features that may be useful for our machine learning models in the future. For example, below are some additional features:
- tl, bl, tr, br: adding features that store bounding box top-left, bottom-left, top-right, and bottom-right corner coordinates for each annotation (in the nanozoomer coordinate space).
- tile_id: assigning annotations their respective tiles they exist in, within the associated ndpi image. 
- loc_in_tile: calculating the pixel wise coordinates of the annotation (bounding box corners and center) relative to the tile the annotation exists in.
The resulting transformed master annotation csv will have the below structure:

| File Name | ID | Pol Type | X  | Y  | Radius | TL | BL | TR | BR | Tile ID | Loc in Tile |
|-----------|----|----------|----|----|--------|----|----|----|----|---------|-------------|
|           |    |          |    |    |        |    |    |    |    |         |             |

However, before performing the transformation of annotations, we must first retrieve the coordinates of each tile for each tiled ndpi img, in json format. The jsons will be stored in /projects/dsci435/smithsonian_sp25/data/annotation_region_tile_coordinates. To get these jsons, we can run the cropper_runner.py script as mentioned above, but add the `--tile_coord_extract` flag to the command line arguments. Below are the commands to run. 
```
cd src/data_preprocessing
python cropper_runner.py --dir --tile_coord_extract --ndpi_directory PATH_TO_NDPI_DIRECTORY --annotations_directory PATH_TO_NDPA_DIRECTORY --output_dir PATH_TO_OUTPUT_TILES_DIRECTORY --tile_overlap 660 --tile_size 2048
```
Once the tile coordinates for each ndpi have been extracted, we can perform the transformation of annotations. To do so, run /src/data_preprocessing/annotation_transformer.py. annotation_transformer.py takes several command line arguments as defined below:
```
    --annotation_filepath : path to the master raw annotation file 
    --annot_region_tile_coords_dir : directory to the tile coordinates of each of the ndpi files 
    --output_dir : directory location to save the transformed annotation file
    --tile_size : the side length (in pixels) of the square tile crop. 
```
Given that the ndpi imgs and the ndpa annotation files are stored in the structure specified in the data section, run following commands below to perform annotation transformation:
```
cd src/data_preprocessing
python annotation_transformer.py --annotation_filepath PATH_TO_MASTER_RAW_ANNOTATIONS_CSV --annot_region_tile_coords_dir /projects/dsci435/smithsonian_sp25/data/annotation_region_tile_coordinates --output_dir PATH_TO_OUTPUT_TRANSFORMED_CSV_DIRECTORY --tile_size 2048

NOTE: we will change the hardcoded absolute path. It was a last minute edit before the midterm code check
```


## Modeling: Work in Progress

## Acknowledgments/Citations

This project uses the [ndpi-tile-cropper-cli](https://github.com/paleopollen/ndpi-tile-cropper-cli) repository, which provides a command-line interface for cropping NDPI tiles. Special thanks to **Punyasena, S. W.**, & **Satheesan, S. P.** for their work.

.


