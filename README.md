# Smithsonian_fossil_Sp25

## Table of Contents
- [Introduction](#introduction)
- [Repo Structure](#repo-structure)
- [Installation](#installation)
- [Data](#data)
- [Data Preprocessing](#data-preprocessing)
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


## Repo Structure


- `src/`
  - `data_preprocessing/` Scripts relevant to pre-processing the data and exploratory data analysis
  - `modeling/` Scripts relevant for modeling
  - `evaluation/` Scripts relevant to model evaluation
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

There is two types of data that should be inputted into the model: ndpi img files, and ndpa annotation files. This is because each ndpi img file has a corresponding ndpa annotation file that has information about the annotations and where they are in the image. 

This project expects each set of files to be stored in their own directory. Specifically, all ndpi files stored in a single directory, and all ndpa files stored in a different directory. The locations of the directories does not matter, as long as neither directory exists within the other. It is also important to note that ndpi files are usually very large (20-50GB), so sufficient space for the ndpi img files directory is essential. 

At this current state in the project:
- ndpi files are stored in NOTS at `/storage/hpc/work/smithsonian/ndpi_files`
- ndpa annotation files are stored in NOTS at `/projects/dsci435/smithsonian_sp25/data/annotations`

Key insights about our data include that there is significant class imbalance of palynomorph types, with only two of the seven categories ("pollen" and "indeterminate") comprising over half of the labels. For information about running exploratory data analysis, please proceed to the Data Preprocessing > Exploratory Data Analysis section.


## Data Preprocessing (Still in progress)

### Creating a master "raw" annotations csv file from ndpa annotation file data

In order to convert all ndpa annotation data into a master raw annotation csv file, run the `annotations_to_csv.py` script using the following commands:
```
cd src/data_preprocessing
python annotations_to_csv.py <ndpa_dir_path> <csv_output_dir_path>
```
The current version of this script will access the ndpa files in the given ndpa annotation file directory and create a csv with the following structure:
| file name |    id    | pol_type  |     x     |     y     |  radius   |
|-----------|----------|-----------|-----------|-----------|-----------|
|           |          |           |           |           |           |

Currently, the `annotations_to_csv.py` script outputs the resulting csv file in both the github directory as `projects/dsci435/smithsonian_sp25/data/all_annotations.csv` and in NOTS as `/home/ak136/Smithsonian_fossil_Sp25/src/data_preprocessing/all_annotations.csv`

Many of the annotations of the palynomorphs in our data are very specific. Because we are interested in only 7 broad categories of palynomorphs, we have been provided a dictionary of different specimen names and the category they belong to. The dictionary was provided to us by the Smithsonian as a csv file named `Dictionary_categories_NDPI_files.csv`. Using this, the `cleaning.py` script replaces the instances of specimen names to one of the seven categories, and stores them in the Github (and currently in NOTS as well). 

The script also adds and populates an `area` column to the csv. 

To run this script, use the following command:
```
cd src/data_preprocessing
python cleaning.py
```

### Exploratory Data Analysis
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
- `tl, bl, tr, br`: adding features that store bounding box top-left, bottom-left, top-right, and bottom-right corner coordinates for each annotation (in the nanozoomer coordinate space).
- `tile_id`: assigning annotations their respective tiles they exist in, within the associated ndpi image. 
- `loc_in_tile`: calculating the pixel wise coordinates of the annotation (bounding box corners and center) relative to the tile the annotation exists in.
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


