# Smithsonian_fossil_Sp25


## Repository Structure

## Installation
# README

## Introduction

## Repository Structure
### Organization of Code Structure

## Installation
### Python and Conda Installation
1. Clone the repository:
   ```sh
   git clone <repository_url>
   cd <repository_name>
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

## Running the Scripts


## Data


## Data Preprocessing

### Creating a master "raw" annotations csv file from ndpa annoataion file data
AJ: instructions on which scripts/how to run scripts to get a master raw annotations csv file

### Tiling the ndpi image(s)
To tile an ndpi image, run /src/data_preprocessing/cropper_runner.py. cropper_runner.py takes several command line arguments as defined below:
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
python cropper_runner.py --dir --ndpi_directory /storage/hpc/work/dsci435/smithsonian/ndpi_files --annotations_directory /projects/dsci435/smithsonian_sp25/data/annotations --output_dir /storage/hpc/work/dsci435/smithsonian/tiles --tile_overlap 660 --tile_size 2048
```

### Creating a transformed master annotations csv. 
This step involves taking the annotation csv made previously, and adding features that may be useful for our machine learning models in the future. For example, below are some additional features:
- tl, bl, tr, br: adding features that store bounding box corner coordinates for each annotation (in the nanozoomer coordinate space), rather than solely having a center coordinate and a radius. (where tl, bl, tr, and br stand for top-left, bottom-left, top-right, and bottom-right respectively).
- tile_id: assigning annotations with their respective tiles they exist in, after the tile the ndpi image they exist in. 
- Tile_location: calculating the coordinates of the annotation (bounding box corners and center) relative to the tile the annotation exists in.

Before performing the transformation of annotations, we must first retrieve the coordinates of each tile for each tiled ndpi img, in json format. The jsons will be stored in /projects/dsci435/smithsonian_sp25/data/annotation_region_tile_coordinates. To get these jsons, we can run the cropper_runner.py script as mentioned above, but add the `--tile_coord_extract` flag to the command line arguments. Below are the commands to run. 
```
cd src/data_preprocessing
python cropper_runner.py --dir --tile_coord_extract --ndpi_directory /storage/hpc/work/dsci435/smithsonian/ndpi_files --annotations_directory /projects/dsci435/smithsonian_sp25/data/annotations --output_dir /storage/hpc/work/dsci435/smithsonian/tiles --tile_overlap 660 --tile_size 2048
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
python annotation_transformer.py --annotation_filepath /projects/dsci435/smithsonian_sp25/data/cleaned_annotations.csv --annot_region_tile_coords_dir /projects/dsci435/smithsonian_sp25/data/annotation_region_tile_coordinates --output_dir /projects/dsci435/smithsonian_sp25/data --tile_size 2048
```


## Modeling: Work in Progress

## Acknowledgments/Citations

