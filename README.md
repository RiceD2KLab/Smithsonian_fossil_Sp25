# Smithsonian_fossil_Sp25


## Repository Structure

## Installation

## Running the Scripts


## Data


## Data Preprocessing

### Tiling the .ndpi image(s)
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

Given that the ndpi imgs and the ndpa annotation files are stored in the format above in the data section, run following commands below to tile a directory of ndpi imgs. 

```
cd /src/data_preprocessing
python cropper_runner.py --dir --ndpi_directory /storage/hpc/work/dsci435/smithsonian/ndpi_files --annotations_directory /projects/dsci435/smithsonian_sp25/data/annotations --output_dir /storage/hpc/work/dsci435/smithsonian/tiles --tile_overlap 660 --tile_size 2048
```

## Modeling: Work in Progress

## Acknowledgments/Citations

