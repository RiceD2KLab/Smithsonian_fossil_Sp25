import os
from entities.ndpa import Ndpa
from tools.coordinate_space_convertor import nanozoomer_to_pixelwise
from tools.ndpi_metadata_extractor import extract_ndpi_metadata
from src import config
from pathlib import Path
import csv
import json

"""
This function converts a given circular annotation to a rectangular bounding box. 
Particularly, it determines the four corner coordinates (in nanometers and in the nanozoomer coordinate space) of the bounding box. 

Inputs:
    - center_nm: a two element tuple representing the a coordinate in nanometers
    - radius_nm: integer representing the radius of the pollen grain annotation in nanometers

Returns:
    - tl: a two-element tuple representing the top left corner coordinate of the bounding box (See tuple format below) 
    - bl: a two-element tuple representing the bottom left corner coordinate of the bounding box (See tuple format below) 
    - tr: a two-element tuple representing the top right corner coordinate of the bounding box (See tuple format below)
    - br: a two-element tuple representing the bottome right corner coordinate of the bounding box (See tuple format below)
    TUPLE FORMAT: First element is x coordinate in nanometers, second element is y coordinate in nanometers.
"""
def bounding_box_generator(center_nm, radius_nm):
    tl = (center_nm[0] - radius_nm, center_nm[1] - radius_nm)
    bl = (center_nm[0] - radius_nm, center_nm[1] + radius_nm)
    tr = (center_nm[0] + radius_nm, center_nm[1] - radius_nm)
    br = (center_nm[0] + radius_nm, center_nm[1] + radius_nm)
    return tl, bl, tr, br

"""
This function assigns annotations their corresponding tiles based on their pixel wise location, and the tile coordinates 
Inputs:
    - c_x_px: the x coordinate of the annotation in pixels
    - c_y_px: the y coordinate of the annotation in pixels
    - radius_px: The radius of the annotation in pixels
    - filename: The name of the file currently working on. Should not end in .ndpi 

Returns:
    - within_tile_list: a list of strings that represent the tiles that full annotation can be found in. 
        In particular, the formatting of the strings is '#x_#y' where # represents the pixel wise coordinates of the top left corner of the tile. 
"""
def annotation_tile_determiner(c_x_px, c_y_px, radius_px, filename):
    within_tile_list = []
    curr_file_tiles_dir = Path(os.path.join(config["abs_path_to_ndpi_tiles_dir"], filename))
    all_tiles = [tile.name for tile in curr_file_tiles_dir.iterdir() if tile.is_dir()] # get a list of all the tiles. 
    for tile in all_tiles:
        topleft_x, topleft_y = map(int, tile[:-1].split('x_')) # parse the tile directory name to extract x and y pixel coordinate of top left tile corner
        # check if the annotation circle extends beyond the boundary of the current tile
        if not (((c_y_px - radius_px) < topleft_y) or # beyond top of tile
            ((c_y_px + radius_px) > (topleft_y + tile_size)) or # beyond bottom of tile
            ((c_x_px - radius_px) < topleft_x) or # beyond left side of tile
            ((c_x_px + radius_px) > topleft_x + tile_size)): # beyond right side of tile 
            within_tile_list.append(tile)
    return within_tile_list


"""
This function is responsible for generating the master annotation csv to be used eventually for modeling. 

Inputs:
    - None
Returns:
    - None; however it creates a master annotation csv file at the location specified at the "abs_path_to_location_for_master_annotation_csv" key
    in the config.json. 
    This master annotation csv includes the following features
        - filename: str representing the name of the file without any file extensions. i.e: "D3283-2_2024_02_06_15_37_28_Kentucky"
        - annot_id: str of an integer representing the id of the annotation in the ndpa file
        - paly_type: a string representing the palynomorph category
        - center: a string representation of a tuple representing the center coordinate (in pixels) of the annotation 
        - radius: a string representation of the radius of the annotation in pixels
        - TL: a string representation of a tuple representing the coordinate of the top-left corner of the annotation bounding box
        - BL: a string representation of a tuple representing the coordinate of the bottom-left corner of the annotation bounding box
        - TR: a string representation of a tuple representing the coordinate of the top-right corner of the annotation bounding box
        - BR: a string representation of a tuple representing the coordinate of the bottom-right corner of the annotation bounding box
        - tile_id: a string representing the tile the annotation exists in. the string format is '#x_#y' where # represents the pixel wise coordinates of the top left corner of the tile. 
"""
def create_master_annotation_csv():
    ndpa_directory = config["abs_path_to_ndpa_dir"]

    # open and load SPECIES_TO_CATEGORIES.json
    species_to_categories_path = os.path.join(os.path.dirname(__file__), 'SPECIES_TO_CATEGORIES.json')
    species_to_category = {}
    with open(species_to_categories_path, 'r') as f:
        species_to_category = json.load(f)

    with open(os.path.join(config["abs_path_to_location_for_master_annotation_csv"], "master_annotation.csv"), mode="w", newline="") as master_annotation_csv:
        headers = ["filename", "annot_id", "paly_type", "center", "radius", "TL", "BL", "TR", "BR", "tile_id"]
        writer = csv.DictWriter(master_annotation_csv, fieldnames=headers)
        writer.writeheader()

        for file in os.listdir(ndpa_directory):
            ndpa = Ndpa(os.path.join(ndpa_directory, file))
            for annot_id, annotation_obj in ndpa.annotations.items():
                specimen_name = file.rsplit('.', 2)[0]
                palynomorph_category = species_to_category[annotation_obj.label]
                radius_nm = annotation_obj.radius_nm
                center_nm = (annotation_obj.center_x_nm, annotation_obj.center_y_nm)
                tl_nm, bl_nm, tr_nm, br_nm = bounding_box_generator(center_nm, radius_nm)

                # convert from nanozoomer corodinate space to pixel-wise coordinate space
                center_px = nanozoomer_to_pixelwise(center_nm[0], center_nm[1], f"{os.path.join(config["abs_path_to_ndpi_dir"], specimen_name)}.ndpi")
                tl_px = nanozoomer_to_pixelwise(tl_nm[0], tl_nm[1], f"{os.path.join(config["abs_path_to_ndpi_dir"], specimen_name)}.ndpi")
                bl_px = nanozoomer_to_pixelwise(bl_nm[0], bl_nm[1], f"{os.path.join(config["abs_path_to_ndpi_dir"], specimen_name)}.ndpi")
                tr_px = nanozoomer_to_pixelwise(tr_nm[0], tr_nm[1], f"{os.path.join(config["abs_path_to_ndpi_dir"], specimen_name)}.ndpi")
                br_px = nanozoomer_to_pixelwise(br_nm[0], br_nm[1], f"{os.path.join(config["abs_path_to_ndpi_dir"], specimen_name)}.ndpi")
                
                # convert radius from nm to px
                ndpi_metadata = extract_ndpi_metadata(f"{os.path.join(config["abs_path_to_ndpi_dir"], specimen_name)}.ndpi")
                nmpp_x = float(ndpi_metadata["openslide.mpp-x"]) * 1000 # mutiply by 1000 to convert from millimeters to nanometers 
                radius_px = radius_nm // nmpp_x # just an arbitrary choice to use nm:px ratio for x direction instead of y

                # within_tile_list is a list of the tiles that the current annotation belongs in
                within_tile_list = annotation_tile_determiner(center_px[0], center_px[1], radius_px, specimen_name)
                
                for tile in within_tile_list:
                    row = {'filename':specimen_name, 'annot_id':str(annot_id), 'paly_type':palynomorph_category, 'center':str(center_px), 'radius':str(radius_px), 'TL':str(tl_px). 'BL':str(bl_px), 'TR':str(tr_px), 'BR':str(br_px), 'tile_id':tile}
                    writer.writerow(row)
    return

















































# from argparse import ArgumentParser
# import pandas as pd
# import os
# import csv
# import glob
# import json
# import openslide
# """
# Usage: 
# python3 annotation_transformer.py --annotation_filepath /projects/dsci435/smithsonian_sp25/data/cleaned_annotations.csv --annot_region_tile_coords_dir /projects/dsci435/smithsonian_sp25/data/annotation_region_tile_coordinates --output_dir /projects/dsci435/smithsonian_sp25/data --tile_size 2048

# EDIT NOTE: consider Numpy for efficient computations
# """

# """
# Command line parser
# Inputs: N/A (reads command line args)
# Return: 
#     - args: A namespace object containing the parsed arguments as attributes
# """
# def get_args():
#     parser = ArgumentParser(description="get args for tile cropper")
#     parser.add_argument("--annotation_filepath", type=str, help="path to the master annotation file")
#     parser.add_argument("--annot_region_tile_coords_dir", type=str, help="directory to all the tile crops for all .ndpi files")
#     parser.add_argument("--output_dir", type=str, help="directory to save the annotation file with all tiles")
#     parser.add_argument("--tile_size", type=int, help="the side length (in pixels) of the square tile crop")
#     args = parser.parse_args()
#     return args

# """
# This function reads necessary metadata for an inputted ndpi file

# Inputs: 
#     - input_ndpi_file_path: absolute path to the ndpi file to read metadata for. 

# Returns:
#     - a two-element tuple: first element is the x coordinate (in nm) of the center of the ndpi file. Second element is the y coordiante (in nm) of the center of the ndpi file. 
#     note: this coordinate location is relative to the whole nanozoomer scanned space
#     - ndpi_width_nm: the width of the ndpi file in nm
#     - ndpi_height_nm: the height of the ndpi file in nm
#     - mmpp_x: the millimeters per pixel in the x direction according to ndpi metadata
#     - mmpp_y: the millimeters per pixel in the y direction according to ndpi metadata
# """
# def read_ndpi_metadata(input_ndpi_file_path):
#     slide = openslide.OpenSlide(input_ndpi_file_path)
#     ndpi_center_x = int(slide.properties["hamamatsu.XOffsetFromSlideCentre"])
#     ndpi_center_y =int(slide.properties["hamamatsu.YOffsetFromSlideCentre"])
#     mmpp_x = float(slide.properties["openslide.mpp-x"])
#     mmpp_y = float(slide.properties["openslide.mpp-y"])
#     ndpi_width_px = int(slide.properties["openslide.level[0].width"])
#     ndpi_height_px = int(slide.properties["openslide.level[0].height"])
#     # file_no_extension = os.path.splitext(os.path.basename(input_ndpi_file_path))[0]
#     # metadata_path = f"/storage/hpc/work/dsci435/smithsonian_2/ndpi_metadatas/{file_no_extension}.json"
#     # # Load JSON file
#     # with open(metadata_path, 'r') as file:
#     #     metadata = json.load(file)
#     # # somehow find a way to read in the metadata
#     # ndpi_center_x = int(metadata["hamamatsu.XOffsetFromSlideCentre"])
#     # ndpi_center_y =int(metadata["hamamatsu.YOffsetFromSlideCentre"])
#     # mmpp_x = float(metadata["openslide.mpp-x"])
#     # mmpp_y = float(metadata["openslide.mpp-y"])
#     # ndpi_width_px = int(metadata["openslide.level[0].width"])
#     # ndpi_height_px = int(metadata["openslide.level[0].height"])

#     # convert ndpi width and height from pixels to nm
#     ndpi_width_nm = mmpp_x * 1000 * ndpi_width_px
#     ndpi_height_nm = mmpp_y * 1000 * ndpi_height_px

#     return (ndpi_center_x, ndpi_center_y), ndpi_width_nm, ndpi_height_nm, mmpp_x, mmpp_y

# """
# This function converts a given circular annotation to a rectangular bounding box. 
# Particularly, it determines the four corner coordinates (in nanometers) of the bounding box. 

# Inputs:
#     - center_x: the x coordinate of the pollen grain center in nanometers
#     - center_y: the y coordinate of the pollen grain center in nanometers
#     - radius: the radius of the pollen grain annotation in nanometers

# Returns:
#     - tl: a two-element tuple representing the top left corner coordinate of the bounding box (See tuple format below) 
#     - bl: a two-element tuple representing the bottom left corner coordinate of the bounding box (See tuple format below) 
#     - tr: a two-element tuple representing the top right corner coordinate of the bounding box (See tuple format below)
#     - br: a two-element tuple representing the bottome right corner coordinate of the bounding box (See tuple format below)
#     TUPLE FORMAT: First element is x coordinate in nanometers, second element is y coordinate in nanometers.
# """
# def bounding_box_generator(center_x, center_y, radius):
#     tl = (center_x - radius, center_y - radius)
#     bl = (center_x - radius, center_y + radius)
#     tr = (center_x + radius, center_y - radius)
#     br = (center_x + radius, center_y + radius)
#     return tl, bl, tr, br

# """
# This function applies transformations (as defined by the subfunctions) to an inputted coordinate (IN NANOMETERS)

# Inputs: 
#     NOTE: the x and y coordinate can be in either nanometers or pixels, but it MUST be consistent for both.
#     - x: the x coordinate of a point to be transformed
#     - y: the y coordinate of a point to be transformed
#     - input_ndpi_file_path: the path to the ndpi file there the point is resides.

# Returns:
#     - new_x: the transformed x coordinate of the point
#     - new_y: the transformed y coordinate of the point


# SUB-FUNCTIONS:
#     - translation: applies a translation to an inputted point
#         Inputs:
#              NOTE: the unit of measure of horiz. and vert. shift must be consistent with the units of the x and y coordinates!
#             - x: the x coordinate of a point to be transformed
#             - y: the y coordinate of a point to be transformed
#             - horizontal_shift: the number of units to shift in the x direction
#             - vertical_shift: the numbe rof units to shift in the y direction
#         Returns:
#             - new_x: the translated x coordinate in the same unit as input x
#             - new_y: the translated y coordinate in the same unit as input y
# """
# def coordinate_transform(x, y, input_ndpi_file_path):
#     def translation(x, y, horizontal_shift, vertical_shift):
#         new_x = x + horizontal_shift
#         new_y = y + vertical_shift
#         return new_x, new_y

#     ndpi_center_nm, ndpi_width, ndpi_height, _, _ = read_ndpi_metadata(input_ndpi_file_path)
#     top_left_x = ndpi_center_nm[0] - (ndpi_width // 2)
#     top_left_y = ndpi_center_nm[1] - (ndpi_height // 2) # keep in mind: this is SUBTRACTION because the upwards direction is (-)!
#     # shift right, shift down
#     new_x, new_y = translation(x, y, (-1) * top_left_x, (-1) * top_left_y) # always want to move in the opposite direction 
#     return new_x, new_y


# """
# This function assigns annotations their corresponding tiles based on their pixel wise location, and the tile coordinates 
# Inputs:
#     - c_x_px: the x coordinate of the annotation in pixels
#     - c_y_px: the y coordinate of the annotation in pixels
#     - radius_px: The radius of the annotation in pixels
#     - filename_without_extension: The name of the file currently working on. Should end in .ndpi 
#     - tile_coords_directory: directory for the tile coordinates (THIS DOES NOT HAVE A / AT THE END)

# Returns:
#     - tile_list: a list of strings that represent the tiles that full annotation can be found in. 
#         In particular, the formatting of the strings is '#x_#y' where # represents the pixel wise coordinates of the top left corner of the tile. 
# """
# def annotation_tile_determiner(c_x_px, c_y_px, radius_px, filename, tile_coords_dir):
#     print(f'{tile_coords_dir}/{filename}_tile_coordindates.json')
#     try:
#         with open(f'{tile_coords_dir}/{filename}_tile_coordindates.json', 'r') as file:
#             xy_dict_list = json.load(file)
#     except FileNotFoundError:
#         print(f"Error: tile_coordinates.json not found")
#         return None
#     except json.JSONDecodeError:
#         print(f"Error: Invalid JSON format in tile_coordinates.json")
#         return None
#     tile_list = []
#     for xy_dict in xy_dict_list:
#         xy_filename = f"{xy_dict['x']}x_{xy_dict['y']}y"
#         topleft_x, topleft_y = map(int, xy_filename[:-1].split('x_')) # parse the tile directory name to extract x and y pixel coordinate of top left tile corner
#         # check if the annotation circle extends beyond the boundary of the current tile
#         if not (((c_y_px - radius_px) < topleft_y) or # beyond top of tile
#             ((c_y_px + radius_px) > (topleft_y + tile_size)) or # beyond bottom of tile
#             ((c_x_px - radius_px) < topleft_x) or # beyond left side of tile
#             ((c_x_px + radius_px) > topleft_x + tile_size)): # beyond right side of tile 
#             tile_list.append(xy_filename)
#     return tile_list


# """
# NOTE: STILL IN PROGRESS
# This function retrieves the relative location of annotation center in its respective tile. 
# Inputs: 
#     - 

# Returns: 
#     - x: the pixel wise x coordinate of the annotation center
#     - y: the pixel wise y coordinate of the annotation center
# """
# def relative_location_in_tile():
#     x = 0
#     y = 0
#     return x, y

# if __name__ == "__main__":
#     args = get_args()
#     master_annotation_filepath = args.annotation_filepath
#     annot_region_tile_coords_dir = args.annot_region_tile_coords_dir
#     output_dir = args.output_dir
#     tile_size = args.tile_size

#     list_of_current_ndpis = os.listdir("/storage/hpc/work/dsci435/smithsonian/ndpi_files")

#     with open(master_annotation_filepath, mode="r", newline="") as master_annotation_csv, open(os.path.join(output_dir, "master_annotations_tiled.csv"), mode="w", newline="") as annotations_tiled:
#         reader = csv.DictReader(master_annotation_csv)
#         headers = reader.fieldnames
#         new_headers = headers + ["tl", "bl", "tr", "br", "tile"]
#         writer = csv.DictWriter(annotations_tiled, fieldnames=new_headers)
#         writer.writeheader()
#         # iterate over the master annotation file
#         for row in reader:
#             file_name = row["file"]
#             ndpi = file_name.removesuffix(".ndpa")
#             if ndpi not in list_of_current_ndpis:
#                 continue
#             print(ndpi)
#             ndpi_without_extension = ndpi.removesuffix(".ndpi")
#             print(ndpi_without_extension)
#             _, _, _, mmpp_x, mmpp_y = read_ndpi_metadata(f"/storage/hpc/work/dsci435/smithsonian/ndpi_files/{ndpi}")
#             nmpp_x = mmpp_x * 1000
#             nmpp_y = mmpp_y * 1000

#             radius_nano = int(row["radius"])
#             c_x_nano = int(row["x"])
#             c_y_nano = int(row["y"])

#             tl, bl, tr, br = bounding_box_generator(c_x_nano, c_y_nano, radius_nano)
            
#             c_x_nano_transformed, c_y_nano_transformed = coordinate_transform(c_x_nano, c_y_nano, f"/storage/hpc/work/dsci435/smithsonian/ndpi_files/{ndpi}")

#             # convert center X coordinate, center Y coordinate, and radius from nanometers to pixels
#             radius_px = radius_nano // nmpp_x
#             c_x_px = c_x_nano_transformed // nmpp_x
#             c_y_px = c_y_nano_transformed // nmpp_y
            
#             tile_list = annotation_tile_determiner(c_x_px, c_y_px, radius_px, ndpi, annot_region_tile_coords_dir)

#             for tile in tile_list:
#                 # new_row = row + [str(tl), str(bl), str(tr), str(br), tile]
#                 new_row = {**row, 'tl': str(tl), 'bl': str(bl), 'tr': str(tr), 'br': str(br), 'tile': tile}
#                 writer.writerow(new_row)
#         print(list_of_current_ndpis)
