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
This function assigns annotations their corresponding tiles based on their pixel wise location, and the tile coordinates. 
NOTE: this function assumes that the ndpi file you are working with has already been tiled, with the location of the tiles in "abs_path_to_ndpi_tiles_dir" in /src/config.json
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