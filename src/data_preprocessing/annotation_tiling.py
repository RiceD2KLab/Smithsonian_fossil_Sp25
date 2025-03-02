from argparse import ArgumentParser
import pandas as pd
import os
import csv

"""
path for current master annotation csv: /projects/dsci435/smithsonian_sp25/data/cleaned_annotations.csv
all cropped tiles dir: /projects/dsci435/smithsonian_sp25/data/tile_imgs
output_dir: /projects/dsci435/smithsonian_sp25/data
nm_to_px_ratio: 229
tile_size: 1024

Usage: 
python3 annotation_tiling.py --annotation_filepath /projects/dsci435/smithsonian_sp25/data/cleaned_annotations.csv --all_cropped_tiles_dir /projects/dsci435/smithsonian_sp25/data/tile_imgs --output_dir /projects/dsci435/smithsonian_sp25/data --nm_to_px_ratio 229 --tile_size 1024
"""

def get_args():
    parser = ArgumentParser(description="get args for tile cropper")
    parser.add_argument("--annotation_filepath", type=str, help="path to the master annotation file")
    parser.add_argument("--all_cropped_tiles_dir", type=str, help="directory to all the tile crops for all .ndpi files")
    parser.add_argument("--output_dir", type=str, help="directory to save the annotation file with all tiles")
    parser.add_argument("--nm_to_px_ratio", type=float, help="the pixel to nanometer ratio")
    parser.add_argument("--tile_size", type=int, help="the side length (in pixels) of the square tile crop")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    master_annotation_filepath = args.annotation_filepath 
    all_cropped_tiles_dir = args.all_cropped_tiles_dir
    output_dir = args.output_dir
    nm_to_px_ratio = args.nm_to_px_ratio
    tile_size = args.tile_size

    with open(master_annotation_filepath, mode="r", newline="") as master_annotation_csv, open(os.path.join(output_dir, "master_annotations_tiled.csv"), mode="w", newline="") as annotations_tiled:
        reader = csv.DictReader(master_annotation_csv)
        writer = csv.writer(annotations_tiled)
        headers = reader.fieldnames
        new_headers = headers + ["tile"]
        writer.writerow(new_headers)
        # iterate over the master annotation file
        for row in reader:
            file_name = row["file"]
            radius_nano = int(row["radius"])
            c_x_nano = int(row["x"])
            c_y_nano = int(row["y"])
            
            # convert center X coordinate, center Y coordinate, and radius from nanometers to pixels
            radius_px = radius_nano / nm_to_px_ratio
            c_x_px = c_x_nano / nm_to_px_ratio
            c_y_px = c_y_nano / nm_to_px_ratio
            
            # iterate over all tile crops for the particular file
            for tile in os.listdir(os.path.join(all_cropped_tiles_dir, file_name.rsplit(".ndpi.ndpa", 1)[0])):
                topleft_x, topleft_y = map(int, s[:-1].split('x_')) # parse the tile directory name to extract x and y pixel coordinate of top left tile corner
                # check if the annotation circle extends beyond the boundary of the current tile
                if not (((c_y_px - radius_px) < topleft_y) or 
                    ((c_y_px + radius_px) > (topleft_y + tile_size)) or 
                    ((c_x_px - radius_px) < topleft_x) or
                    ((c_x_px + radius_px) > topleft_x + tile_size)):
                    annotation_row_with_tile = row + [tile]
                    writer.writerow(annotation_row_with_tile)
