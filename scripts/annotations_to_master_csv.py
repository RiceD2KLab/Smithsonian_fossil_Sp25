import xml.etree.ElementTree as ET
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser

annotations_df = pd.DataFrame()

# from src.SPECIES_TO_CATEGORIES import SPECIES_TO_CATEGORY


def get_args():
    """
    Gets command line arguments.
    """
    parser = ArgumentParser(description="get args for tile cropper")
    parser.add_argument("--annotations_directory", type=str, help="directory to all the annotations")
    # parser.add_argument("--output_dir", type=str, help="output directory for tiles")
    args = parser.parse_args()
    return args


def parse_ndpa_file(ndpa_dir_path):
    """
    Parses all .ndpa files in a ndpa_dir_path and extracts circle annotation data.
    
    Args:
        ndpa_dir_path (str): Path to directory with all .ndpa
    """
        
    # INTERNAL NOTE: use file path: /projects/dsci435/smithsonian_sp25/data/annotations
    annotations = []

    # gets all file and directory paths within ndpa_dir_path
    files = os.listdir(ndpa_dir_path)  

    file_idx = 0
    num_files = len(files)

    # for each of the files, get annotation data
    for file_name in files:
        file_idx += 1
        full_path = os.path.join(ndpa_dir_path, file_name)
        # print("\nProcessing [%d/%d]: %s" % (file_idx, num_files, full_path))

        tree = ET.parse(full_path)
        root = tree.getroot()

        for viewstate in root.findall("ndpviewstate"):
            annotation = viewstate.find("annotation")
            if annotation is not None and annotation.get("type") == "circle":
                
                # extract title
                title_element = viewstate.find("title")
                title = title_element.text if title_element is not None else "ERROR"  # Get the text of the title
                if title_element is None:
                    print("NONE")

                annotation_data = { 
                    "file": os.path.basename(file_name),
                    "id": viewstate.get("id"),
                    "pol_type": title,
                    "x": int(annotation.find("x").text),
                    "y": int(annotation.find("y").text),
                    "radius": int(annotation.find("radius").text)
                }
                # print(annotation_data)
                annotations.append(annotation_data)
    
    # convert list of dictionaries holding annotations across all files to dataframe
    annotations_df = pd.DataFrame(annotations)
    print(annotations_df.head())


def rename_specimen_names(csv_file_path, output_dir):
    """
    Renames specific specimen names with one of seven palynomorph categories.
    Resulting CSV file wil be stored in output_dir.
    
    Args:
        csv_file_path (str): Path to master CSV file
        output_dir (str): Path to directory for updated CSV
    """

    # HARDCODED FOR NOW
    # get the categories dictionary csv
    categories_dict = SPECIES_TO_CATEGORY

    annotations_df['pol_type'].str.lower().strip()
    annotations_df.replace(categories_dict)
        
    annotations_df["area"] = annotations_df.apply(lambda: np.pi * (row["radius"] ** 2), axis=1)


    annotations_df


    print("nunique pol_types AFTER lowercasing + replacement: %d" % annotations_df["pol_type"].nunique())

    print(annotations_df["pol_type"].unique())



if __name__ == "__main__":
    args = get_args()

    annotations_dir = args.annotations_directory
    # output_dir = args.output_dir
    
    parse_ndpa_file(annotations_dir)