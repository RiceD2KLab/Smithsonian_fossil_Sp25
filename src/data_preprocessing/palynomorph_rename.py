import xml.etree.ElementTree as ET
import pandas as pd
import csv
from argparse import ArgumentParser
import numpy as np


def get_args():
    """
    Gets command line arguments.
    """
    parser = ArgumentParser(description="get args for tile cropper")
    parser.add_argument("--annotations_directory", type=str, help="directory to all the annotations")
    parser.add_argument("--output_dir", type=str, help="output directory for tiles")
    args = parser.parse_args()
    return args


def rename_specimen_names(csv_file_path, output_dir):
    """
    Renames specific specimen names with one of seven palynomorph categories.
    Resulting CSV file wil be stored in output_dir.
    
    Args:
        csv_file_path (str): Path to master CSV file
        output_dir (str): Path to directory for updated CSV
    """

    # HARDCODED FOR NOW
    
    # get the annotation csv
    # FOR US: /home/ak136/Smithsonian_fossil_Sp25/src/data_preprocessing/all_annotations.csv
    annotations_df = pd.read_csv(csv_file_path)
    # get the categories dictionary csv
    categories_dict = {}
    with open('/home/ak136/Smithsonian_fossil_Sp25/src/data_preprocessing/Dictionary_categories_NDPI_files.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        line = 0
        for row in spamreader:
            if line > 0:
                try:
                    cat = row[0].lower()
                    spec = row[1].lower()
                    print("%s -- %s" % (cat, spec))

                    
                    if (spec in categories_dict.keys()) and categories_dict[spec] != cat:
                        print("Repeated specimen_name with inconsistent category!!!\nSpecimen: %s" % (spec))
                    else:
                        categories_dict[spec] = cat
                except: 
                    print(line)

                
            line += 1

        
    annotations_df["area"] = annotations_df.apply(lambda: np.pi * (row["radius"] ** 2), axis=1)


    annotations_df



    print("nunique pol_types BEFORE lowercasing + replacement: %d" % annotations_df["pol_type"].nunique())
    annotations_df["pol_type"] = annotations_df["pol_type"].str.lower()

    print("nunique pol_types AFTER lowercasing + BEFORE replacement: %d" % annotations_df["pol_type"].nunique())
    annotations_df["pol_type"] = annotations_df["pol_type"].replace(categories_dict)

    print("nunique pol_types AFTER lowercasing + replacement: %d" % annotations_df["pol_type"].nunique())

    print(annotations_df["pol_type"].unique())






# annotations_df.to_csv('/projects/dsci435/smithsonian_sp25/data/cleaned_annotations.csv', index=False)
# annotations_df.to_csv('/home/ak136/Smithsonian_fossil_Sp25/src/data_preprocessing/cleaned_annotations.csv', index=False)





if __name__ == "__main__":
    args = get_args()

    annotations_dir = args.annotations_directory
    output_dir = args.output_dir
    
    rename_specimen_names(annotations_dir, output_dir)