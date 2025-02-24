import pandas as pd
import numpy as np
import os

# get csv to pd
annotations_df = pd.read_csv('/projects/dsci435/smithsonian_sp25/data/all_annotations.csv')

#############################################################
########################## OVERALL ##########################
#############################################################

# how many annotations do we have? 
annotations_df.count


# number of annotations per slide
file_annotation_counts = annotations_df.groupby("file_name").size().reset_index(name="row_count")


# TODO change based on area calculation
# get area per annotation + add as column
def calculate_area(row):
    if row["annotation_type"] == "rectangle":
        return row["width"] * row["height"]
    elif row["annotation_type"] == "circle":
        return np.pi * (row["radius"] ** 2)
    else:
        return np.nan  # If annotation type is unknown
    
annotations_df["area"] = annotations_df.apply(calculate_area, axis=1)


# what percentage of data are we using? 


# Sum annotation areas per file
file_annotation_area = annotations_df.groupby("file_name")["annotation_area"].sum().reset_index()


#############################################################
########################## BY POLLEN ########################
#############################################################


