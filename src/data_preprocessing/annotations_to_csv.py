import xml.etree.ElementTree as ET
import os
import pandas as pd

# variables
annotations = []

# get the paths of the annotation files
path = "/projects/dsci435/smithsonian_sp25/data/annotations"
files = os.listdir(path)  # Returns both files and directories

file_idx = 0
num_files = len(files)

# for each of the files, get annotation data
for file_name in files:
    file_idx += 1
    full_path = os.path.join(path, file_name)
    print("\nProcessing [%d/%d]: %s" % (file_idx, num_files, full_path))

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
# print(annotations_df.head())

# if all annotations have pollen type, make csv
if annotations_df[annotations_df["pol_type"] == "ERROR"].empty:
    print("Proceeding with making csv!")
    annotations_df.to_csv('/projects/dsci435/smithsonian_sp25/data/all_annotations.csv', index=False)
    annotations_df.to_csv('/home/ak136/Smithsonian_fossil_Sp25/src/data_preprocessing/all_annotations.csv', index=False)

