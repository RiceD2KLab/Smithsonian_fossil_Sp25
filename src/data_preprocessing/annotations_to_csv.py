import xml.etree.ElementTree as ET
import os

# import pandas as pd

# variables
annotations = []


# get the paths of the annotation files
path = "/projects/dsci435/smithsonian_sp25/data/annotations"
files = os.listdir(path)  # Returns both files and directories

files = files[0:2]

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
            annotation_data = {         # QUESTION: will we ever not have one of these? + does this have the annotation itself?
                "file": os.path.basename(file_name),
                "id": viewstate.get("id"),
                "x": int(annotation.find("x").text),
                "y": int(annotation.find("y").text),
                "radius": int(annotation.find("radius").text)
            }
            annotations.append(annotation_data)

        
# convert list of dictionaries holding annotations across all files to dataframe
annotations_df = pd.DataFrame(annotations)
print(annotations_df.head())

# put into csv
annotations_df.to_csv('/projects/dsci435/smithsonian_sp25/data/all_annotations.csv', index=False)

