import xml.etree.ElementTree as ET
import pandas as pd
import csv
import numpy as np

# get the annotation csv
annotations_df = pd.read_csv("/home/ak136/Smithsonian_fossil_Sp25/src/data_preprocessing/all_annotations.csv")
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



print("nunique pol_types BEFORE lowercasing + replacement: %d" % annotations_df["pol_type"].nunique())
annotations_df["pol_type"] = annotations_df["pol_type"].str.lower()

print("nunique pol_types AFTER lowercasing + BEFORE replacement: %d" % annotations_df["pol_type"].nunique())
annotations_df["pol_type"] = annotations_df["pol_type"].replace(categories_dict)

print("nunique pol_types AFTER lowercasing + replacement: %d" % annotations_df["pol_type"].nunique())

print(annotations_df["pol_type"].unique())



# get area per annotation + add as column
def calculate_area(row):
    return np.pi * (row["radius"] ** 2)
annotations_df["area"] = annotations_df.apply(calculate_area, axis=1)


annotations_df



# annotations_df.to_csv('/projects/dsci435/smithsonian_sp25/data/cleaned_annotations.csv', index=False)
# annotations_df.to_csv('/home/ak136/Smithsonian_fossil_Sp25/src/data_preprocessing/cleaned_annotations.csv', index=False)



