import xml.etree.ElementTree as ET
import os
import numpy as np
import pandas as pd
# from argparse import ArgumentParser

# from src.SPECIES_TO_CATEGORIES import SPECIES_TO_CATEGORY

# erase later once we figure out species to category!!
SPECIES_TO_CATEGORY = SPECIES_TO_CATEGORY = {
"alg": "alg",
"alg?": "indet",
"alnipollenites": "pol",
"alnipollenites verus": "pol",
"alnus": "pol",
"amerospore": "fun",
"arecipites": "pol",
"arecipites reticulatus": "pol",
"arecipites tenuiexinous": "pol",
"arecipites?": "pol",
"azolla": "alg",
"betulaceae": "pol",
"betulaceae type": "pol",
"betulaceae/myricaceae": "pol",
"bipol": "bipol",
"bisaccate": "bipol",
"bisaccate pollen": "bipol",
"bombacacidites": "pol",
"botryococcus": "alg",
"bulbilspore": "fun",
"calamuspollenites": "pol",
"caryapollenites": "pol",
"caryapollenites prodromus": "pol",
"caryapollenites veripites": "pol",
"cicatricosisporites": "spo",
"cicatricosisporites dorogensis": "spo",
"clavatricolporites": "pol",
"cyathidites": "spo",
"deltoidospora": "spo",
"didymospore": "fun",
"din": "din",
"din?": "indet",
"di-triporites": "pol",
"ericipites": "pol",
"fun": "fun",
"germling": "fun",
"illexpollenites": "pol",
"inaperturopollenites": "pol",
"juglandaceae/betulaceae": "pol",
"juglans": "pol",
"laevigatosporites": "spo",
"laevigatosporites ovatus": "spo",
"laevigatosporites sp.": "spo",
"liliacidites": "pol",
"liliacidites?": "pol",
"maceopolipollenites": "pol",
"momipites": "pol",
"momipites wyomingensis": "pol",
"momipites?": "pol",
"monocolpites": "pol",
"monocolpites?": "pol",
"monocolpopollenites": "pol",
"monolete": "spo",
"monoporopollenites annulatus": "pol",
"monosulcites": "pol",
"paly": "indet",
"paly?": "indet",
"pediastrum": "alg",
"phragmospore": "fun",
"pistillipollenites": "pol",
"platycarya platycarioides": "pol",
"platycaryapollenites swasticoides": "pol",
"pol": "pol",
"pol?": "indet",
"psilamonocolpites": "pol",
"retipollenites": "pol",
"retitrescolpites": "pol",
"sabal": "pol",
"sabal?": "pol",
"spo": "spo",
"spo?": "indet",
"stereisporites?": "spo",
"tct": "indet",
"tetracolporopollenites": "pol",
"trichotomosulcites": "pol",
"tricolpites": "pol",
"tricolporites": "pol",
"triletes": "spo",
"triporopollenites": "pol",
"ulmipollenites undulosus": "pol",
"ulmipollenites undulosus": "pol",
"sapindaceae?": "pol",
"fraxinoipollenites": "pol",
"plicatopollis lunata": "pol",
"carya simplex": "pol",
"betula claripites": "pol",
"caryapollenites wodehousei": "pol",
"caryapollenites veripites": "pol",
"pediastrum": "alg",
"arecaceae": "pol",
"betula": "pol",
"bisaccate": "bipol",
"botryoccocus": "alg",
"calamuspollenites?": "pol",
"fraxinoipollenites": "pol",
"juglandaceae/betulaceae?": "pol",
"monocolpate": "pol",
"palyy": "indet",
"platycaryapollenites platycaryoides": "pol",
"retitricolpites": "pol",
"trilete": "spo",
"ulmipollenites": "pol",
"paly": "indet"
}



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
                    "id": int(viewstate.get("id")),
                    "pol_type": title,
                    "x": int(annotation.find("x").text),
                    "y": int(annotation.find("y").text),
                    "radius": int(annotation.find("radius").text)
                }
                # print(annotation_data)
                annotations.append(annotation_data)
    
    # convert list of dictionaries holding annotations across all files to dataframe
    return pd.DataFrame(annotations)



def rename_specimen_names(csv_file_path, output_dir):
    """
    Renames specific specimen names with one of seven palynomorph categories.
    Resulting CSV file wil be stored in output_dir.
    
    Args:
        csv_file_path (str): Path to master CSV file
        output_dir (str): Path to directory for updated CSV
    """

    print("n none pol_types AFTER lowercasing + replacement: %d" % len(annotations_df["pol_type"]))

    # get the categories dictionary csv
    categories_dict = SPECIES_TO_CATEGORY

    annotations_df['pol_type'] = annotations_df['pol_type'].str.strip()
    annotations_df['pol_type'] = annotations_df['pol_type'].str.lower()

    annotations_df.replace(categories_dict, inplace=True)
        
    # annotations_df["area"] = annotations_df.apply(lambda: np.pi * (row["radius"] ** 2), axis=1)


    annotations_df


    print("nunique pol_types AFTER lowercasing + replacement: %d" % annotations_df["pol_type"].nunique())

    print(annotations_df["pol_type"].unique())
    filtered_df = annotations_df[annotations_df['pol_type'].isnull()]
    print(filtered_df)




annotations_df = parse_ndpa_file("/projects/dsci435/smithsonian_sp25/data/annotations")

rename_specimen_names("beep", "boop")

### CHECK ON WHAT TO DO WITH THIS
# this is specific for the dataset we got from smithsonian
# if args.annotation_directory.equals("/projects/dsci435/smithsonian_sp25/data/annotations"):


print("changing the things")

annotations_df.loc[
    (annotations_df['file'] == 'D5410_1_R_2024_02_0911_03_13_Utah.ndpi.ndpa') & 
    (annotations_df["id"] == 47), 
    'pol_type'
    ] = 'pol'

annotations_df.loc[
    (annotations_df['file'] == 'Giraffe_24_2_43_R_Giraffe_2025_01_14_10_06_00.ndpi.ndpa') &
    (annotations_df["id"] == 44)
    , 'pol_type'
    ] = 'indet'

print("nunique pol_types AFTER lowercasing + replacement: %d" % annotations_df["pol_type"].nunique())

print(annotations_df["pol_type"].unique())


"""
if __name__ == "__main__":
    args = get_args()

    annotations_dir = args.annotations_directory
    # output_dir = args.output_dir
    
    parse_ndpa_file(annotations_dir)"
"""