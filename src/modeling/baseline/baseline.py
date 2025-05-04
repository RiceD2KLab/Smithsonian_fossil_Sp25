import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from PIL import Image
from torchvision import transforms
import subprocess

from src.modeling.baseline import baseline_config
from src import config
from src.tools.coordinate_space_convertor import pixelwise_to_nanozoomer
from src.evaluation.baseline.baseline_eval import extract_all_prediction_bboxes 

"""
This function prompts the user to define a couple different file system paths for:
    - baseline outputs
    - re_formatted tile inputs (9 focal plane subset of original 25 tiles, and tiles resized to 1024)

Inputs: 
    - None

Returns: 
    - none
    - creates baseline_outputs directory and subdirectories within it
    - populates /.../modling/baseline/baseline_config.json
"""
def baseline_config_setup():

    print("Welcome to the baseline model setup script! Please provide the following configurations:\n")
    
    # prompt user for inputs 
    abs_path_to_baseline_outputs_dir = input("Enter the absolute path to the directory that will hold the baseline model outputs: ")
    confidence_threshold_for_predictions = float(input("Enter a confidence threshold (decimal ex: 0.004) for making predictions: "))
    abs_path_to_model_weights = input("Enter the absolute path to the pretrained model weights")
    
    # create dirs and subdirs
    abs_path_to_baseline_outputs = os.path.join(abs_path_to_baseline_outputs_dir, "baseline_outputs")
    os.makedirs(abs_path_to_baseline_outputs, exist_ok=True)
    os.makedirs(os.path.join(abs_path_to_baseline_outputs, "prediction_ndpas"))
    os.makedirs(os.path.join(abs_path_to_baseline_outputs, "reformatted_tiles"))


    baseline_config = {
        "abs_path_to_baseline_outputs_dir": abs_path_to_baseline_outputs_dir,
        "abs_path_to_reformatted_tiles_directory": os.path.join(abs_path_to_baseline_outputs, "reformatted_tiles"),
        "abs_path_to_ndpa_output_directory": os.path.join(abs_path_to_baseline_outputs, "prediction_ndpas"),
        "confidence_threshold_for_predictions": confidence_threshold_for_predictions,
        "abs_path_to_baseline_model_outputs": os.path.join(abs_path_to_baseline_outputs, "model_outputs"),
        "abs_path_to_baseline_eval_results_dir": abs_path_to_baseline_outputs,
        "abs_path_to_model_weights": abs_path_to_model_weights
    }
    with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "baseline_config.json"), 'w') as f:
        json.dump(baseline_config, f, indent=4)
    
    return

"""
This function performs reformatting preparation to input data into the baseline model, and saves the reformatted tiles to the directory specified at 
"abs_path_to_reformatted_tile_directory" in /.../modeling/baseline/baseline_config.json. 
More specifically:
    - selects 9 focal planes from the initial 25. These focal planes are chosen evenly spaced out among the original 25. 
    - resizes the tiles to size 1024x1024

Inputs:
    - abs_path_to_original_tiles_dir: string representing the absolute path to the original tiles_dir
    - abs_path_to_reformatted_tile_dir: string representing the absolute path for the reformatted_tile_dir

Returns:
    - none
    - saves reformatted tiles to the directory specified at "abs_path_to_reformatted_tile_directory" in /.../modeling/baseline/baseline_config.json. 
"""
def baseline_input_preparation(abs_path_to_original_tiles_dir, abs_path_to_reformatted_tile_dir):
    """

    STEP 1: select 9 focal planes from the original 25 and save to abs_path_to_reformatted_tile_dir
    
    """
    valid_indices = set(str(i) for i in range(0, 25, 3))  # {'0', '3', '6', ..., '24'}
    
    # iterate over each of the ndpi's
    for ndpi_tiles in os.listdir(abs_path_to_original_tiles_dir):
        ndpi_tiles_path = os.path.join(abs_path_to_original_tiles_dir, ndpi_tiles)
        if not os.path.isdir(ndpi_tiles_path):
            continue
        
        # iterate over each tile
        for tile in os.listdir(ndpi_tiles_path):
            tile_path = os.path.join(ndpi_tiles_path, tile)
            if not os.path.isdir(tile_path):
                continue
            
            
            rel_path = os.path.relpath(tile_path, abs_path_to_original_tiles_dir)
            dest_subdir = os.path.join(abs_path_to_reformatted_tile_dir, rel_path)
            os.makedirs(dest_subdir, exist_ok=True) # make directory for each individual tile

            # iterate over each individual focal plane in the tile
            for file in os.listdir(tile_path):
                if file.endswith('z.png'):
                    try:
                        index = file.replace('z.png', '')
                        if index in valid_indices: # only choose the even spaced out focal planes
                            src_path = os.path.join(tile_path, file)
                            dst_path = os.path.join(dest_subdir, file)
                            shutil.copy2(src_path, dst_path)
                    except Exception as e:
                        print(f"Skipping file {file}: {e}")

    """

    STEP 2: resize all tiles to 1024 x 1024
    
    """
    # Resize transform
    resize_transform = transforms.Resize((1024, 1024))

    # Loop through all subdirectories and files
    for ndpi, tile, focal_planes in os.walk(abs_path_to_reformatted_tile_dir):
        for focal_plane in focal_planes:
            if focal_plane.endswith(".png"):
                focal_plane_path = os.path.join(ndpi, focal_plane)

                # Open image
                img = Image.open(focal_plane_path)

                # Apply resize transform
                img_resized = resize_transform(img)

                # Save the resized image back to the same path (overwrite)
                img_resized.save(focal_plane_path)
    return

"""
this function renames the directory that stores the baseline outputs

Input:
    - abs_path_to_baseline_model_outputs: String represenging the absolute path to the outputs of the baseline model

Returns:
    - None
    - renames the baseline model outputs directory to only what the use intends
        ex: usually, the baseline model outs a new directory called "baseline_outputs_2025-05-02_10-58-57", 
        where baseline_outputs is all the user specifies. But this function will rename is such that the 
        date and time portion of the directory name is removed. So, the renamed directory is only "baseline_outputs".

"""
def rename_baseline_outputs_dir(abs_path_to_baseline_model_outputs):
    # base_dir is the directory that holds the baseline model outputs directory
    # directory_name is the name of the baseline outputs directory
    base_dir, directory_name = os.path.split(abs_path_to_baseline_model_outputs)
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)

        # Match a directory named baseline_outputs_<timestamp>
        if os.path.isdir(full_path) and name.startswith("model_outputs_"):
            # New name with timestamp removed
            new_path = os.path.join(base_dir, directory_name)

            # Rename the directory
            os.rename(full_path, new_path)
            return  # Stop after first match if only one such folder exists



"""
this function runs baseline model on an input directory of tiles. It outputs the predictions to a specified output

Intputs:
    - abs_path_to_original_tiles_dir: string representing the absolute path to the original tiles_dir
    - abs_path_to_reformatted_tile_dir: string representing the absolute path for the reformatted_tile_dir

Returns:
    - None
    - Saves outputs to the specified directory

"""
def baseline_run(abs_path_to_reformatted_tile_dir, abs_path_to_detections_dir):
    path_to_baseline_model_script = os.path.join(os.path.dirname(__file__), "pollen-detection-cli", "src", "pollen_detection_cli.py")

    command = [
        "python",
        path_to_baseline_model_script,
        "-m", baseline_config["abs_path_to_model_weights"],
        "-c", abs_path_to_reformatted_tile_dir,
        "-d", abs_path_to_detections_dir
    ]

    try: 
        subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        print(f"Error: Command not found: {command[0]}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Command '{e.cmd}' failed with return code {e.returncode}\n{e.stderr}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # rename the output directory to only what the user specifies! See more info in the rename_baseline_outputs_dir docstring
    rename_baseline_outputs_dir(abs_path_to_detections_dir)
    return

"""
This function creates and populate all the subelements of an ndpviewstate element. This is specifically for ndpa creation for predictions

Inputs:
    - id: integer representing the ID of the annotation
    - confidence_score: this is a float that represents the condfidence score of the prediction
    - bbox: a list of 4 integers [x1, y1, x2, y2] where (x1, y1) is the top left coordinate, and (x2, y2) is the bottom right coordinate)
    - ndpi_sample_name: string representing the name of the ndpi file without the .ndpi extension 

Returns:
    - ndpviewstate: an ET.Element that includes all the information about the different 
"""
def create_ndpviewstate(id, confidence_score, bbox, ndpi_sample_name):
    ndpviewstate = ET.Element("ndpviewstate")
    ndpviewstate.set("id", str(id))
    ET.SubElement(ndpviewstate, "title")
    
    ET.SubElement(ndpviewstate, "details").text = f"confidence score: {str(confidence_score)}"
    
    ET.SubElement(ndpviewstate, "coordformat").text = "nanometers"

    ET.SubElement(ndpviewstate, "showtitle").text = "0"

    ET.SubElement(ndpviewstate, "showhistogram").text = "0"

    ET.SubElement(ndpviewstate, "showlineprofile").text = "0"

    annotation = ET.SubElement(ndpviewstate, "annotation", {
        "type":"freehand",
        "displayname":"AnnotateRectangle",
        "color": "#ff0000"
    })

    ET.SubElement(annotation, "measuretype").text = "0"
    ET.SubElement(annotation, "closed").text = "1"

    pointlist = ET.SubElement(annotation, "pointlist")
    
    # below are the points in pixelwise
    points = [
        (bbox[0], bbox[1]), # top left
        (bbox[0], bbox[3]), # bottom left
        (bbox[2], bbox[3]), # bottom right
        (bbox[2], bbox[1])  # top right
    ]

    # convert bounding box coordinates to nanozoomer format
    for i in range(4):
        points[i] = pixelwise_to_nanozoomer(points[i][0], points[i][1], os.path.join(config["abs_path_to_ndpi_dir"], f"{ndpi_sample_name}.ndpi"))
    
    # add each point to the ndpviewstate tree
    for x_val, y_val in points:
        point = ET.SubElement(pointlist, "point")
        ET.SubElement(point, "x").text = str(x_val)
        ET.SubElement(point, "y").text = str(y_val)

    ET.SubElement(annotation, "specialtype").text = "rectangle"

    return ndpviewstate


"""
This function saves the ndpa file output(s)

Inputs:
    - abs_path_to_detections_dir: a string representing the absolute path to the directory that stores the basline model's outputs
    - abs_path_to_ndpa_output_directory: a string representing the absolute path to the directory to store all the ndpa outputs. 
    - prediction_confidence threshold: the floating point number specified at confidence_threshold_for_predictions in baseline_config.json

Returns:
    - None
    - Saves n ndpa files to the directory specified at abs_path_to_ndpa_output_directory in baseline_config.json. 
        where n is the number of NDPI files originally making predictions for 
"""
def save_ndpas(abs_path_to_detections_dir, abs_path_to_ndpa_output_directory, prediction_confidence_threshold, abs_path_to_ndpi_dir):
    pred_boxes, pred_scores, pred_labels = extract_all_prediction_bboxes(abs_path_to_detections_dir, prediction_confidence_threshold)
    
    for ndpi_sample_name in pred_boxes.keys():
        # create a new root
        root = ET.Element("annotations")
        ndpi_bboxes = pred_boxes[ndpi_sample_name]
        ndpi_bboxes_prediction_scores = pred_scores.get(ndpi_sample_name, [])

        # attribute each bounding box with it's corresponding score
        id = 1
        for box, score in zip(ndpi_bboxes, ndpi_bboxes_prediction_scores):
            ndpviewstate = create_ndpviewstate(id, score, box, ndpi_sample_name)
            root.append(ndpviewstate)
            id += 1
        
        # save the whole tree to a location
        xml_str = ET.tostring(root, encoding="utf-8")
        pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")

        with open(os.path.join(abs_path_to_ndpa_output_directory, f"{ndpi_sample_name}.ndpi_predictions.ndpa"), "w", encoding="utf-8") as f:
            f.write(pretty_xml)
    return 
