import json
import os
import shutil
from PIL import Image
from torchvision import transforms

"""
This function prompts the user to define a couple different file system paths for:
    - baseline model outputs
    - re_formatted tile inputs (9 focal plane subset of original 25 tiles, and tiles resized to 1024)

Inputs: 
    - abs_path_to_baseline_config_json: string representing the absolute path to the baseline_config.json file

Returns: 
    - none
    - populates /.../modlin/baseline/baseline_config.json
"""
def config_setup(abs_path_to_baseline_config_json):
    # Define the path to config.json relative to this script
    print("Welcome to the setup script! Please provide the following configurations:\n")
    
    # prompt user for inputs while creating config dictionary
    baseline_config = {
        "abs_path_to_reformatted_tile_directory": input("Enter the absolute path to a directory to store reformatted tiles for baseline model: "),
        "abs_path_to_baseline_model_output": input("Enter the absolute path to the location for where to store the baseline model outputs: ")
    }
    with open(abs_path_to_baseline_config_json, 'w') as f:
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
    for ndpi_tiles in os.listdir(source_dir):
        ndpi_tiles_path = os.path.join(source_dir, ndpi_tiles)
        if not os.path.isdir(ndpi_tiles_path):
            continue
        
        # iterate over each tile
        for tile in os.listdir(ndpi_tiles_path):
            tile_path = os.path.join(ndpi_tiles_path, tile)
            if not os.path.isdir(tile_path):
                continue
            
            
            rel_path = os.path.relpath(tile_path, source_dir)
            dest_subdir = os.path.join(destination_dir, rel_path)
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
    path_to_baseline_model_weights = os.path.join(os.path.dirname(__file__), "bestValModel_encoder.paramOnly")

    command = [
        "python",
        path_to_baseline_model_script,
        "-m", path_to_baseline_model_weights,
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
    return
