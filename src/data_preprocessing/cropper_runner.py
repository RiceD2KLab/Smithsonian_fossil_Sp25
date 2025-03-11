import subprocess
import os
import tempfile
import sys
from argparse import ArgumentParser
import xml.etree.ElementTree as ET
import math
import bioformats
import javabridge
import json
import openslide


"""
ndpi storage location: /storage/hpc/work/dsci435/smithsonian
single usage
usage: python3 cropper_runner.py --input_file_path /storage/hpc/work/dsci435/smithsonian/ndpi_files/D9151-A-2_L_2024_02_02_16_08_52_Texas.ndpi --annotations_directory /projects/dsci435/smithsonian_sp25/data/annotations --output_dir /home/jwl9/test_cropper --tile_overlap 660 --tile_size 2048

whole directory usage
usage: python3 cropper_runner.py --dir --ndpi_directory /storage/hpc/work/dsci435/smithsonian/ndpi_files --annotations_directory /projects/dsci435/smithsonian_sp25/data/annotations --output_dir /home/jwl9/test_cropper --tile_overlap 660 --tile_size 2048

for tile_coordinate extraction
single usage:
    python3 cropper_runner.py --tile_coord_extract --input_file_path /storage/hpc/work/dsci435/smithsonian/ndpi_files/D9151-A-2_L_2024_02_02_16_08_52_Texas.ndpi --annotations_directory /projects/dsci435/smithsonian_sp25/data/annotations --output_dir /home/jwl9/test_cropper --tile_overlap 660 --tile_size 2048
"""

def get_args():
    parser = ArgumentParser(description="get args for tile cropper")
    parser.add_argument("--dir", action="store_true", help="flag for running cropper on a whole directory")
    parser.add_argument("--tile_coord_extract", action="store_true", help="flag for whether we actually want to crop the tiles or if we just want to get the coordinates of tiles")
    parser.add_argument("--input_file_path", type=str, help="input ndpi file")
    parser.add_argument("--ndpi_directory", type=str, help="represents the directory to all the ndpis")
    parser.add_argument("--annotations_directory", type=str, help="directory to all the annotations")
    parser.add_argument("--output_dir", type=str, help="output directory for tiles")
    parser.add_argument("--tile_overlap", type=int, help="pixel overlap amount")
    parser.add_argument("--tile_size", type=int, help="size of the tiles")
    args = parser.parse_args()
    return args

"""
This function reads necessary metadata for an inputted ndpi file

Inputs:
    - input_ndpi_file_path: the absolute path to the ndpi file path to read metadata for

returns:
    - a two-element tuple: first element is the x coordinate (in nm) of the center of the ndpi file. Second element is the y coordiante (in nm) of the center of the ndpi file. 
    note: this coordinate location is relative to the whole nanozoomer scanned space
    - ndpi_width_nm: the width of the ndpi file in nm
    - ndpi_height_nm: the height of the ndpi file in nm
    - mmpp_x: the millimeters per pixel in the x direction according to ndpi metadata
    - mmpp_y: the millimeters per pixel in the y direction according to ndpi metadata
"""
def read_ndpi_metadata(input_ndpi_file_path):
    slide = openslide.OpenSlide(input_ndpi_file_path)
    ndpi_center_x = int(slide.properties["hamamatsu.XOffsetFromSlideCentre"])
    ndpi_center_y =int(slide.properties["hamamatsu.YOffsetFromSlideCentre"])
    mmpp_x = float(slide.properties["openslide.mpp-x"])
    mmpp_y = float(slide.properties["openslide.mpp-y"])
    ndpi_width_px = int(slide.properties["openslide.level[0].width"])
    ndpi_height_px = int(slide.properties["openslide.level[0].height"])

    # file_no_extension = os.path.splitext(os.path.basename(input_ndpi_file_path))[0]
    # metadata_path = f"/storage/hpc/work/dsci435/smithsonian_2/ndpi_metadatas/{file_no_extension}.json"
    # # Load JSON file
    # with open(metadata_path, 'r') as file:
    #     metadata = json.load(file)
    # # somehow find a way to read in the metadata
    # ndpi_center_x = int(metadata["hamamatsu.XOffsetFromSlideCentre"])
    # ndpi_center_y =int(metadata["hamamatsu.YOffsetFromSlideCentre"])
    # mmpp_x = float(metadata["openslide.mpp-x"])
    # mmpp_y = float(metadata["openslide.mpp-y"])
    # ndpi_width_px = int(metadata["openslide.level[0].width"])
    # ndpi_height_px = int(metadata["openslide.level[0].height"])

    # convert ndpi width and height from pixels to nm
    ndpi_width_nm = mmpp_x * 1000 * ndpi_width_px
    ndpi_height_nm = mmpp_y * 1000 * ndpi_height_px

    return (ndpi_center_x, ndpi_center_y), ndpi_width_nm, ndpi_height_nm, mmpp_x, mmpp_y

"""
This function applies a translation to an inputted point
Inputs:
    NOTE: the unit of measure of horiz. and vert. shift must be consistent with the units of the x and y coordinates!
    - x: the x coordinate of a point to be transformed
    - y: the y coordinate of a point to be transformed
    - horizontal_shift: the number of units to shift in the x direction
    - vertical_shift: the number of units to shift in the y direction
Returns:
    - new_x: the translated x coordinate in the same unit as input x
    - new_y: the translated y coordinate in the same unit as input y
"""
def translation(x, y, horizontal_shift, vertical_shift):
    new_x = x + horizontal_shift
    new_y = y + vertical_shift
    return new_x, new_y


"""
This function perfrms a transformation on given points. 

Inputs: 
    - x and y should be the nanozoomer coordinate locations (x and y should be in nanometers). 
    - input_ndpi_file_path: absolute path to the ndpi file in question. 
Return:
    - new_x: transformed x coordinate in nm
    - new_y: transformed y coordinate in nm
"""
def point_transformation(x, y, input_ndpi_file_path):

    ndpi_center_nm, ndpi_width, ndpi_height, _, _ = read_ndpi_metadata(input_ndpi_file_path)
    top_left_x = ndpi_center_nm[0] - (ndpi_width // 2)
    top_left_y = ndpi_center_nm[1] - (ndpi_height // 2) # keep in mind: this is SUBTRACTION because the upwards direction is (-)!
    # shift right, shift down
    new_x, new_y = translation(x, y, (-1) * top_left_x, (-1) * top_left_y) # always want to move in the opposite direction 
    return new_x, new_y


"""
This function retrieves the bounds of the annotation region in nanometers. 
Input: 
    - input_ndpi_file_path: string representing the absolute path to the input ndpi file.
    - annotations_directory: string representing the directory to all the annotations
Output:
    - min_x_nano: the top left x coordinate of the annotated region in nanometers 
    - min_y_nano: the top left y coordiante of the annotated region in nanometers
    - max_x_nano: the bottom right x coordinate of the annotated region in nanometers
    - max_y_nano: the bottom right y coordinate of the annotated region in nanometers
"""
def annotation_region_bounds_retrieval(input_ndpi_file_path, annotations_directory):
    file_no_extension = os.path.splitext(os.path.basename(input_ndpi_file_path))[0]
    tree = ET.parse(f"{annotations_directory}/{file_no_extension}.ndpi.ndpa")
    root = tree.getroot()
    for viewstate in root.findall("ndpviewstate"):
        annotation = viewstate.find("annotation")
        if annotation is not None and annotation.get("displayname") == "AnnotateRectangle":
            pointlist = annotation.find("pointlist")
            if pointlist is not None:
                xs = []
                ys = []
                for point in pointlist.findall("point"):
                    xs.append(int(point.find('x').text))
                    ys.append(int(point.find('y').text))
                min_x_nano = min(xs)
                max_x_nano = max(xs)
                min_y_nano = min(ys)
                max_y_nano = max(ys)

                return min_x_nano, min_y_nano, max_x_nano, max_y_nano
                                

def run_ndpi_cropper_command(input_ndpi_file_path: str, annotations_directory: str, output_dir: str, tile_overlap: int, tile_size: int, tile_coord_extract:bool):
    """Runs the ndpi_tile_cropper CLI command with the given input file and outputs to a specified location."""
    _, _, _, mmpp_x, mmpp_y = read_ndpi_metadata(input_ndpi_file_path)
    nmpp_x = mmpp_x * 1000
    nmpp_y = mmpp_y * 1000
    
    min_x_nano, min_y_nano, max_x_nano, max_y_nano = annotation_region_bounds_retrieval(input_ndpi_file_path, annotations_directory)
    print("annotation region bounds below:")
    print(min_x_nano, min_y_nano, max_x_nano, max_y_nano)

    # apply transformations to bounds:
    min_x_nano, min_y_nano = point_transformation(min_x_nano, min_y_nano, input_ndpi_file_path)
    max_x_nano, max_y_nano = point_transformation(max_x_nano, max_y_nano, input_ndpi_file_path)

    # convert from nanometers to pixels
    rsx = math.floor(min_x_nano / nmpp_x)
    rex = math.ceil(max_x_nano / nmpp_x)
    rsy = math.ceil(min_y_nano / nmpp_y)
    rey = math.floor(max_y_nano / nmpp_y)

    print("translated region bounds in pixels below:")
    print(rsx, rsy, rex, rey)

    # Define the CLI command
    file = ""
    if tile_coord_extract:
        file = "ndpi-tile-cropper-cli/src/tile_cropper_coordinate_extractor.py"
    else:
        file = "ndpi-tile-cropper-cli/src/ndpi_tile_cropper_cli.py"
    command = [
        "python",
        file,
        "-i", input_ndpi_file_path,
        "-o", output_dir,
        "-l", str(tile_overlap),
        "-s", str(tile_size),
        "-rsx", str(int(rsx)),
        "-rex", str(int(rex)),
        "-rsy", str(int(rsy)),
        "-rey", str(int(rey))
    ]
    
    try:
        print("conversion finished, ready to run subprocess")
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print("subprocess finished")
        print(result.stdout)
    except FileNotFoundError:
        print(f"Error: Command not found: {command[0]}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Command '{e.cmd}' failed with return code {e.returncode}\n{e.stderr}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    args = get_args()
    directory_mode = args.dir
    tile_coord_extract = args.tile_coord_extract

    javabridge.start_vm(class_path=bioformats.JARS, run_headless=True)

    if not directory_mode: # single ndpi file mode
        input_ndpi = args.input_file_path
        annotations_directory = args.annotations_directory
        output_dir = args.output_dir
        tile_overlap = args.tile_overlap
        tile_size = args.tile_size
        run_ndpi_cropper_command(input_ndpi, annotations_directory, output_dir, tile_overlap, tile_size, tile_coord_extract)
    else: # crop all the ndpis in a directory
        ndpi_directory = args.ndpi_directory
        annotations_directory = args.annotations_directory
        output_dir = args.output_dir
        tile_overlap = args.tile_overlap
        tile_size = args.tile_size
        file_num = 1
        for ndpi_file in os.listdir(ndpi_directory):
            print(file_num)
            if ndpi_file.endswith(".ndpi"):
                # run the tile cropper
                ndpi_full_path = os.path.join(ndpi_directory, ndpi_file)
                run_ndpi_cropper_command(ndpi_full_path, annotations_directory, output_dir, tile_overlap, tile_size, tile_coord_extract)
            file_num += 1
    
    javabridge.kill_vm()