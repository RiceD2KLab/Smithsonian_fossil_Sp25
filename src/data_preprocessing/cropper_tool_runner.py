from src.tools.ndpi_metadata_extractor import extract_ndpi_metadata
from src.tools.annotated_region_bounds_extractor import annotation_region_bounds_retrieval
from src.tools.coordinate_space_convertor import nanozoomer_to_pixelwise
from src import config
import subprocess
import os

"""
this function runs the tile cropper tool located in the /src/data_preprocessing/ndpi-tile_cropper submodule.
It runs the tile cropper on a directory of ndpi files, only on the annotated region of each ndpi. 

Input:
    - None

Returns:
    - None.
    - Action: the tile crops are stored at the location specified in "abs_path_to_ndpi_tiles_dir" in /src/config.json.
        Specifically, the structure is as follows:
        tiles_location/ # all ndpi tiles location specified in config.json
        |___ filename/ # specific file working on (i.e. D3283-2_2024_02_06_15_37_28_Kentucky)
        |    |___ 0x_0y/ # tiles for the specific filename. This tile name is the top left coordinate of the tile
        |    |    |___ 0z.png # the focal planes of the tiles from 0 to 24 (or however many focal planes exist in the ndpi file being used)
        |    |    |___ 1z.png
        |    |    |___ ...
        |    |    |___ 24z.png
        |    |___ 0x_2048y/
        |    |    |___ 0z.png
        |    |    |___ 1z.png
        |    |    |___ ...
        |    |    |___ 24z.png
        |    |___ ...x_...y/
        ...
"""
def annotated_region_cropper():
    for ndpi_file in os.listdir(config["abs_path_to_ndpi_dir"]):
        # retrieve the annotation region top-left and bottom-right corners in nanometers
        tl_x_nm, tl_y_nm, br_x_nm, br_y_nm = annotation_region_bounds_retrieval(os.path.join(config["abs_path_to_ndpi_dir"], ndpi_file))

        # convert region bounds retrieved above to pixel wise coordinate space
        tl_px = nanozoomer_to_pixelwise(tl_x_nm, tl_y_nm, os.path.join(config["abs_path_to_ndpi_dir"], ndpi_file))
        br_px = nanozoomer_to_pixelwise(br_x_nm, br_y_nm, os.path.join(config["abs_path_to_ndpi_dir"], ndpi_file))

        path_to_cropper_tool = os.path.join(os.path.dirname(__file__), 'ndpi-tile-cropper-cli', 'src', 'ndpi_tile_cropper_cli.py')

        # construct command to run cropper tool
        command = [
        "python",
        path_to_cropper_tool,
        "-i", os.path.join(config["abs_path_to_ndpi_dir"], ndpi_file),
        "-o", config["abs_path_to_ndpi_tiles_dir"],
        "-l", str(config["tile_overlap"]),
        "-s", str(config["tile_size"]),
        "-rsx", str(int(tl_px[0])),
        "-rex", str(int(br_px[0])),
        "-rsy", str(int(tl_px[1])),
        "-rey", str(int(br_px[1]))
        ]

        # run cropper command
        try: 
            subprocess.run(command, capture_output=True, text=True, check=True)
        except FileNotFoundError:
            print(f"Error: Command not found: {command[0]}")
        except subprocess.CalledProcessError as e:
            print(f"Error: Command '{e.cmd}' failed with return code {e.returncode}\n{e.stderr}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")