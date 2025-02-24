import subprocess
import os
import tempfile
import sys

def run_ndpi_cropper_command(input_file: str):
    """Runs the ndpi_tile_cropper CLI command with the given input file and outputs to a temporary folder."""
    
    # Define temporary output directory within the repo
    temp_output_dir = os.path.join(tempfile.gettempdir(), "ndpi_output")
    os.makedirs(temp_output_dir, exist_ok=True)
    
    # Define the CLI command
    command = [
        "python",
        "external/ndpi_tile_cropper/src/ndpi_tile_cropper_cli.py",
        "-i", input_file,
        "-o", temp_output_dir
    ]
    
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        print(result.stdout)
    except FileNotFoundError:
        print(f"Error: Command not found: {command[0]}")
    except subprocess.CalledProcessError as e:
        print(f"Error: Command '{e.cmd}' failed with return code {e.returncode}\n{e.stderr}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ndpi_to_png_tiles.py <input_file>")
        sys.exit(1)
    
    input_file_path = sys.argv[1]
    run_ndpi_cropper_command(input_file_path)
