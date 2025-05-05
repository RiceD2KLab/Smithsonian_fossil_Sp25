import json
import os

# Define the path to config.json relative to this script
CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'config.json'))

def prompt_user_for_config():
    print("Welcome to the setup script! Please provide the following configuration:\n")

    config = {
        "abs_path_to_ndpi_dir": input("Enter absolute path to directory of all ndpi images: "),
        "abs_path_to_ndpa_dir": input("Enter absolute path to directory of all ndpa annotation files: "),
        "abs_path_to_location_for_master_annotation_csv": input("Enter absolute path to the location for where to store the master annotation csv: "),
        "abs_path_to_ndpi_tiles_dir": input("Enter absolute path to directory for where to store all ndpi tiles: "),
        "tile_size": int(input("Enter tile size (type: integer): ")),
        "tile_overlap": int(input("Enter tile_overlap amount in pixels (type: integer): "))
    }

    return config

def save_config(config, config_file_path):
    with open(config_file_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"\n Config saved to: {config_file_path}")

def main():
    config_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'config.json'))
    config = prompt_user_for_config()
    save_config(config, config_file_path)

if __name__ == '__main__':
    main()
