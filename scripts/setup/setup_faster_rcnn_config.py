# scripts/setup_faster_rcnn_config.py

import os
import json


def setup_faster_rcnn_config():
    print("🛠  Setting up Faster R-CNN config...\n")
    
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'modeling', 'faster_rcnn','faster_rcnn_config.json'))


    if not os.path.exists(config_path):
        print("⚠️  Config file does not exist. Please make sure the path is correct.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    ndpi_tiles_path = input("📂 Enter the absolute path to your NDPI tiles: ").strip()
    #annotation_csv_path = input("📄 Enter the absolute path to the master annotation CSV: ").strip()
    ndpi_file = input("📂 Enter the absolute path to your NDPI file: ").strip()
    model_path = input("🧠 Enter the absolute path to the trained model (.pth): ").strip()
    output_dir = input("📂 Enter the absolute path to save the output files: ").strip()
    
    config["abs_path_to_ndpi_tiles"] = ndpi_tiles_path
    #config["abs_path_to_master_annotation_csv"] = annotation_csv_path
    config["abs_path_to_ndpi_file"] = ndpi_file
    config["abs_path_to_model"] = model_path
    config["abs_path_to_output_files"] = output_dir
   

    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"\n✅ Config updated successfully at: {config_path}")

if __name__ == "__main__":
    setup_faster_rcnn_config()
