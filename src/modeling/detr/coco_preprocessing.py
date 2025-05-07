from detr_utils import (
    convert_csv_to_coco,
    split_by_tile_id,
    filter_category
)
import sys
import os
from src.modeling.detr.config_extractor import load_config as detr_load_config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from src import config as project_config

def main(config_path="src/modeling/detr/detr_config.json"):
    config = detr_load_config(config_path)

    convert_csv_to_coco(
        csv_path=project_config["abs_path_to_location_for_master_annotation_csv"],
        image_root=project_config["abs_path_to_ndpi_tiles_dir"],
        output_json = f"{config['output_dir']}/pollen_dataset.json",
        focal_length=config['focal_length']
    )

    print("Finished converting csv to coco format")

    filter_category(
        input_json=f"{config['output_dir']}/pollen_dataset.json",
        output_json=f"{config['output_dir']}/filtered_pollen_dataset.json",
        exclude_label="indet"
    )

    split_by_tile_id(
        coco_json=f"{config['output_dir']}/filtered_pollen_dataset.json",
        train_json=f"{config['output_dir']}/pollen_train.json",
        val_json=f"{config['output_dir']}/pollen_val.json",
        val_ratio=config["val_split"],
        seed=config["seed"]
    )
    

if __name__ == "__main__":
    main()
