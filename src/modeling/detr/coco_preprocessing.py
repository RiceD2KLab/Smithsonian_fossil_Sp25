from detr_utils import (
    convert_csv_to_coco,
    split_by_tile_id,
    filter_category
)
from src.modeling.detr.config_extractor import load_config as detr_load_config
from src import config as project_config


def main(config_path="config.json"):
    config = detr_load_config(config_path)

    convert_csv_to_coco(
    csv_path=project_config["abs_path_to_location_for_master_annotation_csv"],
    image_root=project_config["abs_path_to_ndpi_tiles_dir"],
    output_json="./tmp/pollen_dataset.json"
    )

    filter_category(
        input_json="./tmp/pollen_dataset.json",
        output_json="./tmp/filtered_pollen_dataset.json",
        exclude_label="indet"
    )

    split_by_tile_id(
    coco_path="./tmp/filtered_pollen_dataset.json",
    train_output_path="./tmp/pollen_train.json",
    val_output_path="./tmp/pollen_val.json",
    val_ratio=config["val_split"],
    seed=config["seed"]
    )
    

if __name__ == "__main__":
    main()