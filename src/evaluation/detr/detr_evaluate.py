import torch
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.modeling.detr.config_extractor import load_config as detr_load_config
from src.modeling.detr.detr_utils import (
    initialize_model,
    evaluate_coco
)
from src import config as project_config

def main(config_path="src/modeling/detr/detr_config.json"):
    # Step 1: Load config
    config = detr_load_config(config_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 2: Load model + processor
    model, processor = initialize_model(
        model_name=config["model_name"],
        num_labels=config["num_labels"],
        weights_path=config["weights_path"],
        device=device
    )

    # Step 3: Evaluate
    evaluate_coco(
        model=model,
        processor=processor,
        image_dir=project_config["abs_path_to_ndpi_tiles_dir"],
        ann_json="tmp/pollen_val.json"
    )

    print("Pipeline complete.")

if __name__ == "__main__":
    main()
