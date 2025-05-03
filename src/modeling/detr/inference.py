import torch
from config_extractor import load_config as detr_load_config
from config_extractor import load_config as project_load_config
from detr_utils import (
    batch_predict,
    apply_tile_level_nms,
    predictions_to_ndpa,
    initialize_model
)

def main(config_path="detr_config.json"):
    # Step 1: Load config
    config = detr_load_config(config_path)
    project_config = project_load_config("../detr_config.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 2: Load model + processor
    model, processor = initialize_model(
        config["model_name"],
        config["num_labels"],
        config["weights_path"],
        device
    )

    # Step 3: Predict on image tiles
    raw_preds = batch_predict(
        model,
        processor,
        image_dir=project_config["abs_path_to_ndpi_tiles_dir"],
        device=device,
        threshold=config["confidence_prediction_threshold"]
    )

    # Step 4: Apply NMS per tile group
    filtered_preds = apply_tile_level_nms(
        raw_preds,
        iou_threshold=config["nms_iou_threshold"]
    )

    # Step 5: Write NDPA files per slide
    predictions_to_ndpa(
        filtered_preds,
        ndpi_base_dir=project_config["abs_path_to_ndpi_dir"],
        output_dir=config["ndpa_output_dir"],
    )

    print("Pipeline complete.")

if __name__ == "__main__":
    main()
