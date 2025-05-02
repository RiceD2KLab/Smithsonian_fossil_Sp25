import os
import json
import torch
from tqdm import tqdm

from model import load_model
from predict import run_on_tile_multi_focus
from ndpi_converter import convert_predictions_to_nanozoomer, write_predictions_to_ndpa
from dataset import TileDataset


def load_config(path="/projects/dsci435/smithsonian_sp25/Smithsonian_fossil_Sp25/src/config.json"):
    with open(path, "r") as f:
        return json.load(f)


def main():
    # Load configuration
    config = load_config()


    tiles_dir = config["abs_path_to_ndpi_tiles_dir"]
    annotation_csv = config["abs_path_to_location_for_master_annotation_csv"]

    os.makedirs(ndpa_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset (for tile metadata) and model
    dataset = TileDataset(
        image_root=tiles_dir,
        annotation_file=annotation_csv
    )
    num_classes = len(dataset.class_mapping) + 1
    
    # model = load_model("/path/to/your_trained_model.pth", num_classes=num_classes, device=device)

    # # Group tile_ids by NDPI filename
    # tile_map = {}
    # for tile_id in dataset.image_names:
    #     row = dataset.annotations[dataset.annotations["tile_id"] == tile_id].iloc[0]
    #     filename = row["filename"]
    #     tile_map.setdefault(filename, []).append(tile_id)

    # print(f"🧠 Found {len(tile_map)} NDPI files with tile groups")

    # # Loop through each NDPI and convert all detections to NDPA
    # for ndpi_filename, tile_ids in tqdm(tile_map.items(), desc="Processing NDPI files"):
    #     full_ndpi_path = os.path.join(ndpi_dir, f"{ndpi_filename}.ndpi")
    #     all_predictions = []

    #     for tile_id in tile_ids:
    #         boxes, labels, scores = run_on_tile_multi_focus(
    #             model=model,
    #             image_root=tiles_dir,
    #             filename=ndpi_filename,
    #             tile_id=tile_id,
    #             device=device,
    #             confidence_threshold=0.5,
    #             draw=False
    #         )

    #         if boxes is not None:
    #             converted = convert_predictions_to_nanozoomer(
    #                 pred_boxes=boxes,
    #                 pred_labels=labels,
    #                 pred_scores=scores,
    #                 ndpi_path=full_ndpi_path
    #             )
    #             all_predictions.extend(converted)

    #     if all_predictions:
    #         output_path = os.path.join(ndpa_dir, f"{ndpi_filename}.ndpa")
    #         write_predictions_to_ndpa(all_predictions, output_path)
    #     else:
    #         print(f"⚠️ No predictions found for {ndpi_filename}")


if __name__ == "__main__":
    main()
