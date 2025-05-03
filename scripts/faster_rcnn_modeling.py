import torch
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modeling.faster_rcnn.model import load_model
from src.modeling.faster_rcnn.predict import run_on_tile_multi_focus
from src.tools.coordinate_space_convertor import convert_predictions_to_nanozoomer_for_faster_rcnn, write_predictions_to_ndpa_for_faster_rcnn
from src.modeling.faster_rcnn.dataset import TileDataset
from src.modeling.faster_rcnn import load_config




def main():
    # Load configuration
    config = load_config()

    tiles_dir = config["abs_path_to_ndpi_tiles"]
    ndpi_dir = config["abs_path_to_ndpi_files"]
    annotation_csv = config["abs_path_to_master_annotation_csv"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = config["abs_path_to_model"]


    # Load dataset (for tile metadata) and model
    dataset = TileDataset(
        image_root=tiles_dir,
        annotation_file=annotation_csv
    )
    num_classes = len(dataset.class_mapping) + 1
    model = load_model(model_path, num_classes=num_classes, device=device)
    print("MODEL LOADED")

    # Group tile_ids by NDPI filename
    tile_map = {}
    for tile_id in dataset.image_names:
        row = dataset.annotations[dataset.annotations["tile_id"] == tile_id].iloc[0]
        filename = row["filename"]
        tile_map.setdefault(filename, []).append(tile_id)

    print(f"🧠 Found {len(tile_map)} NDPI files with tile groups")

    # Loop through each NDPI and convert all detections to NDPA
    for ndpi_filename, tile_ids in tqdm(tile_map.items(), desc="Processing NDPI files"):
        print(ndpi_filename)
        print(tile_ids)
        full_ndpi_path = os.path.join(ndpi_dir, f"{ndpi_filename}.ndpi")
        print(full_ndpi_path)
        all_predictions = []

        for tile_id in tile_ids:
            print(tile_id)
            tile_path = os.path.join(tiles_dir, ndpi_filename)
            boxes, labels, scores = run_on_tile_multi_focus(
                model=model,
                filename=tile_path,
                tile_id=tile_id,
                device=device,
                confidence_threshold=0.5,
                draw=False
            )
            print(boxes)
            print(labels)
            print(scores)

            if boxes is not None:
                print("B")
                converted = convert_predictions_to_nanozoomer_for_faster_rcnn(
                    pred_boxes=boxes,
                    pred_labels=labels,
                    pred_scores=scores,
                    ndpi_path=full_ndpi_path
                )
                all_predictions.extend(converted)

        if all_predictions:
            print("MADE IT HERE")
            path = "/projects/dsci435/smithsonian_sp25/Smithsonian_fossil_Sp25/scripts"
            output_path = os.path.join(path, f"{ndpi_filename}.ndpa")
            write_predictions_to_ndpa_for_faster_rcnn(all_predictions, output_path)
        else:
            print(f"⚠️ No predictions found for {ndpi_filename}")


if __name__ == "__main__":
    main()
