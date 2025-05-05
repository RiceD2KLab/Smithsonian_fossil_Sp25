import torch
import os
import sys
from tqdm import tqdm
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modeling.faster_rcnn.model import load_model
from src.modeling.faster_rcnn.predict import run_on_tile_multi_focus
from src.tools.faster_rcnn_predictions_to_ndpa import convert_predictions_to_nanozoomer_for_faster_rcnn, write_predictions_to_ndpa_for_faster_rcnn, parse_tile_id, write_predictions_to_csv
from src.modeling.faster_rcnn.dataset import TileDataset
from src.modeling.faster_rcnn import load_config

def get_all_tile_ids(tile_root_dir):
    tile_ids = []
    for folder_name in os.listdir(tile_root_dir):
        full_folder = os.path.join(tile_root_dir, folder_name)
        if os.path.isdir(full_folder):
            for file_name in os.listdir(full_folder):
                if file_name.endswith(".png"):
                    tile_id = os.path.splitext(file_name)[0]
                    tile_ids.append((folder_name, tile_id))  # folder is ndpi_filename
    return tile_ids


def main():
    print("🚀 Initializing Faster R-CNN Inference Pipeline...\n")

    # Load configuration
    config = load_config()
    print("📄 Loaded configuration.")

    tiles_dir = config["abs_path_to_ndpi_tiles"]
    ndpi_dir = config["abs_path_to_ndpi_files"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = config["abs_path_to_model"]
    class_mapping = {'alg': 1, 'fun': 2, 'pol': 3, 'spo': 4, 'din': 5, 'Bipol': 6, 'paly': 7}

    print(f"🧠 Using device: {device.upper()}")
    print(f"📦 Loading model from: {model_path}\n")

    num_classes = 8 # Adjust this based on your dataset
    print(f"📚 Detected {num_classes} classes: {class_mapping}\n")

    model = load_model(model_path, num_classes=num_classes, device=device)
    print("✅ Model loaded successfully.\n")
    
    tile_index = get_all_tile_ids(tiles_dir)
    print(f"🔍 Found {len(tile_index)} tile images to process.")
    
    tile_map = {}
    for ndpi_file, tile_id in tile_index:
        tile_map.setdefault(ndpi_file, []).append(tile_id)
    
    # Process each NDPI group
    for ndpi_filename, tile_ids in tqdm(tile_map.items(), desc="🧪 Processing NDPI files"):
        full_ndpi_path = os.path.join(ndpi_dir, f"{ndpi_filename}.ndpi")
        all_predictions = []
        raw_predictions = []

        print(f"\n🗂️  NDPI File: {ndpi_filename} | Tiles: {len(tile_ids)}")

        for tile_id in tqdm(tile_ids, desc=f"🔬 Analyzing tiles for {ndpi_filename}", leave=False):
            tile_path = os.path.join(tiles_dir, ndpi_filename)
            boxes, labels, scores = run_on_tile_multi_focus(
                model=model,
                filename=tile_path,
                tile_id=tile_id,
                device=device,
                confidence_threshold=0.75,
                draw=False
            )

            if boxes is not None:
                
                for i in range(boxes.shape[0]):
                    xmin, ymin, xmax, ymax = boxes[i].tolist()
                    pred = {
                        "label": int(labels[i].item()),
                        "score": float(scores[i].item()),
                        "xmin_px": xmin,
                        "ymin_px": ymin,
                        "xmax_px": xmax,
                        "ymax_px": ymax,
                        "tile_id": tile_id
                    }
                    raw_predictions.append(pred)
                    
                tile_offset_px = parse_tile_id(tile_id)
                    
                converted = convert_predictions_to_nanozoomer_for_faster_rcnn(
                    pred_boxes=boxes,
                    pred_labels=labels,
                    pred_scores=scores,
                    ndpi_path=full_ndpi_path,
                    tile_offset_px=tile_offset_px
                )
                all_predictions.extend(converted)

        if all_predictions:
            print(f"{raw_predictions} \n")

            output_csv_path = f"../../src/evaluation/faster_rcnn/tmp/raw_predictions.csv"
            write_predictions_to_csv(
                predictions=raw_predictions,
                output_csv_path=output_csv_path,
                class_mapping=class_mapping,
                filename=ndpi_filename
            )

        
            output_dir = config["abs_path_to_output_files"]
            os.makedirs(output_dir, exist_ok=True)
            print(f"📂 Saving predictions to: {output_dir}")
            output_path = os.path.join(output_dir, f"{ndpi_filename}.ndpi.ndpa")
            write_predictions_to_ndpa_for_faster_rcnn(all_predictions, output_path, class_mapping=class_mapping)
        else:
            print(f"⚠️ No predictions found for {ndpi_filename}")

    print("\n🎉 All NDPI files processed. NDPA annotations generated.\n")


if __name__ == "__main__":
    main()