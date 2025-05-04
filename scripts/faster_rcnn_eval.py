import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.faster_rcnn.faster_rcnn_eval import calculate_map
from src.modeling.faster_rcnn import load_config  


def main():
    config = load_config()

    mean_ap = calculate_map(
        gt_csv_path=config["abs_path_to_master_annotation_csv"],
        pred_csv_path="../src/evaluation/faster_rcnn/tmp/raw_predictions.csv",
        iou_threshold=0.5
    )
    
    print("\n✅ Evaluation Results:")
    print(f"  - Class-specific mAP: {mean_ap['class_specific_mAP']:.4f}")
    print(f"  - Class-agnostic mAP: {mean_ap['class_agnostic_mAP']:.4f}\n")
    


if __name__ == "__main__":
    main()
