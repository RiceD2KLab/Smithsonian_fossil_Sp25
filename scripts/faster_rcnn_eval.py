import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.faster_rcnn_eval import evaluate_faster_rcnn


def main():
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN predictions using mAP.")
    parser.add_argument('--iou_threshold', type=float, default=0.5, help='IoU threshold for true positive')
    
    args = parser.parse_args()

    mean_ap = evaluate_faster_rcnn(
        master_csv_path=config["abs_path_to_master_annotation_csv"],
        predictions_csv_path=args.pred_csv,
        iou_threshold=args.iou_threshold
    )
    
    print(f"\n✅ Mean Average Precision (mAP): {mean_ap:.4f}\n")


if __name__ == "__main__":
    main()
