import pandas as pd
import numpy as np

def parse_coords(s):
    """Convert coordinate string '(x, y)' into a tuple of floats."""
    return tuple(map(float, s.strip("()").split(", ")))

def calculate_iou(box1, box2):
    """Compute IoU between two boxes: (xmin, ymin, xmax, ymax)."""
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def calculate_map(gt_csv_path, pred_csv_path, iou_threshold=0.5):
    gt_df = pd.read_csv(gt_csv_path)
    pred_df = pd.read_csv(pred_csv_path)

    # Parse bounding boxes
    gt_df["TL"] = gt_df["TL"].apply(parse_coords)
    gt_df["BR"] = gt_df["BR"].apply(parse_coords)
    pred_df["TL"] = pred_df["TL"].apply(parse_coords)
    pred_df["BR"] = pred_df["BR"].apply(parse_coords)

    # Group by tile_id
    ground_truth = {}
    for _, row in gt_df.iterrows():
        tile_id = row["tile_id"]
        bbox = (*row["TL"], *row["BR"])
        label = row["paly_type"]
        
        ground_truth.setdefault(tile_id, []).append({"bbox": bbox, "label": label})
        
    predictions = {}
    for _, row in pred_df.iterrows():
        tile_id = row["tile_id"]
        bbox = (*row["TL"], *row["BR"])
        label = row["paly_type"]
        score = row.get("score", 1.0)  # default to 1.0 if missing
        predictions.setdefault(tile_id, []).append({"bbox": bbox, "label": label, "score": score})

    # Calculate AP per tile
    aps = []
    for tile_id, gt_boxes in ground_truth.items():
        preds = predictions.get(tile_id, [])
        preds = sorted(preds, key=lambda x: x["score"], reverse=True)

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        matched_gt = set()

        for i, pred in enumerate(preds):
            max_iou = 0
            max_j = -1
            for j, gt in enumerate(gt_boxes):
                if j in matched_gt:
                    continue
                iou = calculate_iou(pred["bbox"], gt["bbox"])
                if iou > max_iou:
                    max_iou = iou
                    max_j = j
            if max_iou >= iou_threshold:
                tp[i] = 1
                matched_gt.add(max_j)
            else:
                fp[i] = 1

        # Compute precision-recall curve
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
        recalls = tp_cum / len(gt_boxes)

        # Compute AP (11-point interpolation)
        ap = 0
        for t in np.linspace(0, 1, 11):
            p = precisions[recalls >= t]
            ap += max(p) if len(p) > 0 else 0
        ap /= 11
        aps.append(ap)

    # Compute mAP
    mean_ap = np.mean(aps) if aps else 0
    return mean_ap
