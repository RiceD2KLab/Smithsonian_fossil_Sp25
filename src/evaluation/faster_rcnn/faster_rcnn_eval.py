import pandas as pd
import numpy as np

def parse_coords(s):
    return tuple(map(float, s.strip("()").split(", ")))

def calculate_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0

def calculate_ap(tp, fp, n_gt):
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
    recalls = tp_cum / (n_gt + 1e-6)
    ap = 0
    for t in np.linspace(0, 1, 11):
        p = precisions[recalls >= t]
        ap += max(p) if len(p) > 0 else 0
    return ap / 11

def calculate_map(gt_csv_path, pred_csv_path, iou_threshold=0.5):
    gt_df = pd.read_csv(gt_csv_path)
    pred_df = pd.read_csv(pred_csv_path)

    gt_df["TL"] = gt_df["TL"].apply(parse_coords)
    gt_df["BR"] = gt_df["BR"].apply(parse_coords)
    pred_df["TL"] = pred_df["TL"].apply(parse_coords)
    pred_df["BR"] = pred_df["BR"].apply(parse_coords)

    def create_dict(df, is_pred=False):
        data = {}
        for _, row in df.iterrows():
            tile_id = row["tile_id"]
            bbox = (*row["TL"], *row["BR"])
            label = row["paly_type"]
            score = row.get("score", 1.0) if is_pred else None
            entry = {"bbox": bbox, "label": label}
            if score is not None:
                entry["score"] = score
            data.setdefault(tile_id, []).append(entry)
        return data

    ground_truth = create_dict(gt_df)
    predictions = create_dict(pred_df, is_pred=True)

    classwise_aps = []
    class_agnostic_aps = []
    all_labels = gt_df["paly_type"].unique()

    for class_label in all_labels:
        aps = []
        for tile_id, gt_boxes in ground_truth.items():
            gt_cls = [gt for gt in gt_boxes if gt["label"] == class_label]
            preds = [pred for pred in predictions.get(tile_id, []) if pred["label"] == class_label]
            preds = sorted(preds, key=lambda x: x["score"], reverse=True)

            tp = np.zeros(len(preds))
            fp = np.zeros(len(preds))
            matched = set()

            for i, pred in enumerate(preds):
                max_iou, max_j = 0, -1
                for j, gt in enumerate(gt_cls):
                    if j in matched:
                        continue
                    iou = calculate_iou(pred["bbox"], gt["bbox"])
                    if iou > max_iou:
                        max_iou = iou
                        max_j = j
                if max_iou >= iou_threshold:
                    tp[i] = 1
                    matched.add(max_j)
                else:
                    fp[i] = 1
            if gt_cls:
                aps.append(calculate_ap(tp, fp, len(gt_cls)))
        if aps:
            classwise_aps.append(np.mean(aps))

    for tile_id, gt_boxes in ground_truth.items():
        preds = sorted(predictions.get(tile_id, []), key=lambda x: x["score"], reverse=True)
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        matched = set()

        for i, pred in enumerate(preds):
            max_iou, max_j = 0, -1
            for j, gt in enumerate(gt_boxes):
                if j in matched:
                    continue
                iou = calculate_iou(pred["bbox"], gt["bbox"])
                if iou > max_iou:
                    max_iou = iou
                    max_j = j
            if max_iou >= iou_threshold:
                tp[i] = 1
                matched.add(max_j)
            else:
                fp[i] = 1
        if gt_boxes:
            class_agnostic_aps.append(calculate_ap(tp, fp, len(gt_boxes)))

    return {
        "class_specific_mAP": np.mean(classwise_aps) if classwise_aps else 0,
        "class_agnostic_mAP": np.mean(class_agnostic_aps) if class_agnostic_aps else 0
    }
