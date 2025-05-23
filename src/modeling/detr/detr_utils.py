import json
import csv
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
from src.tools.coordinate_space_convertor import pixelwise_to_nanozoomer
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from torchvision.ops import nms, box_iou
from PIL import Image
from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor,
    get_cosine_schedule_with_warmup
)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

logger = logging.getLogger(__name__)

def pre_scan_images(image_root: str) -> Dict[str, Path]:
    """
    Scan image_root recursively, mapping relative paths to Paths.
    """
    image_root_path = Path(image_root)  
    files: Dict[str, Path] = {}
    for path in image_root_path.rglob('*'):
        if path.is_file():
            rel = path.relative_to(image_root_path).as_posix()
            files[rel] = path
    return files


def convert_csv_to_coco(
    csv_path: Path,
    image_root: Path,
    output_json: Path,
    focal_length=25
) -> None:
    """
    Convert annotation CSV into COCO JSON.
    """
    csv_path = Path(csv_path)
    image_root = Path(image_root)
    output_json = Path(output_json)
    available = pre_scan_images(image_root)
    images, annotations = [], []
    categories: Dict[str, int] = {}
    image_id_map: Dict[str, int] = {}
    ann_id = 1
    with csv_path.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                tl = tuple(map(float, row['TL'].strip('()').split(',')))
                br = tuple(map(float, row['BR'].strip('()').split(',')))
            except Exception as e:
                logger.warning(f"Skipping row with invalid bbox: {e}")
                continue
            x, y = tl
            w, h = br[0] - x, br[1] - y
            label = row['paly_type']
            cat_id = categories.setdefault(label, len(categories))
            base = row['filename'].replace('.ndpi.ndpa', '')
            tile = row['tile_id']
            for z in range(focal_length):
                rel = f"{base}/{tile}/{z}z.png"
                path = available.get(rel)
                if not path:
                    continue
                img_id = image_id_map.get(rel)
                if img_id is None:
                    with Image.open(path) as img:
                        iw, ih = img.size
                    img_id = len(images) + 1
                    images.append({'id': img_id, 'file_name': rel, 'width': iw, 'height': ih})
                    image_id_map[rel] = img_id
                annotations.append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': cat_id,
                    'bbox': [x, y, w, h],
                    'area': w * h,
                    'iscrowd': 0
                })
                ann_id += 1
    coco_dict = {
        'images': images,
        'annotations': annotations,
        'categories': [{'id': v, 'name': k} for k, v in categories.items()]
    }
    with output_json.open('w') as out:
        json.dump(coco_dict, out, indent=2)
    logger.info(f"Saved COCO JSON to {output_json}")


def filter_category(
    input_json: str,
    output_json: str,
    exclude_label: str
) -> None:
    """
    Remove annotations and category named exclude_label.
    """
    input_json=Path(input_json)
    output_json=Path(output_json)
    with input_json.open() as f:
        data = json.load(f)
    cat_ids = {c['id'] for c in data['categories'] if c['name'] == exclude_label}
    if not cat_ids:
        raise ValueError(f"Category '{exclude_label}' not found.")
    cid = cat_ids.pop()
    data['annotations'] = [a for a in data['annotations'] if a['category_id'] != cid]
    data['categories'] = [c for c in data['categories'] if c['id'] != cid]
    valid_ids = {a['image_id'] for a in data['annotations']}
    data['images'] = [i for i in data['images'] if i['id'] in valid_ids]
    with output_json.open('w') as out:
        json.dump(data, out, indent=2)
    logger.info(f"Filtered category '{exclude_label}' to {output_json}")


class CocoDetectionTransform(CocoDetection):
    """
    Return (PIL.Image, {'boxes': Tensor[N,4], 'labels': Tensor[N]}).
    """
    def __getitem__(self, idx: int):
        img, _ = super().__getitem__(idx)
        img_id = self.ids[idx]
        anns = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        return img, {'boxes': torch.tensor(boxes), 'labels': torch.tensor(labels)}


def collate_fn(batch, processor):
    images, targets = zip(*batch)
    hf_anns = []

    for i, tgt in enumerate(targets):
        coco_boxes = []
        areas      = []
        anns_list  = []

        for x0, y0, x1, y1 in tgt["boxes"].tolist():
            w = x1 - x0
            h = y1 - y0
            coco_boxes.append([x0, y0, w, h])
            areas.append(w * h)

        ann_list = []
        for bbox, cid, area in zip(coco_boxes,
                                   tgt["labels"].tolist(),
                                   areas):
            anns_list.append({
                "bbox":        bbox,
                "category_id": int(cid),
                "area":        float(area),
                "iscrowd":     0
            })

        hf_anns.append({
            "image_id":   i,         # just the batch‐local index
            "annotations": anns_list
        })

    # Now call the processor once, with both images and annotations
    encoding = processor(
        images=list(images),
        annotations=hf_anns,
        return_tensors="pt"
    )
    return encoding


def split_by_tile_id(
    coco_json: str,
    train_json: str,
    val_json: str,
    val_ratio: float = 0.2,
    seed: int = 42
) -> None:
    """
    Split by tile_id grouping.
    """
    coco_json=Path(coco_json)
    train_json = Path(train_json)
    val_json=Path(val_json)
    with coco_json.open() as f:
        data = json.load(f)
    groups = defaultdict(list)
    for img in data['images']:
        key = '/'.join(Path(img['file_name']).parts[:2])
        groups[key].append(img)
    tiles = list(groups)
    random.seed(seed)
    random.shuffle(tiles)
    split_idx = int(len(tiles) * (1 - val_ratio))
    train_tiles, val_tiles = tiles[:split_idx], tiles[split_idx:]
    train_imgs = [img for t in train_tiles for img in groups[t]]
    val_imgs   = [img for t in val_tiles   for img in groups[t]]
    train_ids = {i['id'] for i in train_imgs}
    val_ids   = {i['id'] for i in val_imgs}
    train_anns = [a for a in data['annotations'] if a['image_id'] in train_ids]
    val_anns   = [a for a in data['annotations'] if a['image_id'] in val_ids]
    for out_path, imgs, anns in [(train_json, train_imgs, train_anns), (val_json, val_imgs, val_anns)]:
        with out_path.open('w') as f:
            json.dump({'images': imgs, 'annotations': anns, 'categories': data['categories']}, f, indent=2)
        logger.info(f"Wrote split JSON to {out_path}")


def initialize_model(
    model_name: str,
    num_labels: int,
    weights_path: Optional[Path],
    device: str = 'cpu',
    coco_json_path = None
) -> Tuple[torch.nn.Module, DetrImageProcessor]:
    processor = DetrImageProcessor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
    if weights_path:
        st = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(st)
    model.to(device)

    
    try:
        with open(coco_json_path, "r") as f:
            data = json.load(f)
        id2label = {cat["id"]: cat["name"] for cat in data["categories"]}
        label2id = {v: k for k, v in id2label.items()}

        model.config.id2label = id2label
        model.config.label2id = label2id
    except:
        print("No existing coco json file, must be using default model")
        id2label = {0: 'pol', 1: 'fun', 2: 'spo', 4: 'alg'}
        label2id = {'pol': 0, 'fun': 1, 'spo': 2, 'alg': 4}
        model.config.id2label = id2label
        model.config.label2id = label2id
    

    return model.eval(), processor


def get_optimizer_scheduler(
    model: torch.nn.Module,
    train_loader: DataLoader,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    epochs: int = 20,
    warmup_frac: float = 0.1
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    warmup_steps = int(warmup_frac * total_steps)
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
    return opt, sched


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: str = 'cpu',
    epochs: int = 20,
    clip_norm: float = 0.1,
    log_every: int = 10,
) -> None:
    model.train()
    step = 0
    for ep in range(epochs):
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"]
            for lbl in labels:
                lbl["boxes"]        = lbl["boxes"].to(device)
                lbl["class_labels"] = lbl["class_labels"].to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            # batch = {k: v.to(device) for k, v in batch.items()}
            # out = model(**batch)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step(); scheduler.step()
            step += 1
            if step % log_every == 0:
                logger.info(f"Epoch {ep+1}/{epochs} Step {step} Loss {loss.item():.4f}")


def evaluate_coco(
    model: torch.nn.Module,
    processor: DetrImageProcessor,
    image_dir: str,
    ann_json: str,
    device: str = 'cpu',
    iou_type: str = 'bbox'
) -> COCOeval:
    image_dir=Path(image_dir)
    ann_json=Path(ann_json)
    coco_gt = COCO(str(ann_json))
    preds = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    for img_id in coco_gt.getImgIds():
        info = coco_gt.loadImgs(img_id)[0]
        path = image_dir / info['file_name']
        img = Image.open(path).convert('RGB')
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        # run inference
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_object_detection(
            outputs, threshold=0.5, target_sizes=[img.size[::-1]]
        )[0]

        # collect each box
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x0, y0, x1, y1 = box.tolist()
            w, h = x1 - x0, y1 - y0
            preds.append({
                "image_id":    img_id,
                "category_id": int(label.item()),
                "bbox":        [x0, y0, w, h],
                "score":       float(score)
            })
    coco_dt = coco_gt.loadRes(preds)
    coco_eval = COCOeval(coco_gt,coco_dt, iouType=iou_type)
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    return coco_eval


def compute_f1_curve(
    preds_by_image: Dict[int, List[dict]],
    gt_by_image: Dict[int, dict],
    thresholds: Optional[List[float]] = None
) -> List[Tuple[float, float, float, float]]:
    if thresholds is None:
        thresholds = [round(t,2) for t in torch.arange(0,1.01,0.05).tolist()]
    results = []
    for thr in thresholds:
        tp=fp=fn=0
        for img_id, gt in gt_by_image.items():
            preds=[p for p in preds_by_image.get(img_id,[]) if p['score']>=thr]
            if preds:
                boxes=torch.tensor([p['bbox'] for p in preds])
                boxes_xyxy=boxes.clone(); boxes_xyxy[:,2:] += boxes_xyxy[:,:2]
                labels=torch.tensor([p['category_id'] for p in preds])
                ious=box_iou(boxes_xyxy.to(gt['boxes'].device), gt['boxes'])
                matched=set()
                for i in range(len(boxes_xyxy)):
                    m,j=ious[i].max(0)
                    if m>=0.5 and labels[i]==gt['labels'][j] and j.item() not in matched:
                        tp+=1; matched.add(j.item())
                fp+=len(boxes_xyxy)-len(matched)
                fn+=len(gt['boxes'])-len(matched)
            else:
                fn+=len(gt['boxes'])
        prec=tp/(tp+fp+1e-6)
        rec=tp/(tp+fn+1e-6)
        f1=2*prec*rec/(prec+rec+1e-6)
        results.append((thr,prec,rec,f1))
    return results


def parse_tile_id(tile_id: str) -> Tuple[int,int]:
    x_str,y_str=tile_id.split('_')
    return int(x_str.replace('x','')), int(y_str.replace('y',''))


def get_slide_name(file_path: str) -> str:
    return Path(file_path).parts[0]


def get_tile_id(file_path: str) -> str:
    return Path(file_path).parts[1]


def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, 'utf-8')
    parsed = minidom.parseString(rough_string)
    return parsed.toprettyxml(indent="  ")

def create_ndpviewstate(pred_id, label, score, rect_points):
    """
    Create <ndpviewstate> element for a prediction.
    
    Args:
        pred_id: Integer ID
        label: Prediction label (you can embed this in <details> or <title>)
        score: Confidence score
        rect_points: List of 4 (x, y) tuples for rectangle corners in order
    """
    state = ET.Element("ndpviewstate", id=str(pred_id))
    ET.SubElement(state, "title").text = f"prediction_{pred_id}"
    ET.SubElement(state, "details").text = f"{label}, score={score:.3f}"
    ET.SubElement(state, "coordformat").text = "nanometers"
    ET.SubElement(state, "lens").text = "60.0"
    ET.SubElement(state, "x").text = "0"
    ET.SubElement(state, "y").text = "0"
    ET.SubElement(state, "z").text = "0"
    ET.SubElement(state, "showtitle").text = "0"
    ET.SubElement(state, "showhistogram").text = "0"
    ET.SubElement(state, "showlineprofile").text = "0"

    annotation = ET.SubElement(state, "annotation", {
        "type": "freehand",
        "displayname": "AnnotateRectangle",
        "color": "#000000"
    })
    ET.SubElement(annotation, "measuretype").text = "2"
    ET.SubElement(annotation, "closed").text = "1"
    pointlist = ET.SubElement(annotation, "pointlist")
    for x, y in rect_points:
        pt = ET.SubElement(pointlist, "point")
        ET.SubElement(pt, "x").text = str(x)
        ET.SubElement(pt, "y").text = str(y)
    ET.SubElement(annotation, "specialtype").text = "rectangle"

    return state

def predictions_to_ndpa(preds, ndpi_base_dir, output_dir, id2label):
    """
    Generate formatted .ndpa annotation files from predictions.

    Args:
        preds: List from apply_tile_level_nms()
        ndpi_base_dir: Where .ndpi files are stored
        output_dir: Where to save .ndpa files
        pixelwise_to_nanozoomer: Coordinate conversion function
    """
    grouped_by_slide = defaultdict(list)
    for pred in preds:
        slide_name = get_slide_name(pred['file_name'])
        tile_id = get_tile_id(pred['file_name'])
        grouped_by_slide[slide_name].append((tile_id, pred))

    os.makedirs(output_dir, exist_ok=True)

    for slide_name, items in grouped_by_slide.items():
        ndpi_path = os.path.join(ndpi_base_dir, f"{slide_name}.ndpi")
        annotations_root = ET.Element("annotations")

        for idx, (tile_id, pred) in enumerate(items, 1):
            x_off, y_off = parse_tile_id(tile_id)
            box = pred['boxes']
            label = pred['labels']
            label = id2label[label]
            score = pred['scores']

            # Rectangle corners
            x_min = int(box[0] + x_off)
            y_min = int(box[1] + y_off)
            x_max = int(box[2] + x_off)
            y_max = int(box[3] + y_off)

            corners = [
                pixelwise_to_nanozoomer(x_min, y_min, ndpi_path),
                pixelwise_to_nanozoomer(x_min, y_max, ndpi_path),
                pixelwise_to_nanozoomer(x_max, y_max, ndpi_path),
                pixelwise_to_nanozoomer(x_max, y_min, ndpi_path),
            ]

            ndpviewstate = create_ndpviewstate(idx, label, score, corners)
            annotations_root.append(ndpviewstate)

        output_path = os.path.join(output_dir, f"{slide_name}.ndpi.ndpa")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(prettify(annotations_root))

    print(f"Pretty NDPA files saved to {output_dir}")

    
def predict_image(model, processor, image_path: str, device: str, threshold: float = 0.5):
    """
    Run inference on a single image and return raw boxes, labels, scores.

    Returns:
        dict with 'boxes', 'labels', 'scores'
    """
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    post = processor.post_process_object_detection(
        outputs, threshold=threshold, target_sizes=[image.size[::-1]]
    )[0]
    return {
        'boxes': post['boxes'].cpu().tolist(),
        'labels': post['labels'].cpu().tolist(),
        'scores': post['scores'].cpu().tolist()
    }

def batch_predict(model, processor, image_dir: str, device: str, threshold: float = 0.5):
    """
    Recursively iterate over nested image directory structure and collect predictions.

    Returns:
        list of predictions per image
    """
    preds = []
    for root, _, files in os.walk(image_dir):
        for fname in files:
            if not fname.lower().endswith(('.png', '.jpg')):
                continue
            img_path = os.path.join(root, fname)
            result = predict_image(model, processor, img_path, device, threshold)
            # Save relative path from image_dir for traceability
            rel_path = os.path.relpath(img_path, image_dir)
            preds.append({
                'file_name': rel_path,
                'boxes': result['boxes'],
                'labels': result['labels'],
                'scores': result['scores']
            })
    return preds

def apply_tile_level_nms(predictions, iou_threshold=0.5):
    """
    Apply NMS across focal planes for each tile_id in the predictions list.

    Args:
        predictions: List of dicts with keys: 'file_name', 'boxes', 'labels', 'scores'
        iou_threshold: IOU threshold for NMS

    Returns:
        List of filtered predictions (after NMS per tile_id)
    """
    tile_groups = defaultdict(list)

    # Group predictions by tile_id
    for pred in predictions:
        tile_id = get_tile_id(pred['file_name'])
        if tile_id is None:
            continue
        for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
            tile_groups[tile_id].append({
                'file_name': pred['file_name'],
                'box': torch.tensor(box),
                'label': label,
                'score': score
            })

    # Apply NMS to each tile group
    filtered_preds = []
    for tile_id, preds in tile_groups.items():
        if not preds:
            continue
        boxes = torch.stack([p['box'] for p in preds])
        scores = torch.tensor([p['score'] for p in preds])
        labels = [p['label'] for p in preds]
        file_names = [p['file_name'] for p in preds]

        keep_idxs = nms(boxes, scores, iou_threshold)

        for idx in keep_idxs:
            filtered_preds.append({
                'file_name': file_names[idx],
                'boxes': boxes[idx].tolist(),
                'labels': labels[idx],
                'scores': scores[idx].item()
            })

    return filtered_preds