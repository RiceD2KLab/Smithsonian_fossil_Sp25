import json
import csv
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import xml.etree.ElementTree as ET

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


def pre_scan_images(image_root: Path) -> Dict[str, Path]:
    """
    Scan image_root recursively, mapping relative paths to Paths.
    """
    files: Dict[str, Path] = {}
    for path in image_root.rglob('*'):
        if path.is_file():
            rel = path.relative_to(image_root).as_posix()
            files[rel] = path
    return files


def convert_csv_to_coco(
    csv_path: Path,
    image_root: Path,
    output_json: Path
) -> None:
    """
    Convert annotation CSV into COCO JSON.
    """
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
            for z in range(25):
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
    input_json: Path,
    output_json: Path,
    exclude_label: str
) -> None:
    """
    Remove annotations and category named exclude_label.
    """
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


def collate_fn(
    batch: List[Tuple],
    processor: DetrImageProcessor
) -> Dict[str, torch.Tensor]:
    images, targets = zip(*batch)
    ann_list = []
    for i, t in enumerate(targets):
        seg = []
        for box, cid in zip(t['boxes'], t['labels']):
            x0, y0, x1, y1 = box.tolist()
            seg.append({'bbox': [x0, y0, x1-x0, y1-y0], 'category_id': int(cid), 'iscrowd': 0})
        ann_list.append({'image_id': i, 'annotations': seg})
    return processor(images=list(images), annotations=ann_list, return_tensors='pt')


def split_by_tile_id(
    coco_json: Path,
    train_json: Path,
    val_json: Path,
    val_ratio: float = 0.2,
    seed: int = 42
) -> None:
    """
    Split by tile_id grouping.
    """
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
    device: str = 'cpu'
) -> Tuple[torch.nn.Module, DetrImageProcessor]:
    processor = DetrImageProcessor.from_pretrained(model_name)
    model = DetrForObjectDetection.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
    if weights_path:
        st = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(st)
    model.to(device)
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
    log_every: int = 10
) -> None:
    model.train()
    step = 0
    for ep in range(epochs):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
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
    image_dir: Path,
    ann_json: Path,
    device: str = 'cpu',
    iou_type: str = 'bbox'
) -> COCOeval:
    coco_gt = COCO(str(ann_json))
    preds = []
    model.eval()
    for img_id in coco_gt.getImgIds():
        info = coco_gt.loadImgs(img_id)[0]
        path = image_dir / info['file_name']
        img = Image.open(path).convert('RGB')
        batch = processor(images=img, return_tensors='pt').to(device)
        with torch.no_grad(): out = model(**batch)
        res = processor.post_process_object_detection(out, threshold=0.3, target_sizes=[img.size[::-1]])[0]
        for score, label, box in zip(res['scores'], res['labels'], res['boxes']):
            x0,y0,x1,y1=box.tolist()
            preds.append({'image_id':img_id,'category_id':int(label.item()),'bbox':[x0,y0,x1-x0,y1-y0],'score':float(score)})
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


def create_annotation_element(
    annot_id: int,
    label: int,
    x_nm: int,
    y_nm: int
) -> ET.Element:
    elem=ET.Element('annotation')
    ET.SubElement(elem,'title').text=str(label)
    ET.SubElement(elem,'annotation_type').text='1'
    ET.SubElement(elem,'group').text=str(label)
    ET.SubElement(elem,'color').text='16711680'
    coords=ET.SubElement(elem,'coordinates')
    ET.SubElement(coords,'coordinate',{'order':'0','x':str(x_nm),'y':str(y_nm)})
    return elem


def predictions_to_ndpa(
    preds: List[dict],
    ndpi_base_dir: Path,
    output_dir: Path
) -> None:
    output_dir.mkdir(parents=True,exist_ok=True)
    grouped=defaultdict(list)
    for p in preds:
        grouped[get_slide_name(p['file_name'])].append(p)
    for slide,items in grouped.items():
        annotations=ET.Element('annotations')
        for i,pred in enumerate(items):
            tile=get_tile_id(pred['file_name'])
            x_off,y_off=parse_tile_id(tile)
            x0,y0,w,h=pred['bbox']
            cx,cy=int(x0+w/2+x_off),int(y0+h/2+y_off)
            annotations.append(create_annotation_element(i,pred['category_id'],cx,cy))
        out_path=output_dir/f"{slide}.ndpa"
        ET.ElementTree(annotations).write(out_path,encoding='utf-8',xml_declaration=True)
    logger.info(f"Finished writing NDPA files to {output_dir}")
