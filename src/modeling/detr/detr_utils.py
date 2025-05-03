import json
import csv
import random
from pathlib import Path
from typing import Dict
from collections import defaultdict
from PIL import Image
import logging

logger = logging.getLogger(__name__)

def pre_scan_images(image_root: Path) -> Dict[str, Path]:
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
                    images.append({
                        'id': img_id,
                        'file_name': rel,
                        'width': iw,
                        'height': ih
                    })
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
    coco = {
        'images': images,
        'annotations': annotations,
        'categories': [{'id': v, 'name': k} for k, v in categories.items()]
    }
    with output_json.open('w') as out:
        json.dump(coco, out, indent=2)
    logger.info(f"Saved COCO JSON to {output_json}")


def filter_category(
    input_json: Path,
    output_json: Path,
    exclude_label: str
) -> None:
    with input_json.open() as f:
        data = json.load(f)
    cat_ids = {c['id'] for c in data['categories'] if c['name'] == exclude_label}
    if not cat_ids:
        raise ValueError(f"Category '{exclude_label}' not found.")
    cid = cat_ids.pop()
    data['annotations'] = [a for a in data['annotations'] if a['category_id'] != cid]
    data['categories'] = [c for c in data['categories'] if c['id'] != cid]
    valid_imgs = {a['image_id'] for a in data['annotations']}
    data['images'] = [i for i in data['images'] if i['id'] in valid_imgs]
    with output_json.open('w') as out:
        json.dump(data, out, indent=2)
    logger.info(f"Filtered category '{exclude_label}' to {output_json}")


def split_by_tile_id(
    coco_json: Path,
    train_json: Path,
    val_json: Path,
    val_ratio: float = 0.2,
    seed: int = 42
) -> None:
    with coco_json.open() as f:
        data = json.load(f)
    groups = defaultdict(list)
    for img in data['images']:
        key = '/'.join(Path(img['file_name']).parts[:2])
        groups[key].append(img)
    tiles = list(groups)
    random.seed(seed)
    random.shuffle(tiles)
    split = int(len(tiles)*(1-val_ratio))
    train_tiles, val_tiles = tiles[:split], tiles[split:]
    train_imgs = [img for t in train_tiles for img in groups[t]]
    val_imgs   = [img for t in val_tiles   for img in groups[t]]
    train_ids = {i['id'] for i in train_imgs}
    val_ids   = {i['id'] for i in val_imgs}
    train_anns = [a for a in data['annotations'] if a['image_id'] in train_ids]
    val_anns   = [a for a in data['annotations'] if a['image_id'] in val_ids]
    for out_path, imgs, anns in [(train_json, train_imgs, train_anns), (val_json, val_imgs, val_anns)]:
        with out_path.open('w') as f:
            json.dump({'images': imgs,'annotations': anns,'categories': data['categories']},f,indent=2)
        logger.info(f"Wrote split JSON to {out_path}")