import torch
import os
from transformers import DetrForObjectDetection
from transformers import DetrImageProcessor
from PIL import Image
from collections import defaultdict
from torchvision.ops import nms
import xml.etree.ElementTree as ET
from src.tools import pixelwise_to_nanozoomer

def initialize_model(model_name: str, num_labels: int, weights_path: str = None, device: str = "cpu"):
    """
    Load a pretrained DETR model and its processor. Optionally load fine-tuned weights.

    Returns:
        model: the DETR model on the specified device
        processor: associated image processor
    """
    model = DetrForObjectDetection.from_pretrained(
        model_name,
        num_labels=num_labels,
        ignore_mismatched_sizes=True
    )
    if weights_path:
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    processor = DetrImageProcessor.from_pretrained(model_name)
    return model, processor

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


def parse_tile_id(tile_id):
    # Extracts integer x/y offsets from "52492x_64767y"
    x_str, y_str = tile_id.split("_")
    x_offset = int(x_str.replace("x", ""))
    y_offset = int(y_str.replace("y", ""))
    return x_offset, y_offset

def get_slide_name(file_path):
    # slide_name/tile_id/focal.png => "slide_name"
    return file_path.split(os.sep)[0]

def get_tile_id(file_path):
    # slide_name/tile_id/focal.png => "tile_id"
    return file_path.split(os.sep)[1]

def create_annotation_element(annot_id, label, x_nm, y_nm):
    element = ET.Element("annotation")
    ET.SubElement(element, "title").text = str(label)
    ET.SubElement(element, "annotation_type").text = "1"  # dot annotation
    ET.SubElement(element, "group").text = str(label)
    ET.SubElement(element, "color").text = "16711680"  # red
    coordinates = ET.SubElement(element, "coordinates")
    ET.SubElement(coordinates, "coordinate", {
        "order": "0",
        "x": str(x_nm),
        "y": str(y_nm)
    })
    return element

def predictions_to_ndpa(preds, ndpi_base_dir, output_dir):
    """
    Generate one .ndpa annotation file per slide_name from predicted objects.

    Args:
        preds: Output list from apply_tile_level_nms()
        ndpi_base_dir: Path containing .ndpi files named like slide_name.ndpi
        output_dir: Where .ndpa XML files should be saved
        pixelwise_to_nanozoomer: Function to convert pixel coordinates to NanoZoomer coordinates
    """
    grouped_by_slide = defaultdict(list)

    # Group by slide_name
    for pred in preds:
        slide_name = get_slide_name(pred['file_name'])
        tile_id = get_tile_id(pred['file_name'])
        grouped_by_slide[slide_name].append((tile_id, pred))

    os.makedirs(output_dir, exist_ok=True)

    for slide_name, items in grouped_by_slide.items():
        ndpi_path = os.path.join(ndpi_base_dir, f"{slide_name}.ndpi")
        annotations = ET.Element("annotations")
        annot_id = 0

        for tile_id, pred in items:
            x_off, y_off = parse_tile_id(tile_id)
            box = pred['boxes']
            label = pred['labels']

            # Center of the box
            x_center = int((box[0] + box[2]) / 2 + x_off)
            y_center = int((box[1] + box[3]) / 2 + y_off)

            # Convert to NanoZoomer coordinate system
            x_nm, y_nm = pixelwise_to_nanozoomer(x_center, y_center, ndpi_path)

            # Create annotation element
            annot_elem = create_annotation_element(annot_id, label, x_nm, y_nm)
            annotations.append(annot_elem)
            annot_id += 1

        # Write .ndpa file
        output_path = os.path.join(output_dir, f"{slide_name}.ndpa")
        tree = ET.ElementTree(annotations)
        tree.write(output_path, encoding='utf-8', xml_declaration=True)

    print(f"Finished writing NDPA files to: {output_dir}")