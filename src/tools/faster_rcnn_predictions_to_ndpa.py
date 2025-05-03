import os
import csv
import xml.etree.ElementTree as ET
from xml.dom import minidom
from src.tools.coordinate_space_convertor import pixelwise_to_nanozoomer

def parse_tile_id(tile_id):
    """
    Extracts (x, y) pixel offsets from tile_id string of format '48991x_36884y'.

    Returns:
        (x_offset, y_offset) as integers
    """
    try:
        x_part, y_part = tile_id.split('x_')
        x = int(x_part)
        y = int(y_part[:-1])  # strip trailing 'y'
        return x, y
    except Exception as e:
        raise ValueError(f"Invalid tile_id format: {tile_id}") from e


def convert_predictions_to_nanozoomer_for_faster_rcnn(pred_boxes, pred_labels, pred_scores, ndpi_path, tile_offset_px):
    """
    Converts model predictions in pixel space (relative to tile) to NanoZoomer coordinate space (in nanometers),
    by first adjusting for the tile's offset in the full NDPI image.

    Args:
        pred_boxes: Tensor of shape (N, 4) [xmin, ymin, xmax, ymax]
        pred_labels: Tensor of shape (N,)
        pred_scores: Tensor of shape (N,)
        ndpi_path:   Absolute path to the NDPI file (for metadata extraction)
        tile_offset_px: Tuple (x_offset, y_offset) for the tile's top-left corner in NDPI space

    Returns:
        List of dicts, each with:
            'label', 'score', 'xmin_nm', 'ymin_nm', 'xmax_nm', 'ymax_nm'
    """
    results = []
    x_offset, y_offset = tile_offset_px

    for i in range(pred_boxes.shape[0]):
        xmin_px, ymin_px, xmax_px, ymax_px = pred_boxes[i].tolist()

        # Adjust pixel coordinates by tile offset
        xmin_adj = xmin_px + x_offset
        ymin_adj = ymin_px + y_offset
        xmax_adj = xmax_px + x_offset
        ymax_adj = ymax_px + y_offset

        # Convert to nanometers using slide metadata
        xmin_nm, ymin_nm = pixelwise_to_nanozoomer(xmin_adj, ymin_adj, ndpi_path)
        xmax_nm, ymax_nm = pixelwise_to_nanozoomer(xmax_adj, ymax_adj, ndpi_path)

        results.append({
            "label": int(pred_labels[i].item()),
            "score": float(pred_scores[i].item()),
            "xmin_nm": int(xmin_nm),
            "ymin_nm": int(ymin_nm),
            "xmax_nm": int(xmax_nm),
            "ymax_nm": int(ymax_nm),
        })

    return results


def write_predictions_to_ndpa_for_faster_rcnn(predictions, output_path, class_mapping, lens_value=60.0, base_x=0, base_y=0):
    """
    Writes model predictions (in nanometers) to an NDPA-format XML file.

    Args:
        predictions: List of dicts, each with keys:
                     'label', 'score', 'xmin_nm', 'ymin_nm', 'xmax_nm', 'ymax_nm'
        output_path: Path to write the .ndpa XML file
        class_mapping: Dict mapping class names to IDs (e.g., {'pol': 1, ...})
        lens_value: NDPA lens zoom value (default: 60.0)
        base_x, base_y: Default x, y view position (optional)
    """

    # Reverse the class mapping from ID -> name
    reverse_mapping = {v: k for k, v in class_mapping.items()}

    annotations = ET.Element("annotations")

    for i, pred in enumerate(predictions):
        label_id = pred['label']
        label_name = reverse_mapping.get(label_id, f"class_{label_id}")

        view = ET.SubElement(annotations, "ndpviewstate", id=str(i + 1))

        # View metadata
        ET.SubElement(view, "title").text = f"prediction_{i+1}"
        ET.SubElement(view, "details").text = f"{label_name}, score={pred['score']:.3f}"
        ET.SubElement(view, "coordformat").text = "nanometers"
        ET.SubElement(view, "lens").text = str(lens_value)
        ET.SubElement(view, "x").text = str(base_x)
        ET.SubElement(view, "y").text = str(base_y)
        ET.SubElement(view, "z").text = "0"
        ET.SubElement(view, "showtitle").text = "0"
        ET.SubElement(view, "showhistogram").text = "0"
        ET.SubElement(view, "showlineprofile").text = "0"

        # Annotation rectangle
        annotation = ET.SubElement(view, "annotation", {
            "type": "freehand",
            "displayname": "AnnotateRectangle",
            "color": "#000000"
        })
        ET.SubElement(annotation, "measuretype").text = "2"
        ET.SubElement(annotation, "closed").text = "1"

        pointlist = ET.SubElement(annotation, "pointlist")
        points = [
            (pred["xmin_nm"], pred["ymin_nm"]),
            (pred["xmin_nm"], pred["ymax_nm"]),
            (pred["xmax_nm"], pred["ymax_nm"]),
            (pred["xmax_nm"], pred["ymin_nm"])
        ]
        for x, y in points:
            point = ET.SubElement(pointlist, "point")
            ET.SubElement(point, "x").text = str(int(x))
            ET.SubElement(point, "y").text = str(int(y))

        ET.SubElement(annotation, "specialtype").text = "rectangle"

    # Pretty-print and write XML
    rough_string = ET.tostring(annotations, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    with open(output_path, "w") as f:
        f.write(reparsed.toprettyxml(indent="  "))


def write_predictions_to_csv(predictions, output_csv_path, class_mapping, filename):
    """
    Converts raw predictions to master-style CSV format and saves to file.

    Args:
        predictions (list of dicts): Each prediction has keys:
            'label', 'score', 'xmin_nm', 'ymin_nm', 'xmax_nm', 'ymax_nm', 'tile_id'
        output_csv_path (str): Path to save the CSV.
        class_mapping (dict): Maps class indices to class names (e.g., 3 → 'pol').
        filename (str): Original NDPI filename.
    """
    
    index_to_class = {v: k for k, v in class_mapping.items()}
    
    with open(output_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "annot_id", "paly_type", "center", "radius",
            "TL", "BL", "TR", "BR", "tile_id"
        ])

        for idx, pred in enumerate(predictions, start=1):
            label = index_to_class.get(pred["label"], "unknown")
            score = float(pred["score"])
            
            xmin, ymin, xmax, ymax = pred["xmin_px"], pred["ymin_px"], pred["xmax_px"], pred["ymax_px"]

            center_x = (xmin + xmax) // 2
            center_y = (ymin + ymax) // 2
            radius = max((xmax - xmin), (ymax - ymin)) // 2

            # Convert coordinates to (x, y) string format
            center = f"({center_x}, {center_y})"
            TL = f"({xmin}, {ymin})"
            BL = f"({xmin}, {ymax})"
            TR = f"({xmax}, {ymin})"
            BR = f"({xmax}, {ymax})"

            writer.writerow([
                filename, idx, label, center, radius,
                TL, BL, TR, BR, pred.get("tile_id", "unknown")
            ])