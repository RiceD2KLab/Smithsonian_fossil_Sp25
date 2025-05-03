import os
import xml.etree.ElementTree as ET
from src.tools.coordinate_space_convertor import pixelwise_to_nanozoomer

def convert_predictions_to_nanozoomer_for_faster_rcnn(pred_boxes, pred_labels, pred_scores, ndpi_path):
    """
    Converts model predictions in pixel space to NanoZoomer coordinate space (nanometers).
    
    Inputs:
        - pred_boxes: Tensor of shape (N, 4) [xmin, ymin, xmax, ymax]
        - pred_labels: Tensor of shape (N,)
        - pred_scores: Tensor of shape (N,)
        - ndpi_path: Absolute path to the NDPI file (for metadata extraction)

    Output:
        - List of dicts: Each dict contains keys:
            'label', 'score', 'xmin_nm', 'ymin_nm', 'xmax_nm', 'ymax_nm'
    """
    results = []
    for i in range(pred_boxes.shape[0]):
        xmin_px, ymin_px, xmax_px, ymax_px = pred_boxes[i].tolist()

        # Convert top-left and bottom-right from pixel to nm
        xmin_nm, ymin_nm = pixelwise_to_nanozoomer(xmin_px, ymin_px, ndpi_path)
        xmax_nm, ymax_nm = pixelwise_to_nanozoomer(xmax_px, ymax_px, ndpi_path)

        result = {
            "label": int(pred_labels[i].item()),
            "score": float(pred_scores[i].item()),
            "xmin_nm": int(xmin_nm),
            "ymin_nm": int(ymin_nm),
            "xmax_nm": int(xmax_nm),
            "ymax_nm": int(ymax_nm),
        }
        results.append(result)
    return results


def write_predictions_to_ndpa_for_faster_rcnn(predictions, output_path, lens_value=60.0, base_x=0, base_y=0):
    """
    Writes model predictions (in nanometers) to an NDPA-format XML file.

    Args:
        predictions: List of dicts, each with keys:
                     'label', 'score', 'xmin_nm', 'ymin_nm', 'xmax_nm', 'ymax_nm'
        output_path: Path to write the .ndpa XML file
        lens_value:  NDPA lens zoom value (default: 60.0)
        base_x, base_y: Default x, y view position (optional)
    """

    annotations = ET.Element("annotations")

    for i, pred in enumerate(predictions):
        view = ET.SubElement(annotations, "ndpviewstate", id=str(i + 1))

        # View metadata
        ET.SubElement(view, "title").text = f"prediction_{i+1}"
        ET.SubElement(view, "details").text = f"label_{pred['label']}, score={pred['score']:.3f}"
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
        # Clockwise rectangle points
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

    # Write XML to file
    tree = ET.ElementTree(annotations)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    print(f"✅ NDPA file written to: {output_path}")