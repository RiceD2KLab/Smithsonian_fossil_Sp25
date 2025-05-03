from src.tools.ndpi_metadata_extractor import extract_ndpi_metadata
import xml.etree.ElementTree as ET
from xml.dom import minidom

"""
This function applies a translation to an inputted point
Inputs:
    NOTE: the unit of measure of horiz. and vert. shift must be consistent with the units of the x and y coordinates!
    - x: the x coordinate of a point to be transformed
    - y: the y coordinate of a point to be transformed
    - horizontal_shift: the number of units to shift in the x direction
    - vertical_shift: the number of units to shift in the y direction
Returns:
    - a two element tuple where:
        - the first element is the translated coordinate's x value
        - the second element is the translated coordinate's y value
"""
def translation(x, y, horizontal_shift, vertical_shift):
    new_x = x + horizontal_shift
    new_y = y + vertical_shift
    return (new_x, new_y)

"""
This function determines the top left coordinate of the inputted ndpi file in the nanozoomer coordinate space
Inputs:
    - input_ndpi_file_path: a string representing the absolute path to the ndpi file

Outputs:
    - a two element tuple where 
        - the first element is an integer representing the x coordinate of the top-left corner of the ndpi file in nm
        - the second element is an integer representing the y coordinate of the top-left corner of the ndpi file in nm
"""
def find_ndpi_tl_coordinate(input_ndpi_file_path):
    ndpi_metadata = extract_ndpi_metadata(input_ndpi_file_path)

    # a tuple representing the center point of the ndpi file in the nanozoomer coordinate system (in nanometers)
    ndpi_center_nm = (int(ndpi_metadata["hamamatsu.XOffsetFromSlideCentre"]), int(ndpi_metadata["hamamatsu.YOffsetFromSlideCentre"]))

    # the number of nanometers per pixel in the x and y direction respectively
    nmpp_x = float(ndpi_metadata["openslide.mpp-x"]) * 1000 # mutiply by 1000 to convert from millimeters to nanometers 
    nmpp_y = float(ndpi_metadata["openslide.mpp-y"]) * 1000 # mutiply by 1000 to convert into millimeters to nanometers 

    # the width and height, respectively, of the ndpi file in pixels
    ndpi_width_px = int(ndpi_metadata["openslide.level[0].width"])
    ndpi_height_px = int(ndpi_metadata["openslide.level[0].height"])

    # convert ndpi height and width from px to nm
    ndpi_width_nm = nmpp_x * ndpi_width_px 
    ndpi_height_nm = nmpp_y * ndpi_height_px 

    # top left coordinate x and y value, respectively, in nanometers
    top_left_x_nm = ndpi_center_nm[0] - (ndpi_width_nm // 2)
    top_left_y_nm = ndpi_center_nm[1] - (ndpi_height_nm // 2) # keep in mind: this is SUBTRACTION because the upwards direction is (-)!

    return (top_left_x_nm, top_left_y_nm)

"""
This function converts nanozoomer coordinate of a specific inputted ndpi file to pixelwise coordinate system. 

Inputs:
    - x_nm: integer representing the x coordinate in the nanozoomer coordinate system (in nanometers)
    - y_nm: integer representing the y coordinate in the nanozoomer coordinate system (in nanometers)
    - input_ndpi_file_path: string representing the absolute path of the ndpi file where the inputted nanozoomer coordinate exists.

Outputs:
    - a two element tuple where:
        - the first element is an integer representing the x coordinate of the transformed point in pixels
        - the second element is an integer representign the y coordinate of the transformed point in pixels
"""
def nanozoomer_to_pixelwise(x_nm, y_nm, input_ndpi_file_path):
    ndpi_tl_coordinate_nm = find_ndpi_tl_coordinate(input_ndpi_file_path)

    # perform translation to shift the ndpi image such that the top left corner gets translated to the origin 
    shifted_coordinate_nm = translation(x_nm, y_nm, (-1) * ndpi_tl_coordinate_nm[0], (-1) * ndpi_tl_coordinate_nm[1])

    ndpi_metadata = extract_ndpi_metadata(input_ndpi_file_path)
    # the number of nanometers per pixel in the x and y direction respectively
    nmpp_x = float(ndpi_metadata["openslide.mpp-x"]) * 1000 # mutiply by 1000 to convert from millimeters to nanometers 
    nmpp_y = float(ndpi_metadata["openslide.mpp-y"]) * 1000 # mutiply by 1000 to convert into millimeters to nanometers 

    # convert the shifted coordinate from nm to px using the ratio retrieved above
    shifted_coordinate_px = (shifted_coordinate_nm[0] // nmpp_x, shifted_coordinate_nm[1] // nmpp_y)
    return shifted_coordinate_px

def pixelwise_to_nanozoomer(x_px, y_px, input_ndpi_file_path):
    """
    Converts pixel coordinates of a specific inputted NDPI file to nanozoomer global coordinate system.

    Inputs:
        - x_px: integer representing the x coordinate in the pixel coordinate system
        - y_px: integer representing the y coordinate in the pixel coordinate system
        - input_ndpi_file_path: string representing the absolute path of the NDPI file

    Outputs:
        - a two element tuple:
            - first element is the x coordinate in nanometers in the nanozoomer global space
            - second element is the y coordinate in nanometers in the nanozoomer global space
    """
    ndpi_metadata = extract_ndpi_metadata(input_ndpi_file_path)

    # Get nm per pixel
    nmpp_x = float(ndpi_metadata["openslide.mpp-x"]) * 1000  # mm to nm
    nmpp_y = float(ndpi_metadata["openslide.mpp-y"]) * 1000  # mm to nm

    # Convert from px to nm
    x_nm_relative = x_px * nmpp_x
    y_nm_relative = y_px * nmpp_y

    # Get the top-left corner of the NDPI file in nanometer coordinates
    ndpi_tl_coordinate_nm = find_ndpi_tl_coordinate(input_ndpi_file_path)

    # Translate back to the NanoZoomer global coordinate system
    x_nm = ndpi_tl_coordinate_nm[0] + x_nm_relative
    y_nm = ndpi_tl_coordinate_nm[1] + y_nm_relative

    return (int(x_nm), int(y_nm))

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

    # ✅ Pretty-print XML output
    rough_string = ET.tostring(annotations, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    with open(output_path, "w") as f:
        f.write(reparsed.toprettyxml(indent="  "))

    print(f"✅ NDPA file written to: {output_path}")