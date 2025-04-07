from src import config
import os
import xml.etree.ElementTree as ET

"""
This function retrieves the bounds of the annotation region in nanometers. 
Input: 
    - input_ndpi_file_path: string representing the absolute path to the input ndpi file.
Returns:
    - min_x_nano: the top left x coordinate of the annotated region in nanometers 
    - min_y_nano: the top left y coordiante of the annotated region in nanometers
    - max_x_nano: the bottom right x coordinate of the annotated region in nanometers
    - max_y_nano: the bottom right y coordinate of the annotated region in nanometers
"""
def annotation_region_bounds_retrieval(input_ndpi_file_path):
    file_no_extension = os.path.splitext(os.path.basename(input_ndpi_file_path))[0]
    tree = ET.parse(f"{os.path.join(config['abs_path_to_ndpa_dir'], file_no_extension)}.ndpi.ndpa")
    root = tree.getroot()
    for viewstate in root.findall("ndpviewstate"):
        annotation = viewstate.find("annotation")
        # this tells us that it is an annotated region
        if annotation is not None and annotation.get("displayname") == "AnnotateRectangle":
            pointlist = annotation.find("pointlist")
            if pointlist is not None:
                xs = []
                ys = []
                for point in pointlist.findall("point"):
                    xs.append(int(point.find('x').text))
                    ys.append(int(point.find('y').text))
                # basically, determine the coordinates of the top left and bottom right coordinates of the annotated region. 
                min_x_nano = min(xs)
                max_x_nano = max(xs)
                min_y_nano = min(ys)
                max_y_nano = max(ys)

                return min_x_nano, min_y_nano, max_x_nano, max_y_nano
                             