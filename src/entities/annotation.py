import os
import xml.etree.ElementTree as ET

class Annotation:
    def __init__(self, annot_id, label, annotation_tree_element):
        self.id = annot_id
        self.label = label
        self.center_x_nm = int(annotation_tree_element.find("x").text)
        self.center_y_nm = int(annotation_tree_element.find("y").text)
        self.radius_nm = int(annotation_tree_element.find("radius").text)