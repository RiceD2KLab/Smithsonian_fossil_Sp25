import os
import xml.etree.ElementTree as ET
from src.entities.annotation import Annotation


class Ndpa:
    """
    This class represents a single ndpa file

    Attributes
    ----------
    - absolute_file_path : str
        the absolute path to the ndpa file
    - filename : str
        the file name of the ndpa file (not the path, just the file name)
    - annotated_region : map<string, tuple>
        following instantiation of an Ndpa object, the annotated_region map will have two key value pairs <str, tuple<str>>:
        - 'tl':(x, y) - where 'tl' represents the top left corner of the annotated region rectangle. The tuple is the (x,y) coordinate of the corner in nm
        - 'br':(x, y) - where 'br' represents the bottom right corner fo the annotated region rectangle. The tuple is the (x,y) coordinate of the corner in nm
    - annotations : map<integer, Annotation object>
        a python dictionary holding all the annotations to the ndpa file. 
            - keys: are integers representing the annotation id within the ndpa file
            - values: Annotation object is an object of the Annotation class that stores information about each annotation
    
    Methods
    ----------
    - populate_annotated_region: this method parses the ndpa file to extract the annotated region bounds and populate self.annotated_region
    - populate_annotations: this method parses the ndpa file to extract all the annotations, creating Annotation objects and populate self.annotations

    """
    def __init__(self, ndpa_file_path):
        """
        constructor with single parameter: ndpa_file_path - a string representing the absolute file path to the ndpa file
        """
        # open the ndpa and extract everything 
        self.absolute_file_path = ndpa_file_path
        self.filename = os.path.basename(ndpa_file_path)
        self.annotated_region = {}
        self.annotations = {}
        # populate annotated_region
        self.populate_annotated_region()
        # populate annotations
        self.populate_annotations()

    def populate_annotated_region(self):
        """
        This function populates the annotated region attribute of the Ndpa object. 
        Inputs:
            - None
        Returns:
            - None
            - Action: adds key-value pairs to self.annotated_region attribute. In particular, it adds 'tl':(x, y) and 'br':(x, y)
                    - keys: 'tl' represents the top left corner of the annotated region, 'br' represents the bottom right of the annotated region
                    - values: two element tuples where the first and second element are the x and y values (in nanometers) of the coordinate
        """
        tree = ET.parse(self.absolute_file_path)
        root = tree.getroot()
        for viewstate in root.findall("ndpviewstate"):
            annotation = viewstate.find("annotation")
            if annotation is not None and annotation.get("displayname") == "AnnotateRectangle":
                pointlist = annotation.find("pointlist")
                if pointlist is not None:
                    xs = []
                    ys = []
                    for point in pointlist.findall("point"):
                        xs.append(int(point.find('x').text))
                        ys.append(int(point.find('y').text))
                    # basically, determine the coordinates of the top left and bottom right coordinates of the annotated region. 
                    min_x_nm = min(xs)
                    max_x_nm = max(xs)
                    min_y_nm = min(ys)
                    max_y_nm = max(ys)

                    self.annotated_region['tl'] = (min_x_nm, min_y_nm)
                    self.annotated_region['br'] = (max_x_nm, max_y_nm)
                    break # currently assuming that the first ndpviewstate is the only annotated region. Change later
        
        return None
    
    def populate_annotations(self):
        """
        This function populates the annotations attribute of the Ndpa object. 
        Inputs:
            - None
        Returns:
            - None
            - Action: adds key-value pairs to self.annotations attribute. In particular, it adds < id : Annotation object >
                    - keys: id is an integer representing the annotation id within the ndpa file. 
                    - values: Annotation object is an object of the Annotation class that stores information about each annotation
        """
        tree = ET.parse(self.absolute_file_path)
        root = tree.getroot()
        for viewstate in root.findall("ndpviewstate"):
            annotation_tree_element = viewstate.find("annotation")
            if annotation_tree_element is not None and annotation_tree_element.get("displayname") == "AnnotateCircle":
                label = viewstate.find("title").text if viewstate.find("title") is not None else "ERROR"
                annot_id = int(viewstate.get("id"))
                self.annotations[annot_id] = Annotation(annot_id, label, annotation_tree_element)
