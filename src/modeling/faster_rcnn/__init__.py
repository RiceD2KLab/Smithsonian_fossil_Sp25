import os
import json

def load_config(path="/projects/dsci435/smithsonian_sp25/Smithsonian_fossil_Sp25/src/modeling/faster_rcnn/faster_rcnn_config.json"):
    with open(path, "r") as f:
        return json.load(f)
    
