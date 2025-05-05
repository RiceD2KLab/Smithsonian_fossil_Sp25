import os
import json

def load_config(path="../src/modeling/faster_rcnn/faster_rcnn_config.json"):
    with open(path, "r") as f:
        return json.load(f)
    
