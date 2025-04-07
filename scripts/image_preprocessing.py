import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing.cropper_tool_runner import annotated_region_cropper

if __name__ == "__main__":
    # run tile cropper
    annotated_region_cropper()