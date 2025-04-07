import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing.annotation_transformer import create_master_annotation_csv

if __name__ == "__main__":
    # create master annotation csv
    # NOTE: this assumes that the ndpi images have already been tiled!
    create_master_annotation_csv()