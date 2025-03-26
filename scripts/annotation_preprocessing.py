from data_preprocessing.annotation_transformer import create_master_annotation_csv

if __name__ == "__main__":
    # create master annotation csv
    # NOTE: this assumes that the ndpi images have already been tiled!
    create_master_annotation_csv()