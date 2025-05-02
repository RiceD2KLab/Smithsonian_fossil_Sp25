from collections import defaultdict
from torchmetrics.detection.mean_ap import MeanAveragePrecision

"""
This function extracts all the ground truth bounding boxes for the annotations in the master_anotation_csv in src/config.json

Inputs:
    - abs_path_to_master_annotation_csv: a string representing the absolute path to the master_annotation_csv

Returns:
    - gt_boxes: a mapping where the keys are strings representing filenames, and the values are a list of lists, 
        where the internal lists are the coordiantes representing the bounding box, in the format [x1, y1, x2, y2], 
        where (x1, y1) represents the top-left coordinate, and (x2, y2) represents the bottom-right coordinate
        ex. {"D9151-A-2_L_2024_02_02_16_08_52_Texas" : [[10, 10, 50, 50], [100, 100, 150, 150]]}
    - gt_labels: a mapping where the keys are strings representing filenames, and the values are a list of n 0s, 
        where n is the number of ground truth boxes in gt_boxes. 0 is the label representing pollen. 
        ex. {"D9151-A-2_L_2024_02_02_16_08_52_Texas" : [0, 0]}
"""
def extract_all_ground_truth_bboxes(abs_path_to_master_annotation_csv):
    # extracting ground_truth bboxes
    gt_boxes = defaultdict(list)
    gt_labels = defaultdict(list)

    # Iterate through every annotation in master_annotation_csv
    with open(abs_path_to_master_annotation_csv, mode='r', newline='') as master_annotation_csv:
        reader = csv.DictReader(master_annotation_csv)
        for row in reader:
            # retrieve top left and bottom right bounding box coordinates and convert to list/workable formate
            tl = row["TL"]
            br = row["BR"]
            tl = ast.literal_eval(tl)
            tl = list(tl)
            br = ast.literal_eval(br)
            br = list(br)
            
            # retrieve the original tile_id of the annotation
            tile = row["tile_id"]
            x_str, y_str = tile.split('_')       # ['6365x', '55347y']
            tile_tl_x = int(x_str.rstrip('x'))        # remove trailing 'x'
            tile_tl_y = int(y_str.rstrip('y'))        # remove trailing 'y'

            # convert the tl, br coordinates into the ndpi wide pixel-wise coordinates
            tl[0] += tile_tl_x
            br[0] += tile_tl_x
            tl[1] += tile_tl_y
            br[1] += tile_tl_y

            tl.extend(br) # formatting quirk

            gt_boxes[row["filename"]].append(tl)
            gt_labels[row["filename"]].append(0)
    
    return gt_boxes, gt_labels


"""
This function extracts all the prediction bounding boxes predicted by the baseline model. 

Inputs:
    - abs_path_to_detections_dir: string representing the absolute path to the directory of segmentation detections

Returns:
    - pred_boxes: a mapping where the keys are strings representing filenames, and the values are a list of lists, 
        where the internal lists are the coordiantes representing the bounding box, in the format [x1, y1, x2, y2], 
        where (x1, y1) represents the top-left coordinate, and (x2, y2) represents the bottom-right coordinate
        ex. {"D9151-A-2_L_2024_02_02_16_08_52_Texas" : [[10, 10, 50, 50], [100, 100, 150, 150]]}
    - pred_scores: a mapping where the keys are strings representing filenames, and the values are a list of floats
        where the floats represent the confidence scores of the boxes in the same order of the boxes listed in the 
        values for each key-value pair of an image in pred_boxes
        ex. {"D9151-A-2_L_2024_02_02_16_08_52_Texas" : [0.8, 0.7]}
    - pred_labels: a mapping where the keys are strings representing filenames, and the values are a list of n 0s, 
        where n is the number of boxes in pred_boxes. 0 is the label representing pollen. 
        ex. {"D9151-A-2_L_2024_02_02_16_08_52_Texas" : [0, 0]}
"""
def extract_all_prediction_bboxes(abs_path_to_detections_dir):
    pred_boxes = defaultdict(list)
    pred_scores = defaultdict(list)
    pred_labels = defaultdict(list)

    for detection in os.listdir(abs_path_to_detections_dir):
        # load in the metadata that stores all the information about each prediction include the coordiantes and confidence score
        with open(os.path.join(abs_path_to_detections_dir, detection, "metadata_1.json"), 'r') as f:
            metadata = json.load(f)
        conf_score = metadata["confidence"]
        """
        note: "pollen_image_coordinates" are the coordinates of the bbox within the tile. 
            - so, a coordinate of (0,0) would represent the top left of the bbox
        """ 
        pred_bbox = list(ast.literal_eval(metadata["pollen_image_coordinates"])) # list of tuples
        tl = pred_bbox[0]
        tl = list(tl)
        br = pred_bbox[1]
        br = list(br)
        
        # multiply coordinates by 2. This is because the tiles are downsized from size 2048 to size 1024
        tl[0] *= 2
        tl[1] *= 2
        br[0] *= 2
        br[1] *= 2

        # add the coords to their tile to get ndpi pixel-wise global coordinate
        tile_id = metadata["tile_image_coordinates"]
        x_str, y_str = tile_id.split('_')       # ['6365x', '55347y']
        tile_tl_x = int(x_str.rstrip('x'))        # remove trailing 'x'
        tile_tl_y = int(y_str.rstrip('y'))        # remove trailing 'y'
        tl[0] += tile_tl_x
        tl[1] += tile_tl_y
        br[0] += tile_tl_x
        br[1] += tile_tl_y

        tl.extend(br) # formatting quirk

        pred_boxes[metadata["sample_filename"]].append(tl)
        pred_scores[metadata["sample_filename"]].append(conf_score)
        pred_labels[metadata["sample_filename"]].append(0)
    
    return pred_boxes, pred_scores, pred_labels

"""
This function calculates the mean average precision of the detections of the baseline segmentation model.
Note, that the baseline segmentation model only segements with respect to a single class, so the 
mean average precision is really just average precision. 

Inputs:
    - abs_path_to_master_annotation_csv: a string representing the absolute path to the master_annotation_csv
    - abs_path_to_detections_dir: string representing the absolute path to the directory of segmentation detections

Returns:
    - the mean average precision results
"""
def calculate_baseline_mAP(abs_path_to_master_annotation_csv, abs_path_to_detections_dir):
    all_targets = []
    all_preds = []

    # extract ground truth bounding boxes and convert to a formate easy for mAP calculation
    gt_boxes, gt_labels = extract_all_ground_truth_bboxes(abs_path_to_master_annotation_csv)
    for filename, boxes in gt_boxes.items():
        all_targets.append({'boxes': torch.tensor(boxes), 'labels': torch.tensor(gt_labels[filename])})
    
    # extract prediction bounding boxes and convert to a format easy for mAP calculation
    pred_boxes, pred_scores, pred_labels = extract_all_prediction_bboxes(abs_path_to_detections_dir)
    for filename, boxes in pred_boxes.items():
        all_preds.append({'boxes': torch.tensor(boxes), 'scores': torch.tensor(pred_scores[filename]), 'labels': torch.tensor(pred_labels[filename])})
    

    metric = MeanAveragePrecision(iou_type="bbox", box_format="xyxy")
    for p, t in zip(all_preds, all_targets):
        metric.update([p], [t])  # each image handled independently

    results = metric.compute()

    return results
