import json
import os

def load_config(config_path):
    """
    Loads the config JSON file and validates required fields.

    Args:
        config_path (str): Path to the config.json file

    Returns:
        dict: A dictionary with config values
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Required keys
    required_keys = [
        "confidence_prediction_threshold",
        "nms_iou_threshold",
        "num_labels",
        "model_name",
        "weighs_path",
        "ndpa_output_dir"
    ]

    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    # Optionally normalize paths
    # config["weighs_path"] = os.path.abspath(config["weighs_path"])
    # config["ndpa_output_dir"] = os.path.abspath(config["ndpa_output_dir"])

    return config
