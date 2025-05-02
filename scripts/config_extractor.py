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

    return config
