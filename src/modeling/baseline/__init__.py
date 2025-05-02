import os
import json

def load_baseline_config():
    config_path = os.path.join(os.path.dirname(__file__), 'baseline_config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

baseline_config = load_baseline_config()