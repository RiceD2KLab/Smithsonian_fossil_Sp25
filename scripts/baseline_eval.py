import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evaluation.baseline.baseline_eval import calculate_baseline_mAP
from src import config
from src.modeling.baseline import baseline_config

if __name__ == "__main__":
    # Evaluate model results and save to specified directory
    print("STARTING calculating mAP for predictions")
    result_mAP = calculate_baseline_mAP(config["abs_path_to_location_for_master_annotation_csv"], baseline_config["abs_path_to_baseline_model_outputs"], baseline_config["confidence_threshold_for_predictions"])
    with open(os.path.join(baseline_config["abs_path_to_baseline_eval_results_dir"], "baseline_eval_results.txt"), "w") as f:
        for key, value in result_mAP.items():
            f.write(f"{key}: {value}\n")
    print(f"FINISHED calculating mAP for predictions, and saved results to {baseline_config['abs_path_to_baseline_eval_results_dir']}")

