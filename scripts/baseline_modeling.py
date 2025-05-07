import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.modeling.baseline.baseline import baseline_config_setup, baseline_input_preparation, baseline_run, save_ndpas
from src.evaluation.baseline.baseline_eval import calculate_baseline_mAP
from src import config
from src.modeling.baseline import baseline_config

if __name__ == "__main__":
    # Run setup configuration
    baseline_config_setup()
    print("FINISHED configuring settings for baseline model")

    # Run baseline input preparation
    print("STARTING baseline model input preparation")
    baseline_input_preparation(config["abs_path_to_ndpi_tiles_dir"], baseline_config["abs_path_to_reformatted_tiles_directory"])
    print("FINISHED baseline model input preparation")

    # # Run baseline model
    print("STARTING running baseline model")
    baseline_run(baseline_config["abs_path_to_reformatted_tiles_directory"], baseline_config["abs_path_to_baseline_model_outputs"])
    print("FINISHED running baseline model")

    # Save prediction NDPA files
    print("STARTING saving NDPA files")
    save_ndpas(baseline_config["abs_path_to_baseline_model_outputs"], 
                baseline_config["abs_path_to_ndpa_output_directory"], 
                baseline_config["confidence_threshold_for_predictions"],
                config["abs_path_to_ndpi_dir"])
    print("FINISHED saving NDPA files")
    

