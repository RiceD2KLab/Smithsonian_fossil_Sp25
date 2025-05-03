# Baseline: Segmentation model
This model is a segmentation model that has been taken from ____________________

--insert run statement for baseline_modling_script--

This script will first prompt the user to input some configurations:
- directory location for the baseline outputs: The baseline model will generate a directory called "baseline_outputs". User will input a directory where the model can store this "baseline_outputs" folder
- prediction confidence threshold: This represents the confidence threshold at which to include predictions in the outputs and evaluation

After the model is finished running, the "baseline_outputs" folder will have the following structure:
--insert code block that represents the directory strucutre --
prediction_ndpas: This subdirectory stores all generated ndpa files that store predictions. The naming convention of the ndpa files is 'sample_name.ndpi_predictions.ndpa'.
- ex. D3094_1_L_2024_02_20_11_32_26_Colorado.ndpi_predictions.ndpa
baseline_eval_results.txt: this is a .txt file that stores the mean average precision results of evaluation for the model.
reformatted tiles: This subdirectory stores the 

