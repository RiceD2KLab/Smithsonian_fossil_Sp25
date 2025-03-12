import xml.etree.ElementTree as ET
import csv
import sys
import os

def parse_ndpa_file(file_path, annotations):
    """
    Parses an .ndpa file (XML format) and extracts circle annotation data, appending it to the annotations list.
    
    Args:
        file_path (str): Path to the .ndpa file.
        annotations (list): List to store extracted annotation data.
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    filename = os.path.basename(file_path).replace(".ndpi.ndpa", "")
    
    for viewstate in root.findall("ndpviewstate"):
        annotation = viewstate.find("annotation")
        if annotation is not None and annotation.get("type") == "circle":
            x = int(annotation.find("x").text)
            y = int(annotation.find("y").text)
            radius = int(annotation.find("radius").text)
            pollen_type = viewstate.find("title").text if viewstate.find("title") is not None else ""
            
            # Calculate bounding box coordinates
            TL = (x - radius, y - radius)  # Top-left
            BL = (x - radius, y + radius)  # Bottom-left
            TR = (x + radius, y - radius)  # Top-right
            BR = (x + radius, y + radius)  # Bottom-right
            side_length = 2 * radius  # Side length of bounding box
            
            annotation_data = {
                "filename": filename,
                "id": viewstate.get("id"),
                "pollen_type": pollen_type,
                "x": x,
                "y": y,
                "radius": radius,
                "TL": TL,
                "BL": BL,
                "TR": TR,
                "BR": BR,
                "side_length": side_length,
                "tile_id": ""  # Empty for now
            }
            annotations.append(annotation_data)

def process_directory(directory):
    """
    Processes all .ndpa files in a directory and writes extracted circle annotation data to a CSV file.
    
    Args:
        directory (str): Path to the directory containing .ndpa files.
    """
    annotations = []
    
    for file_name in os.listdir(directory):
        if file_name.endswith(".ndpa"):
            file_path = os.path.join(directory, file_name)
            parse_ndpa_file(file_path, annotations)
    
    # Write to CSV
    output_csv = "annotations.csv"
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["filename", "id", "pollen_type", "x", "y", "radius", "TL", "BL", "TR", "BR", "side_length", "tile_id"])
        writer.writeheader()
        writer.writerows(annotations)
    
    print(f"Data successfully written to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <directory>")
        print("       python script.py -h (for help)")
        sys.exit(1)
    
    if sys.argv[1] in ["-h", "--help"]:
        print("Usage: python script.py <directory>")
        print("Parses all NDPA files in the given directory and extracts circle annotations, saving them to annotations.csv.")
        sys.exit(0)
    
    
    process_directory(sys.argv[1])
