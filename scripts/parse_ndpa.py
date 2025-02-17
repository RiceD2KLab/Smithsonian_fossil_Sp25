import xml.etree.ElementTree as ET
import csv
import sys

def parse_ndpa_file(file_name):
    """
    Parses an .ndpa fileand extracts circle annotation data, saving it to a CSV file.
    
    Args:
        file_name (str): Path to the .ndpa file.
    """
    tree = ET.parse(file_name)
    root = tree.getroot()
    
    annotations = []
    
    for viewstate in root.findall("ndpviewstate"):
        annotation = viewstate.find("annotation")
        if annotation is not None and annotation.get("type") == "circle":
            annotation_data = {
                "id": viewstate.get("id"),
                "x": int(annotation.find("x").text),
                "y": int(annotation.find("y").text),
                "radius": int(annotation.find("radius").text)
            }
            annotations.append(annotation_data)
    
    # write to a csv
    output_csv = "annotations.csv"
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["id", "x", "y", "radius"])
        writer.writeheader()
        writer.writerows(annotations)
    
    print(f"Data successfully written to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <ndpa_file>")
        sys.exit(1)
    
    if sys.argv[1] in ["-h", "--help"]:
        print("Usage: python script.py <ndpa_file>")
        print("Parses an NDPA file and extracts circle annotations, saving them to annotations.csv.")
        sys.exit(0)
    
    parse_ndpa_file(sys.argv[1])