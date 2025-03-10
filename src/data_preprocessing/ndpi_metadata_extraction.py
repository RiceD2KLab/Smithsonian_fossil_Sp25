import os
import json
import openslide

def extract_metadata(ndpi_path):
    """Extract metadata from an NDPI file."""
    try:
        slide = openslide.OpenSlide(ndpi_path)
        metadata = {
            'dimensions': slide.dimensions,
        }
        for key, value in slide.properties.items():
            metadata[key] = value
        slide.close()
        return metadata
    except Exception as e:
        print(f"Error reading {ndpi_path}: {e}")
        return None

def extract_ndpi_metadata(input_dir, output_dir):
    """Process all NDPI files in input_dir and save metadata as JSON in output_dir."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.ndpi'):
            ndpi_path = os.path.join(input_dir, filename)
            metadata = extract_metadata(ndpi_path)
            
            if metadata:
                json_filename = os.path.splitext(filename)[0] + '.json'
                json_path = os.path.join(output_dir, json_filename)
                
                with open(json_path, 'w') as json_file:
                    json.dump(metadata, json_file, indent=4)
                print(f"Saved metadata: {json_path}")

if __name__ == "__main__":
    input_directory = "/storage/hpc/work/dsci435/smithsonian/ndpi_files/"  
    output_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../tmp/ndpi_metadata_json"))
    extract_ndpi_metadata(input_directory, output_directory)
