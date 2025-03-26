import openslide
"""
NEW DOCSTRING:
This function extracts the metadata for an inputted ndpi file

Inputs:
    - intput_ndpi_file_path: absolute path to the ndpi file to extract 

Returns:
    - a mapping of strings to strings, where the keys represent the metadata feature name, 
        and the keys represent the value for the corresponding feature as a string. 

Throws:
    - an exception if there is some error in extracting the metadata for the inputted ndpi file





OLD DOCSTRING
This function extracts the metadata for an inputted ndpi file

Inputs: 
    - input_ndpi_file_path: absolute path to the ndpi file to extract metadata for. 

Returns:
    - a two-element tuple: first element is the x coordinate (in nm) of the center of the ndpi file. Second element is the y coordiante (in nm) of the center of the ndpi file. 
    note: this coordinate location is relative to the whole nanozoomer scanned space
    - ndpi_width_nm: the width of the ndpi file in nm
    - ndpi_height_nm: the height of the ndpi file in nm
    - mmpp_x: the millimeters per pixel in the x direction according to ndpi metadata
    - mmpp_y: the millimeters per pixel in the y direction according to ndpi metadata
"""
def extract_ndpi_metadata(input_ndpi_file_path):
    try:
        slide = openslide.OpenSlide(input_ndpi_file_path)
        # recreate the metadata as a python dictionary for return
        metadata = {}
        for key, value in slide.properties.items():
            metadata[key] = value
        slide.close()
        return metadata
    except Exception as e:
        print(f"Error extracting metadata for {input_ndpi_file_path}: {e}")
        return None
    # print(type(slide.properties))
    # print(slide.properties)
    
    # ndpi_center_x = int(slide.properties["hamamatsu.XOffsetFromSlideCentre"])
    # ndpi_center_y =int(slide.properties["hamamatsu.YOffsetFromSlideCentre"])
    # mmpp_x = float(slide.properties["openslide.mpp-x"])
    # mmpp_y = float(slide.properties["openslide.mpp-y"])
    # ndpi_width_px = int(slide.properties["openslide.level[0].width"])
    # ndpi_height_px = int(slide.properties["openslide.level[0].height"])

    # # convert ndpi width and height from pixels to nm
    # ndpi_width_nm = mmpp_x * 1000 * ndpi_width_px
    # ndpi_height_nm = mmpp_y * 1000 * ndpi_height_px

    # return (ndpi_center_x, ndpi_center_y), ndpi_width_nm, ndpi_height_nm, mmpp_x, mmpp_y