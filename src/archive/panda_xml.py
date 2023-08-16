"""parse_xml.py
Converts XML annotations (typically in PASCAL VOC format) to a Pandas DataFrame
and exports them to a JSON format.

This module provides functions to read XML annotations, convert them to a
structured Pandas DataFrame, and then export the data to a JSON file. The
primary use case is for image annotations used in object detection tasks.

Returns:
    DataFrame: A Pandas DataFrame containing the parsed XML annotations.
"""

import os
import xml.etree.ElementTree as ET
import pandas as pd

XFILE_DIR = './data/processed/img/ds10_norway/xmls/'


def xml_to_dataframe(xml_file):
    """
    Convert a given XML file to a Pandas DataFrame.

    This function reads an XML file, typically in the PASCAL VOC format, and
    extracts the annotations for each object present in the corresponding
    image. The annotations include the object's class, bounding box
    coordinates, and the image's dimensions.

    Args:
        xml_file (str): Path to the XML file to be parsed.

    Returns:
        DataFrame: A Pandas DataFrame containing the annotations from the XML
        file.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract image filename
    filename = root.find('filename').text

    rows = []
    for obj in root.findall('object'):
        row = {
            'filename': filename,
            'width': int(root.find('size/width').text),
            'height': int(root.find('size/height').text),
            'class': obj.find('name').text,
            'xmin': float(obj.find('bndbox/xmin').text),
            'ymin': float(obj.find('bndbox/ymin').text),
            'xmax': float(obj.find('bndbox/xmax').text),
            'ymax': float(obj.find('bndbox/ymax').text)
        }
        rows.append(row)
    return pd.DataFrame(rows)

def main(xml_dir):
    """
    Parse all XML files in a given directory and aggregate them into a single
    DataFrame.

    This function iterates over all XML files in the specified directory,
    converts each file to a DataFrame using the xml_to_dataframe function, and
    then aggregates them into a single DataFrame.

    Args:
        xml_dir (str): Directory containing the XML files to be parsed.

    Returns:
        DataFrame: A Pandas DataFrame containing the aggregated annotations
        from all XML files.
    """
    all_data = []
    for i in os.listdir(xml_dir):
        # print(f"Processing: {i}")
        if i.endswith('.xml'):
            df1 = xml_to_dataframe(os.path.join(xml_dir, i))
            all_data.append(df1)
    if not all_data:
        print("No XML files found in the specified directory.")
        return
    return pd.concat(all_data, ignore_index=True)

# Usage
df = main(XFILE_DIR)
print(df.head())
df.to_json('./data/processed/json/ds10_annotations.json', orient='records')
