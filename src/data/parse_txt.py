"""parse_txt.py
This module provides utilities to parse annotations from .txt files and map
them to their corresponding images.

Given a directory containing images and their associated YOLO-formatted
annotation files, the module provides functions to extract the annotations and
create a structured Pandas DataFrame. This DataFrame can then be exported to a
JSON format for further processing or analysis.

Returns:
    DataFrame: A Pandas DataFrame containing the parsed annotations mapped to
    their respective images.
"""

import os
import pandas as pd

OUTPUT = './data/processed/dataset2/json/annotations.json'
FOLDER_IN = './data/processed/dataset2/img'


def parse_txt_file(txt_file):
    """
    Parse a YOLO-formatted .txt file and extract the annotation data.

    Given a path to a .txt file containing YOLO-formatted annotations, this
    function reads the file and extracts the class ID, bounding box coordinates
    (center x, center y, width, height), and returns them in a dictionary
    format.

    Args:
        txt_file (str): Path to the YOLO-formatted annotation .txt file.

    Returns:
        dict: A dictionary containing the parsed annotation data.
    """
    with open(txt_file, mode='r', encoding='utf-8') as f:
        data = f.readline().strip().split()
    return {
        'class_id': int(data[0]),
        'x_center': float(data[1]),
        'y_center': float(data[2]),
        'width': float(data[3]),
        'height': float(data[4])
    }

def main(dataset_dir):
    """
    Parse a directory containing images and their annotations to create a
    DataFrame.

    This function iterates over all the images in the specified directory,
    reads their associated YOLO-formatted .txt annotation files, and aggregates
    the annotations into a structured Pandas DataFrame.

    Args:
        dataset_dir (str): Directory containing the images and their
        associated .txt annotation files.

    Returns:
        DataFrame: A Pandas DataFrame containing the annotations mapped to
        their respective images.
    """
    rows = []
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.jpg'):
            txt_filename = filename.replace('.jpg', '.txt')
            txt_filepath = os.path.join(dataset_dir, txt_filename)
            if os.path.exists(txt_filepath):
                annotation = parse_txt_file(txt_filepath)
                annotation['image'] = filename
                rows.append(annotation)
    return pd.DataFrame(rows)

# Usage
df = main(FOLDER_IN)
print(df.head())

# Export to JSON
df.to_json(OUTPUT, orient='records', lines=True)
