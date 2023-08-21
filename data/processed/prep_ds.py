"""xml2json.py
This module provides utilities to parse annotations from .txt files and map them to their corresponding images.

Converts XML annotations (typically in PASCAL VOC format) to a Pandas DataFrame and exports them to a JSON format.

Given a directory containing images and their associated YOLO-formatted annotation files, the module provides functions to extract the annotations and create a structured Pandas DataFrame. This DataFrame can then be exported to a JSON format for further processing or analysis.

Returns:
    DataFrame: A Pandas DataFrame containing the parsed annotations mapped to their respective images.
"""

import os
import xml.etree.ElementTree as ET
import pandas as pd

def xml_to_annotations(xml_file):
    """Convert a given XML file to the consistent annotation format."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    image_path = root.find('path').text
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    annotations = []
    for obj in root.findall('object'):
        annotation = {
            'class': obj.find('name').text,
            'xmin': int(obj.find('bndbox/xmin').text),
            'ymin': int(obj.find('bndbox/ymin').text),
            'xmax': int(obj.find('bndbox/xmax').text),
            'ymax': int(obj.find('bndbox/ymax').text)
        }
        annotations.append(annotation)
    return {
        'image_path': image_path,
        'width': width,
        'height': height,
        'annotations': annotations
    }

def parse_yolo_line(line, image_dir):
    """Parse a YOLO-formatted line and extract the annotation data."""
    parts = line.strip().split()
    image_path = os.path.join(image_dir, parts[0])
    width = 720  # Placeholder, as width and height are not provided in YOLO format
    height = 720
    annotations = [{
        'class': 'pothole',  # Placeholder, as class is not provided in YOLO format
        'xmin': int((float(parts[1]) - float(parts[3])/2) * width),
        'ymin': int((float(parts[2]) - float(parts[4])/2) * height),
        'xmax': int((float(parts[1]) + float(parts[3])/2) * width),
        'ymax': int((float(parts[2]) + float(parts[4])/2) * height)
    }]
    return {
        'image_path': image_path,
        'width': width,
        'height': height,
        'annotations': annotations
    }

def parse_dataset_line(line):
    """Parse a dataset line and extract the annotation data."""
    parts = line.strip().split()
    image_path = parts[0]
    num_boxes = int(parts[1])
    annotations = []
    for i in range(2, 2 + 4 * num_boxes, 4):
        annotation = {
            'class': 'pothole',  # Placeholder, as class is not provided in this format
            'xmin': int(parts[i]),
            'ymin': int(parts[i+1]),
            'xmax': int(parts[i] + parts[i+2]),
            'ymax': int(parts[i+1] + parts[i+3])
        }
        annotations.append(annotation)
    return {
        'image_path': image_path,
        'annotations': annotations
    }

def main():
    # Placeholder for demonstration purposes
    xml_annotations = xml_to_annotations('path_to_xml_file.xml')
    yolo_annotations = parse_yolo_line('0 0.506250 0.395139 0.220833 0.195833', 'path_to_images')
    dataset_annotations = parse_dataset_line('trn/positive/G0010033.jpg 6 1990 1406 66 14 1464 1442 92 16 1108 1450 54 16 558 1434 102 16 338 1450 72 18 262 1450 58 22')

if __name__ == "__main__":
    main()
