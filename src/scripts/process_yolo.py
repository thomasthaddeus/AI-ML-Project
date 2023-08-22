"""process_yolo.py

_summary_

_extended_summary_

Returns:
    _type_: _description_
"""

import os
import xml.etree.ElementTree as ET
import pandas as pd


def xml_to_annotations(xml_string):
    """Convert a given XML string to the consistent annotation format."""
    root = ET.fromstring(xml_string)
    image_path = root.find('path').text if root.find('path') is not None else root.find('filename').text
    width = float(root.find('size/width').text)
    height = float(root.find('size/height').text)
    annotations = []
    for obj in root.findall('object'):
        annotation = {
            'class': obj.find('name').text,
            'xmin': float(obj.find('bndbox/xmin').text),
            'ymin': float(obj.find('bndbox/ymin').text),
            'xmax': float(obj.find('bndbox/xmax').text),
            'ymax': float(obj.find('bndbox/ymax').text)
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
            'xmin': float(parts[i]),
            'ymin': float(parts[i+1]),
            'xmax': float(parts[i] + parts[i+2]),
            'ymax': float(parts[i+1] + parts[i+3])
        }
        annotations.append(annotation)
    return {
        'image_path': image_path,
        'annotations': annotations
    }

def process_xml_file(xml_file_path):
    with open(xml_file_path, 'r') as f:
        xml_data = f.read()
    return xml_to_annotations(xml_data)

def process_yolo_file(yolo_file_path, image_dir):
    annotations = []
    with open(yolo_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            annotations.append(parse_yolo_line(line, image_dir))
    return annotations

def main():
    xml_directory = 'data/processed/xml'  # Directory containing all XML files
    yolo_directory = 'data/processed/txt'  # Directory containing all YOLO-formatted text files

    all_xml_annotations = []
    all_yolo_annotations = []

    # Process XML files
    for filename in os.listdir(xml_directory):
        if filename.endswith('.xml'):
            xml_file_path = os.path.join(xml_directory, filename)
            all_xml_annotations.append(process_xml_file(xml_file_path))

    # Process YOLO files
    for yolo_file in os.listdir(yolo_directory):
        if yolo_file.endswith('.txt'):
            yolo_file_path = os.path.join(yolo_directory, yolo_file)
            all_yolo_annotations.extend(process_yolo_file(yolo_file_path, yolo_directory))

    # Convert to DataFrame and save to JSON
    df_xml = pd.DataFrame(all_xml_annotations)
    df_yolo = pd.DataFrame(all_yolo_annotations)

    # Combine both DataFrames
    combined_df = pd.concat([df_xml, df_yolo], ignore_index=True)

    # Save to JSON
    combined_df.to_json('annotations.json', orient='records', lines=True)

if __name__ == "__main__":
    main()
