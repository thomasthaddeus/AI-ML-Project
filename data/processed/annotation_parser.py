"""annotation_parser.py

This module provides utilities to parse annotations from various formats (txt,
line, and XML) and map them to their corresponding images. The primary use
cases include:

1. Parsing annotations from .txt files, especially those in YOLO format.
2. Extracting bounding box annotations from dataset files where each line
represents an image and its associated bounding boxes.
3. Converting XML annotations, typically in PASCAL VOC format, into a
structured format.

Given a directory or file with annotations, the module offers functions to
extract these annotations, adjust any necessary file extensions, and create a
structured Pandas DataFrame. This DataFrame can then be exported to a JSON
format for further processing, analysis, or model training.

The module consolidates the functionality of previously separate utilities into
a single, cohesive class, making it easier to handle various annotation formats
in a unified manner.

Returns:
    DataFrame: A Pandas DataFrame containing the parsed annotations mapped to
    their respective images.

Usage:
    To load the config file into the class:
        parser = AnnotationParser()
        parser.load_config('config.ini')
        parser.process_all_datasets()
"""

import os
import xml.etree.ElementTree as ET
import configparser
import pandas as pd


class AnnotationParser:
    """
    A utility class for parsing image annotations from various formats.

    The AnnotationParser class provides methods to extract annotations from
    different formats such as .txt (YOLO format), line-based datasets, and XML
    (typically PASCAL VOC format). It consolidates the functionality of
    previously separate utilities into a unified interface, allowing for
    streamlined processing and conversion of annotations into a structured
    Pandas DataFrame. This DataFrame can then be exported to a JSON format
    for further processing or analysis.

    Attributes:
        OUTPUT (str): Default path for the output JSON file.
        FOLDER_IN (str): Default directory containing images and their
          annotations.
        DS3_TRN, DS3_TST, DS4_TRN, DS4_TST (str): Default paths for dataset
          files.
        XFILE_DIR (str): Default directory for XML files.

    Methods:
        load_config(config_file): Load parameters from a configuration file.
        process_all_datasets(): Process all dataset constants and create
        annotation sets for each.
        ... [other methods]

    Usage:
        parser = AnnotationParser()
        parser.load_config('config.ini')
        parser.process_all_datasets()

    Returns:
        DataFrame: A Pandas DataFrame containing the parsed annotations mapped
        to their respective images.
    """
    OUTPUT = './data/processed/annotations.json'
    DS_TRAIN = "./data/processed/trn"
    DS_TEST = "./data/processed/test"
    XFILE_DIR = './data/processed/xml_data/'


    def load_config(self, config_file):
        """
        Load parameters from a configuration file.

        Args:
            config_file (str): Path to the configuration (.ini) file.
        """
        config = configparser.ConfigParser()
        config.read(config_file)

        # Update attributes based on config
        if 'DEFAULT' in config:
            for key in config['DEFAULT']:
                setattr(self, key.upper(), config['DEFAULT'][key])

    def process_all_datasets(self):
        """
        Process all dataset constants and create annotation sets for each.

        This function iterates through each of the dataset constants, reads the
        dataset, creates a DataFrame of annotations, and then exports the
        DataFrame to a JSON file.
        """
        datasets = [DS_TRAIN, DS_TEST]
        for dataset in datasets:
            df = self.line_to_dataframe(dataset)
            json_filename = dataset.replace(".txt", ".json").replace(
                "/processed/", "/processed/json/"
            )
            df.to_json(json_filename, orient="records", lines=True)
            print(f"Processed {dataset} and saved to {json_filename}")

    def line_to_dataframe(self, dataset):
        """
        Read the dataset file and return a DataFrame with image paths and
        annotations.

        This function reads the specified dataset file, parses each line using
        the parse_line function, and aggregates the parsed data into a
        structured Pandas DataFrame.

        Args:
            dataset (str): Path to the dataset file.

        Returns:
            DataFrame: A Pandas DataFrame containing the image paths and their
            associated bounding box annotations.
        """

        with open(file=dataset, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        data = [self.from_line(line) for line in lines]
        return pd.DataFrame(data)

    @staticmethod
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

    def parse_yolo_line(self, line, image_dir):
        """Parse a YOLO-formatted line and extract the annotation data."""
        parts = line.strip().split()
        image_path = os.path.join(image_dir, parts[0])
        width = 720
        height = 720
        annotations = [{
            'class': 'pothole',
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

    def parse_dataset_line(self, line):
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

    def process_xml_directory(self, xml_directory):
        """Process all XML files in a directory and return their annotations."""
        all_xml_annotations = []
        for filename in os.listdir(xml_directory):
            if filename.endswith('.xml'):
                xml_file_path = os.path.join(xml_directory, filename)
                with open(file=xml_file_path, mode='r', encoding='utf-8') as f:
                    xml_data = f.read()
                all_xml_annotations.append(self.xml_to_annotations(xml_data))
        return all_xml_annotations

    def process_yolo_directory(self, yolo_directory, image_dir):
        """Process all YOLO-formatted files in a directory and return their annotations."""
        all_yolo_annotations = []
        for yolo_file in os.listdir(yolo_directory):
            if yolo_file.endswith('.txt' or '.csv'):
                yolo_file_path = os.path.join(yolo_directory, yolo_file)
                with open(file=yolo_file_path, mode='r', encoding='utf-8') as f:
                    lines = f.readlines()
                for line in lines:
                    all_yolo_annotations.append(self.parse_yolo_line(line, image_dir))
        return all_yolo_annotations

    def process_all_files(self, xml_directory='data/processed/xml', yolo_directory='data/processed/txt', image_dir='data/processed/trn'):
        """Process all XML and YOLO-formatted files and save the results to a JSON file."""
        all_xml_annotations = self.process_xml_directory(xml_directory)
        all_yolo_annotations = self.process_yolo_directory(yolo_directory, image_dir)

        # Convert to DataFrame and save to JSON
        df_xml = pd.DataFrame(all_xml_annotations)
        df_yolo = pd.DataFrame(all_yolo_annotations)

        # Combine both DataFrames
        combined_df = pd.concat([df_xml, df_yolo], ignore_index=True)

        # Save to JSON
        combined_df.to_json(self.OUTPUT, orient='records', lines=True)

process = AnnotationParser()
process.process_all_files()
