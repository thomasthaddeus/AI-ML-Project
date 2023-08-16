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


    def __init__(self):
        # Default values (can be overridden by config)
        self.OUTPUT = './data/processed/dataset2/json/annotations.json'
        self.FOLDER_IN = './data/processed/dataset2/img'
        self.DS3_TRN = "./data/processed/ds3_trn.txt"
        self.DS3_TST = "./data/processed/ds3_tst.txt"
        self.DS4_TRN = "./data/processed/ds4_trn.txt"
        self.DS4_TST = "./data/processed/ds4_tst.txt"
        self.XFILE_DIR = './data/processed/xml_data/'

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
        datasets = [self.DS3_TRN, self.DS3_TST, self.DS4_TRN, self.DS4_TST]
        for dataset in datasets:
            df = self.line_to_dataframe(dataset)
            json_filename = dataset.replace(".txt", ".json").replace(
                "/processed/", "/processed/json/"
            )
            df.to_json(json_filename, orient="records", lines=True)
            print(f"Processed {dataset} and saved to {json_filename}")

    @staticmethod
    def from_txt(txt_file):
        """
        Parse a YOLO-formatted .txt file and extract the annotation data.

        Given a path to a .txt file containing YOLO-formatted annotations, this
        function reads the file and extracts the class ID, bounding box
        coordinates (center x, center y, width, height), and returns them in a
        dictionary format.

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

    def txt_to_dataframe(self, dataset_dir):
        """
        Parse a directory containing images and their annotations to create a
        DataFrame.

        This function iterates over all the images in the specified directory,
        reads their associated YOLO-formatted .txt annotation files, and
        aggregates the annotations into a structured Pandas DataFrame.

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
                    annotation = self.from_txt(txt_filepath)
                    annotation['image'] = filename
                    rows.append(annotation)
        return pd.DataFrame(rows)


    @staticmethod
    def from_line(line):
        """
        Parse a single line from the dataset to extract image path and bounding
        boxes.

        Given a line from the dataset file, this function extracts the image
        path, adjusts its file extension from .bmp to .jpg, and extracts the
        bounding box annotations associated with the image.

        Args:
            line (str): A line from the dataset file.

        Returns:
            dict: A dictionary containing the image path and its associated
            bounding box annotations.
        """
        parts = line.strip().split()
        image_path = parts[0].replace(".bmp", ".jpg")
        num_boxes = int(parts[1])
        boxes = [
            {
                "x": int(parts[i]),
                "y": int(parts[i + 1]),
                "width": int(parts[i + 2]),
                "height": int(parts[i + 3]),
            }
            for i in range(2, 2 + 4 * num_boxes, 4)
        ]
        return {"image_path": image_path, "boxes": boxes}

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
    def from_xml(xml_file):
        """
        Convert a given XML file to a Pandas DataFrame.

        This function reads an XML file, typically in the PASCAL VOC format, and
        extracts the annotations for each object present in the corresponding
        image. The annotations include the object's class, bounding box
        coordinates, and the image's dimensions.

        Args:
            xml_file (str): Path to the XML file to be parsed.

        Returns:
            DataFrame: A Pandas DataFrame containing the annotations from the
            XML file.
        """
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text
        rows = []
        for obj in root.findall('object'):
            row = {
                'filename': filename,
                'width': int(root.find('size/width').text),
                'height': int(root.find('size/height').text),
                'class': obj.find('name').text,
                'xmin': int(obj.find('bndbox/xmin').text),
                'ymin': int(obj.find('bndbox/ymin').text),
                'xmax': int(obj.find('bndbox/xmax').text),
                'ymax': int(obj.find('bndbox/ymax').text)
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def xml_to_dataframe(self, xml_dir):
        """
        Parse all XML files in a given directory and aggregate them into a
        single DataFrame.

        This function iterates over all XML files in the specified directory,
        converts each file to a DataFrame using the xml_to_dataframe function,
        and then aggregates them into a single DataFrame.

        Args:
            xml_dir (str): Directory containing the XML files to be parsed.

        Returns:
            DataFrame: A Pandas DataFrame containing the aggregated annotations
            from all XML files.
        """
        all_data = []
        for i in os.listdir(xml_dir):
            if i.endswith('.xml'):
                df = self.from_xml(os.path.join(xml_dir, i))
                all_data.append(df)
        if not all_data:
            print("No XML files found in the specified directory.")
            return
        return pd.concat(all_data, ignore_index=True)
