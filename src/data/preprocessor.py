"""preprocessor.py
Annotation Preprocessing Utility.

This module provides a utility class for preprocessing image annotations from
various formats. It allows for loading annotations, splitting datasets based on
provided criteria, and combining annotations from different datasets into a
unified format.

Returns:
    DataFrame: A Pandas DataFrame containing the preprocessed annotations.

prep2.py
This module provides a DatasetPreprocessor class to preprocess various datasets
for object detection tasks. The class includes methods to handle different
formats of datasets and convert them into a standardized format suitable for
object detection models. The standardized format includes center coordinates,
width-height format for bounding boxes, normalized coordinates, and mapped
class labels to class IDs.
"""

import os
import pandas as pd
import PIL.Image as Image
import numpy as np


class Preprocessor:
    """
    A utility class for preprocessing image annotations.

    The Preprocessor class provides methods to load annotations from JSON
    files, split datasets based on provided splits, and combine annotations
    from different datasets into a unified format.

    Attributes:
        parser (AnnotationParser): An instance of the AnnotationParser class.
        annotations (dict): A dictionary to store loaded annotations.
    """

    def __init__(self, parser):
        """
        Initialize the Preprocessor with a given parser.

        Args:
            parser (AnnotationParser): An instance of the AnnotationParser
            class.
        """
        self.parser = parser
        self.annotations = {}

    def load_annotations(self, *json_files):
        """
        Load annotations from the provided JSON files into a dictionary.

        Given a list of JSON file paths, this method reads each file and stores
        the annotations in a dictionary with the file name as the key.

        Args:
            *json_files (str): Paths to the JSON files containing annotations.
        """
        for file in json_files:
            df = pd.read_json(file, lines=True)
            self.annotations[file] = df

    def split_dataset(self, splits_file):
        """
        Split the dataset based on the provided splits file.

        Given a splits file, this method divides the dataset into training and
        validation sets based on the specified split criteria.

        Args:
            splits_file (str): Path to the JSON file containing split criteria.
        """

        splits = pd.read_json(splits_file)
        train_files = splits[splits["split"] == "train"]["filename"].tolist()
        val_files = splits[splits["split"] == "val"]["filename"].tolist()

        train_df = self.annotations["df1_annotations.json"][
            self.annotations["df1_annotations.json"]["filename"].isin(train_files)
        ]
        val_df = self.annotations["df1_annotations.json"][
            self.annotations["df1_annotations.json"]["filename"].isin(val_files)
        ]

        self.annotations["df1_train"] = train_df
        self.annotations["df1_val"] = val_df

    def preprocess(self):
        """
        Combine annotations from different datasets into a unified format.

        This method aggregates annotations from different datasets into a
        single DataFrame, ensuring a consistent format for further processing
        or analysis.

        Returns:
            DataFrame: A Pandas DataFrame containing the combined annotations.
        """
        all_data = pd.concat(
            [
                df
                for key, df in self.annotations.items()
                if key not in ["df1_annotations.json", "df1_splits.json"]
            ]
        )
        return all_data

    def load_data(self, json_file, img_dir):
        """
        Load image and mask data from the provided JSON file and image
        directory.

        Args:
            json_file (str): Path to the JSON file containing image annotations.
            img_dir (str): Directory containing the images referenced in the
            JSON file.

        Returns:
            tuple: A tuple containing two numpy arrays:
            - images (numpy.ndarray): An array of normalized images of shape
                (num_images, 128, 128, 3).
            - masks (numpy.ndarray): An array of masks corresponding to the
                images, where each mask is of shape (128, 128)
                and contains binary values (0 or 1) indicating the absence or
                presence of an object.
        """
        df = pd.concat(pd.read_json(json_file, lines=True))
        images = []
        masks = []

        def process_row(row):
            img_path = os.path.join(img_dir, row["image_path"])
            img = Image.open(img_path).resize((128, 128))
            img_array = np.array(img) / 255.0  # Normalize
            images.append(img_array)

            mask = np.zeros((img.height, img.width))
            for box in row["boxes"]:
                x, y, i, j = box["x"], box["y"], box["width"], box["height"]
                mask[y : y + j, x : x + i] = 1
            masks.append(mask)

        df.apply(process_row, axis=1)

        return np.array(images), np.array(masks)

class DatasetPreprocessor:
    """
    A class to preprocess various datasets for object detection tasks.
    """

    def prep_ds1(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the first dataset by converting bounding box coordinates
        to center coordinates and width-height format. Also normalizes the
        coordinates and maps class labels to class IDs.

        Args:
            df (pd.DataFrame): The input dataframe with columns ['xmin',
            'xmax', 'ymin', 'ymax', 'class'].

        Returns:
            pd.DataFrame: A dataframe with columns ['filename', 'x_center',
            'y_center', 'width', 'height', 'class_id'].
        """
        # Convert to center coordinates and width-height format
        df['x_center'] = (df['xmin'] + df['xmax']) / 2
        df['y_center'] = (df['ymin'] + df['ymax']) / 2
        df['width'] = df['xmax'] - df['xmin']
        df['height'] = df['ymax'] - df['ymin']

        # Normalize the coordinates
        df['x_center'] /= df['width']
        df['y_center'] /= df['height']
        df['width'] /= df['width']
        df['height'] /= df['height']

        # Convert class label to class ID
        df['class_id'] = df['class'].map({'pothole': 0})  # Assuming only 'pothole' class for now

        return df[[
            'filename',
            'x_center',
            'y_center',
            'width',
            'height',
            'class_id'
        ]]

    def prep_ds2(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the second dataset. This dataset is assumed to be already
        in the desired format.

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: The same input dataframe.
        """
        return df

    def prep2(self, data: dict, img_width: int, img_height: int) -> pd.DataFrame:
        """
        Preprocess the third and fourth datasets by converting bounding box
        coordinates to center coordinates and width-height format. Also normalizes
        the coordinates and sets class ID for 'pothole' as 0.

        Args:
            data (dict): The input data in dictionary format with keys ['annotations'].
            img_width (int): The width of the image.
            img_height (int): The height of the image.

        Returns:
            pd.DataFrame: A dataframe with columns ['filename', 'x_center', 'y_center', 'width', 'height', 'class_id'].
        """
        processed_data = []
        for annotation in data['annotations']:
            for box in annotation['boxes']:
                x_center = box['x'] + box['width'] / 2
                y_center = box['y'] + box['height'] / 2

                # Normalize the coordinates
                x_center /= img_width
                y_center /= img_height
                width = box['width'] / img_width
                height = box['height'] / img_height

                # Assuming class ID is 0 for 'pothole'
                processed_data.append({
                    'filename': annotation['image_name'],
                    'x_center': x_center,
                    'y_center': y_center,
                    'width': width,
                    'height': height,
                    'class_id': 0
                })
        return pd.DataFrame(processed_data)
