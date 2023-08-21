"""prep2.py
This module provides a DatasetPreprocessor class to preprocess various datasets
for object detection tasks. The class includes methods to handle different
formats of datasets and convert them into a standardized format suitable for
object detection models. The standardized format includes center coordinates,
width-height format for bounding boxes, normalized coordinates, and mapped
class labels to class IDs.
"""

import pandas as pd

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

    def preprocess_dataset3_4(self, data: dict, img_width: int, img_height: int) -> pd.DataFrame:
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
