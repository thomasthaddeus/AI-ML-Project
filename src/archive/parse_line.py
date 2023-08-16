"""parse_line.py
This module provides utilities to parse bounding box annotations from dataset files.

Given a dataset file where each line represents an image and its associated
bounding box annotations, the module provides functions to extract the
annotations, adjust the image file extensions, and create a structured Pandas
DataFrame. This DataFrame can then be exported to a JSON format for further
processing or analysis.

Returns:
    DataFrame: A Pandas DataFrame containing the parsed annotations mapped to
    their respective images.
"""

import pandas as pd

DS3_TRN = "./data/processed/ds3_trn.txt"
DS3_TST = "./data/processed/ds3_tst.txt"
DS4_TRN = "./data/processed/ds4_trn.txt"
DS4_TST = "./data/processed/ds4_tst.txt"


def parse_line(line):
    """
    Parse a single line from the dataset to extract image path and bounding
    boxes.

    Given a line from the dataset file, this function extracts the image path,
    adjusts its file extension from .bmp to .jpg, and extracts the bounding box
    annotations associated with the image.

    Args:
        line (str): A line from the dataset file.

    Returns:
        dict: A dictionary containing the image path and its associated
        bounding box annotations.
    """
    parts = line.strip().split()
    image_path = parts[0].replace(".bmp", ".jpg")  # Adjust the file extension
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


def read_dataset(dataset):
    """
    Read the dataset file and return a DataFrame with image paths and
    annotations.

    This function reads the specified dataset file, parses each line using the
    parse_line function, and aggregates the parsed data into a structured
    Pandas DataFrame.

    Args:
        dataset (str): Path to the dataset file.

    Returns:
        DataFrame: A Pandas DataFrame containing the image paths and their
        associated bounding box annotations.
    """
    with open(file=dataset, mode="r", encoding="utf-8") as f:  # pylint: disable=invalid-name
        lines = f.readlines()
    data = [parse_line(line) for line in lines]
    return pd.DataFrame(data)


def process_all_datasets():
    """
    Process all dataset constants and create annotation sets for each.

    This function iterates through each of the dataset constants, reads the
    dataset, creates a DataFrame of annotations, and then exports the DataFrame
    to a JSON file.
    """
    datasets = [DS3_TRN, DS3_TST, DS4_TRN, DS4_TST]
    for dataset in datasets:
        df = read_dataset(dataset)  #pylint: disable=invalid-name
        json_filename = dataset.replace(".txt", ".json").replace(
            "/processed/", "/processed/json/"
        )
        df.to_json(json_filename, orient="records", lines=True)
        print(f"Processed {dataset} and saved to {json_filename}")


# Call the function to process all datasets
process_all_datasets()
