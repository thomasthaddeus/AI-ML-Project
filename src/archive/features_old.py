"""build_features.py
_summary_

_extended_summary_

Returns:
    _type_: _description_
"""

import cv2
import numpy as np

def extract_color_histogram(image, bins=(8, 8, 8)):
    """
    Extract color histogram from an image.

    Args:
        image (numpy.array): Input image.
        bins (tuple): Number of bins for histogram in each channel.

    Returns:
        numpy.array: Flattened color histogram.
    """
    # Compute histogram for each channel (R, G, B)
    hist_r = cv2.calcHist([image], [0], None, [bins[0]], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [bins[1]], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [bins[2]], [0, 256])

    # Normalize histograms
    hist_r /= hist_r.sum()
    hist_g /= hist_g.sum()
    hist_b /= hist_b.sum()

    # Concatenate histograms into a single feature vector
    hist = np.concatenate([hist_r, hist_g, hist_b]).flatten()

    return hist

def extract_features(images):
    """
    Extract features from a list of images.

    Args:
        images (list of numpy.array): List of images.

    Returns:
        numpy.array: Extracted features for each image.
    """
    feature_list = [extract_color_histogram(image) for image in images]
    return np.array(feature_list)
