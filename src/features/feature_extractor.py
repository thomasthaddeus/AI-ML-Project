"""feature_extractor.py

This module provides utilities to extract various image features such as Local
Binary Pattern (LBP), Canny edge detection, color histogram, Histogram of
Oriented Gradients (HOG), and basic image statistics.

Returns:
    numpy.ndarray: A concatenated array of all extracted features.

# Example usage:
# image = cv2.imread('path_to_image.jpg')
# extractor = FeatureExtractor(image)
# features = extractor.extract_all()
"""

from skimage.feature import local_binary_pattern, hog
import cv2
import numpy as np


class FeatureExtractor:
    """
    A class to extract various features from an image.

    This class provides methods to extract features like LBP, Canny edges,
    color histogram, HOG, and basic image statistics from a given image. It
    also provides a method to extract all features and concatenate them into a
    single vector.
    """
    def __init__(self, image):
        self.image = cv2.cvtColorTwoPlane(
            image, cv2.COLOR_BGR2GRAY
        )  # Convert to grayscale for some features


    def lbp(self, P=8, R=1):
        """
        Compute Local Binary Pattern (LBP) for the image.

        LBP is a simple yet efficient texture operator which labels the pixels
        of an image by thresholding the neighborhood of each pixel and
        considers the result as a binary number.

        Args:
            P (int, optional): Number of circularly symmetric neighbor set
                points. Defaults to 8.
            R (int, optional): Radius of circle. Defaults to 1.

        Returns:
            numpy.ndarray: LBP image.
        """
        return local_binary_pattern(self.image, P=P, R=R, method="uniform")


    def canny_edge(self, lower_threshold=100, upper_threshold=200):
        """
        Compute Canny edge detection for the image.

        The Canny edge detection operator was developed by John F. Canny in
        1986 and uses a multi-stage algorithm to detect a wide range of edges
        in images.

        Args:
            lower_threshold (int, optional): First threshold for the hysteresis
                procedure. Defaults to 100.
            upper_threshold (int, optional): Second threshold for the
                hysteresis procedure. Defaults to 200.

        Returns:
            numpy.ndarray: Binary image of edges.
        """
        return cv2.Canny(self.image, lower_threshold, upper_threshold)


    def color_histogram(self, bins=8):
        """
        Compute color histogram for the grayscale image.

        A histogram represents the distribution of pixel intensities in an
        image.

        Args:
            bins (int, optional): Number of bins for the histogram. Default is 8

        Returns:
            numpy.ndarray: Flattened histogram array.
        """
        hist = cv2.calcHist([self.image], [0], None, [bins], [0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()


    def hog_features(self, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        """
        Compute Histogram of Oriented Gradients (HOG) for the image.

        HOG is a feature descriptor used in object detection.

        Args:
            pixels_per_cell (tuple, optional): Size (in pixels) of a cell. Defaults to (8, 8).
            cells_per_block (tuple, optional): Number of cells in each block. Defaults to (2, 2).

        Returns:
            numpy.ndarray: HOG feature vector.
        """
        return hog(
            self.image,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=False,
        )


    def image_statistics(self):
        """
        Compute basic image statistics: mean, median, standard deviation.

        These statistics provide basic information about the pixel intensity
        distribution in the image.

        Returns:
            tuple: Mean, median, and standard deviation of the image.
        """
        mean = np.mean(self.image)
        median = np.median(self.image)
        std = np.std(self.image)
        return mean, median, std


    def extract_all(self):
        """
        Extract all features and concatenate them into a single vector.

        This method extracts LBP, Canny edges, color histogram, HOG, and basic
        image statistics and concatenates them to form a single feature vector.

        Returns:
            numpy.ndarray: Concatenated feature vector.
        """
        lbp_hist = np.histogram(self.lbp(), bins=8, range=(0, 256))[0]
        canny_edges = self.canny_edge().flatten()
        color_hist = self.color_histogram()
        hog_feat = self.hog_features()
        stats = self.image_statistics()

        # Concatenate all features into a single vector
        return np.concatenate([lbp_hist, canny_edges, color_hist, hog_feat, stats])
