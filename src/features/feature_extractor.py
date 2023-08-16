"""

_extended_summary_

Returns:
    _type_: _description_


# Example usage:
# image = cv2.imread('path_to_image.jpg')
# extractor = FeatureExtractor(image)
# features = extractor.extract_all()
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern, hog


class FeatureExtractor:
    def __init__(self, image):
        self.image = cv2.cvtColor(
            image, cv2.COLOR_BGR2GRAY
        )  # Convert to grayscale for some features

    def lbp(self, P=8, R=1):
        """Compute Local Binary Pattern."""
        return local_binary_pattern(self.image, P=P, R=R, method="uniform")

    def canny_edge(self, lower_threshold=100, upper_threshold=200):
        """Compute Canny edge detection."""
        return cv2.Canny(self.image, lower_threshold, upper_threshold)

    def color_histogram(self, bins=8):
        """Compute color histogram."""
        hist = cv2.calcHist([self.image], [0], None, [bins], [0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def hog_features(self, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        """Compute Histogram of Oriented Gradients."""
        return hog(
            self.image,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            visualize=False,
        )

    def image_statistics(self):
        """Compute basic image statistics: mean, median, standard deviation."""
        mean = np.mean(self.image)
        median = np.median(self.image)
        std = np.std(self.image)
        return mean, median, std

    def extract_all(self):
        """Extract all features and concatenate them."""
        lbp_hist = np.histogram(self.lbp(), bins=8, range=(0, 256))[0]
        canny_edges = self.canny_edge().flatten()
        color_hist = self.color_histogram()
        hog_feat = self.hog_features()
        stats = self.image_statistics()

        # Concatenate all features into a single vector
        return np.concatenate([lbp_hist, canny_edges, color_hist, hog_feat, stats])
