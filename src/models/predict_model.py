"""predict_model.py

This module provides utilities for validating and making predictions using a
trained convolutional neural network (CNN) for binary image classification
tasks, specifically for detecting potholes in images.

The module defines functions for validating the model's performance on test
data and making predictions on new, unseen data.

Returns:
    float: The accuracy of the model on the test data.
"""

import numpy as np


def validate_model(model, X_test, y_test):
    """
    Validate the performance of a trained model on test data.

    This function evaluates the provided model using the test data and prints
    the accuracy of the model.

    Args:
        model (keras.models.Model): The trained CNN model.
        X_test (numpy.ndarray): Test images.
        y_test (numpy.ndarray): True labels for the test images.

    Returns:
        float: The accuracy of the model on the test data.
    """
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Loss: {loss:.4f}")
    return accuracy


def make_predictions(model, X_new):
    """
    Make predictions on new, unseen data using the trained model.

    This function uses the provided model to make predictions on a batch of new
    images. It returns the predicted labels for each image.

    Args:
        model (keras.models.Model): The trained CNN model.
        X_new (numpy.ndarray): New images for which predictions are to be made.

    Returns:
        numpy.ndarray: Predicted labels for the new images.
    """
    predictions = model.predict(X_new)
    predicted_labels = np.where(predictions > 0.5, 1, 0).flatten()
    return predicted_labels
