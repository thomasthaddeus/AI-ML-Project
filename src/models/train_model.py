"""train_model.py

This module provides utilities for designing, compiling, and training a
convolutional neural network (CNN) for binary image classification tasks,
specifically for detecting potholes in images.

The module defines functions for designing the CNN architecture, specifying the
optimizer, and building and training the model using provided training and
validation data.

Returns:
    keras.models.Model: A trained Keras model for binary image classification.
"""

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam

def design_model():
    """
    Design a convolutional neural network (CNN) for binary image classification.

    This function defines a CNN architecture using Keras Sequential API. The
    model consists of convolutional layers, max-pooling layers, a flatten
    layer, and dense layers. The final layer uses a sigmoid activation function
    for binary classification.

    Returns:
        keras.models.Model: A Keras model with the defined CNN architecture.
    """
    model = Sequential([
        Conv2D(
            32,
            (3, 3),
            activation='relu',
            input_shape=(128, 128, 3)
        ),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    return model

def my_optimizer():
    """
    Define the Adam optimizer for training the model.

    This function specifies the Adam optimizer with a learning rate of 0.0001.

    Returns:
        keras.optimizers.Adam: Adam optimizer with the specified learning rate.
    """
    return Adam(lr=0.0001)

def build_model(model, X_train, y_train, X_val, y_val): # pylint: disable=C0103
    """
    Compile and train the provided CNN model using the training and validation
    data.

    This function compiles the provided model using binary cross-entropy loss
    and the Adam optimizer. It then trains the model using the provided
    training data and validates it using the validation data.

    Args:
        model (keras.models.Model): The CNN model to be trained.
        X_train (numpy.ndarray): Training images.
        y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray): Validation images.
        y_val (numpy.ndarray): Validation labels.

    Returns:
        tuple: A trained Keras model and its training history.
    """
    model.compile(
        optimizer=my_optimizer(),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=10,
        validation_data=(X_val, y_val)
    )
    return model, history
