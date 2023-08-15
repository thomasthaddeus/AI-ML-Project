"""visualize.py

Basic visualization class using `seaborn` and `matplotlib` to visualize the
performance of a neural network image classifier for potholes.

This class provides methods to:

1. Plot the training and validation accuracy and loss over epochs.
2. Visualize the confusion matrix.
3. Display a set of images with their predicted and true labels.

Usage example:
# visualizer = NeuralNetworkVisualizer()
# visualizer.plot_training_history(history)
# visualizer.plot_confusion_matrix(cm, ["Not Pothole", "Pothole"])
# visualizer.visualize_predictions(
    sample_img, sample_preds, sample_true_labels, ["Not Pothole", "Pothole"]
)
"""

import numpy as np
from keras.utils import plot_model
from keras.models import Model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



class NeuralNetworkVisualizer:
    """
    A class to visualize the performance of a neural network image classifier.

    This class provides utility functions to visualize training history,
    confusion matrix, and predictions of a neural network model trained for
    image classification.
    """

    def __init__(self):
        """
        Initializes the NeuralNetworkVisualizer class and sets the seaborn
        style.

        The seaborn style is set to "whitegrid" for better visualization.
        """
        sns.set_style("whitegrid")

    def plot_training_history(self, history):
        """
        Plots the training and validation accuracy and loss over epochs.

        This method visualizes the accuracy and loss of the model over each
        epoch for both training and validation datasets.

        Args:
            history (History): A history object obtained from the training of a
            neural network model.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot training & validation accuracy values
        axes[0].plot(history.history["accuracy"])
        axes[0].plot(history.history["val_accuracy"])
        axes[0].set_title("Model Accuracy")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].legend(["Train", "Validation"], loc="upper left")

        # Plot training & validation loss values
        axes[1].plot(history.history["loss"])
        axes[1].plot(history.history["val_loss"])
        axes[1].set_title("Model Loss")
        axes[1].set_ylabel("Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend(["Train", "Validation"], loc="upper left")

        plt.show()

    def plot_confusion_matrix(self, cm, class_names):
        """
        Plots the confusion matrix using seaborn.

        This method visualizes the confusion matrix which shows the actual vs
        predicted classifications.

        Args:
            cm (numpy.ndarray): The confusion matrix.
            class_names (list): List of class names for labeling the matrix.
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=sns.diverging_palette(220, 20, as_cmap=True),
            xticklabels=class_names,
            yticklabels=class_names,
        )
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.title("Confusion Matrix")
        plt.show()

    def visualize_predictions(self, images, predictions, true_labels, class_names):
        """
        Visualizes a set of images with their predicted and true labels.

        This method displays a set of images side by side with their predicted and true labels.

        Args:
            images (numpy.ndarray): Array of images to be displayed.
            predictions (numpy.ndarray): Array of predicted labels for the images.
            true_labels (numpy.ndarray): Array of true labels for the images.
            class_names (list): List of class names for labeling the predictions.
        """
        num_images = len(images)
        fig, axes = plt.subplots(1, num_images, figsize=(20, 5))

        for i, ax in enumerate(axes):
            ax.imshow(images[i])
            ax.set_title(
                f"Pred: {class_names[np.argmax(predictions[i])]}, True: {class_names[true_labels[i]]}"
            )
            ax.axis("off")

        plt.show()

    def plot_roc_curve(self, y_true, y_pred_prob):
        """
        Plot the ROC curve for binary or multiclass classification.

        Args:
            y_true (numpy.ndarray): True labels.
            y_pred_prob (numpy.ndarray): Predicted probabilities.
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc(fpr, tpr):.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()

    def calculate_auc(self, y_true, y_pred_prob):
        """
        Calculate and return the AUC value.

        Args:
            y_true (numpy.ndarray): True labels.
            y_pred_prob (numpy.ndarray): Predicted probabilities.
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        return auc(fpr, tpr)

    def plot_precision_recall_curve(self, y_true, y_pred_prob):
        """
        Plot the precision-recall curve.

        Args:
            y_true (numpy.ndarray): True labels.
            y_pred_prob (numpy.ndarray): Predicted probabilities.
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
        average_precision = average_precision_score(y_true, y_pred_prob)
        plt.figure(figsize=(10, 7))
        plt.plot(recall, precision, color='b', lw=2, label=f'Precision-Recall curve (area = {average_precision:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="upper right")
        plt.show()

    def plot_feature_importance(self, feature_importances, feature_names):
        """
        Visualize the importance of each feature.

        Args:
            feature_importances (numpy.ndarray): Importance values for each feature.
            feature_names (list): Names of the features.
        """
        sorted_idx = feature_importances.argsort()
        plt.figure(figsize=(10, len(feature_names) // 2))
        plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.show()

    def plot_probability_histogram(self, y_pred_prob, bins=50):
        """
        Display a histogram of the predicted probabilities.

        Args:
            y_pred_prob (numpy.ndarray): Predicted probabilities.
            bins (int): Number of histogram bins.
        """
        plt.figure(figsize=(10, 7))
        plt.hist(y_pred_prob, bins=bins, edgecolor='k', alpha=0.7)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency')
        plt.title('Histogram of Predicted Probabilities')
        plt.show()

    def plot_learning_rate_schedule(self, learning_rates, epochs):
        """
        Visualize the learning rate over epochs.

        Args:
            learning_rates (list): Learning rates for each epoch.
            epochs (list): Epoch numbers.
        """
        plt.figure(figsize=(10, 7))
        plt.plot(epochs, learning_rates, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.show()

    def visualize_embeddings(self, embeddings, method='tsne'):
        """
        Display embeddings using techniques like t-SNE or PCA.

        Args:
            embeddings (numpy.ndarray): The embeddings to visualize.
            method (str): The method to use for visualization ('tsne' or 'pca').
        """
        if method == 'tsne':
            reduced = TSNE(n_components=2).fit_transform(embeddings)
        elif method == 'pca':
            reduced = PCA(n_components=2).fit_transform(embeddings)
        else:
            raise ValueError("Method should be 'tsne' or 'pca'.")

        plt.figure(figsize=(10, 7))
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(f'{method.upper()} Visualization of Embeddings')
        plt.show()

    def plot_model_architecture(self, model, filename='model_architecture.png'):
        """
        Display the architecture of the neural network.

        Args:
            model (keras.Model): The neural network model.
            filename (str): The filename to save the visualization.
        """
        plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)
        plt.imshow(plt.imread(filename))
        plt.axis('off')
        plt.show()

    def visualize_activation_maps(self, model, layer_name, input_image):
        """
        For convolutional neural networks, visualize the activation maps.

        Args:
            model (keras.Model): The neural network model.
            layer_name (str): The name of the layer for which to visualize the activations.
            input_image (numpy.ndarray): The input image for the model.
        """
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        activations = intermediate_layer_model.predict(np.expand_dims(input_image, axis=0))

        num_activations = activations.shape[-1]
        plt.figure(figsize=(15, 15))
        for i in range(num_activations):
            plt.subplot(6, 6, i + 1)
            plt.imshow(activations[0, :, :, i], cmap='viridis')
            plt.axis('off')
        plt.tight_layout()
        plt.show()


    def plot_learning_rate_schedule(self, learning_rates, epochs):
        """
        Visualize the learning rate over epochs.

        Args:
            learning_rates (list): Learning rates for each epoch.
            epochs (list): Epoch numbers.
        """
        plt.figure(figsize=(10, 7))
        plt.plot(epochs, learning_rates, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.show()

    def visualize_embeddings(self, embeddings, method='tsne'):
        """
        Display embeddings using techniques like t-SNE or PCA.

        Args:
            embeddings (numpy.ndarray): The embeddings to visualize.
            method (str): The method to use for visualization ('tsne' or 'pca').
        """
        if method == 'tsne':
            reduced = TSNE(n_components=2).fit_transform(embeddings)
        elif method == 'pca':
            reduced = PCA(n_components=2).fit_transform(embeddings)
        else:
            raise ValueError("Method should be 'tsne' or 'pca'.")

        plt.figure(figsize=(10, 7))
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.title(f'{method.upper()} Visualization of Embeddings')
        plt.show()

    def plot_model_architecture(self, model, filename='model_architecture.png'):
        """
        Display the architecture of the neural network.

        Args:
            model (keras.Model): The neural network model.
            filename (str): The filename to save the visualization.
        """
        plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)
        plt.imshow(plt.imread(filename))
        plt.axis('off')
        plt.show()

    def visualize_activation_maps(self, model, layer_name, input_image):
        """
        For convolutional neural networks, visualize the activation maps.

        Args:
            model (keras.Model): The neural network model.
            layer_name (str): The name of the layer for which to visualize the activations.
            input_image (numpy.ndarray): The input image for the model.
        """
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
        activations = intermediate_layer_model.predict(np.expand_dims(input_image, axis=0))

        num_activations = activations.shape[-1]
        plt.figure(figsize=(15, 15))
        for i in range(num_activations):
            plt.subplot(6, 6, i + 1)
            plt.imshow(activations[0, :, :, i], cmap='viridis')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
