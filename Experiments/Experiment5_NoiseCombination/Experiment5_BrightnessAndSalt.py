#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 01:11:50 2023

@author: ericwei
"""


"""
==================================================================================
NOTE: Due to similar function objectives across various experiments, some functions 
within this script may not be accompanied by an individual detailed introduction. 
For specific usage notes and detailed descriptions of such functions, it's advisable 
to refer to previous experiments where the functions were initially introduced and described.
==================================================================================
"""

from matplotlib.lines import Line2D
import os

# Third-party Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, MobileNet, DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
import cv2


# Matplotlib for color mappings
from matplotlib.colors import ListedColormap


def generate_brightness_variation(image, variation_intensity):
    # Convert the image to the HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Apply brightness variation to the V channel
    image_hsv[:,:,2] = image_hsv[:,:,2] * variation_intensity

    # Convert back to the RGB color space
    brightened_image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    brightened_image = np.clip(brightened_image, 0, 1)  # Clip values to be in the range [0, 1]
    return brightened_image



def generate_original_images(data_dir, categories):
    original_images = []
    labels = []

    for category in categories:
        if category == '.DS_Store':
            continue
        image_files = os.listdir(os.path.join(data_dir, category))
        for image_file in image_files:
            image = Image.open(os.path.join(data_dir, category, image_file))
            image = image.convert('RGB')
            image = image.resize((224, 224))
            original_image_array = img_to_array(image) / 255.0
            original_images.append(original_image_array)
            labels.append(category)

    return original_images, labels



def generate_brightness_variation_images(original_images, labels, variation_intensity, num_iterations):
    brightness_variation_images = []

    for original_image in original_images:
        brightened_image = original_image
        for _ in range(num_iterations):
            brightened_image = generate_brightness_variation(brightened_image, variation_intensity)
        brightness_variation_images.append(brightened_image)

    return original_images, brightness_variation_images, labels


def generate_salt_and_pepper_noise(image, salt_prob, pepper_prob):
    # Create a copy of the original image
    noisy_image = np.copy(image)

    # Generate a random matrix with the same shape as the image
    random_noise = np.random.rand(*image.shape[:2])

    # Create a noise pattern matrix initialized with zeros
    noise_pattern = np.zeros_like(image)

    # Apply salt noise: Set the pixels where the random noise is less than the salt probability to 1 (white)
    salt_mask = random_noise < salt_prob
    noisy_image[salt_mask] = 1
    noise_pattern[salt_mask] = 1

    # Apply pepper noise: Set the pixels where the random noise is less than the salt probability plus pepper probability to 0 (black)
    pepper_mask = (random_noise >= salt_prob) & (random_noise < salt_prob + pepper_prob)
    noisy_image[pepper_mask] = 0
    noise_pattern[pepper_mask] = -1

    return noisy_image, noise_pattern


def generate_noisy_images(color_varied_images, labels, salt_prob, pepper_prob):
    noisy_images = []
    noise_patterns = []

    for image_array in color_varied_images:
        noisy_image_array, noise_pattern = generate_salt_and_pepper_noise(image_array, salt_prob, pepper_prob)

        noisy_images.append(noisy_image_array)
        noise_patterns.append(noise_pattern)

    return noisy_images, noise_patterns, labels



def generate_gradual_brightness_variation(data_dir, categories, variation_intensity, salt_prob, pepper_prob, num_iterations):
    # Load an example image
    image_path = os.path.join(data_dir, categories[12], os.listdir(os.path.join(data_dir, categories[12]))[12])
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.resize((224, 224))
    original_image_array = img_to_array(image) / 255.0

    images = [original_image_array]
    noise_samples = []
    noisy_images = [original_image_array]
    for i in range(num_iterations):
        brightened_image = generate_brightness_variation(images[-1], variation_intensity)

        # Add Gaussian noise to the brightened image
        noisy_image_array, gaussian_noise = generate_salt_and_pepper_noise(brightened_image, salt_prob, pepper_prob)

        # Calculate noise sample (difference between noisy and original)
        noise_sample = noisy_image_array - images[-1]
        noise_sample = (noise_sample - noise_sample.min()) / (noise_sample.max() - noise_sample.min()) # Normalize to [0, 1]

        # Append to the lists
        images.append(brightened_image)
        noise_samples.append(noise_sample)
        noisy_images.append(noisy_image_array)

    return images, noise_samples, noisy_images


def plot_gradual_brightness_variation(images, noise_samples, noisy_images):
    # Plot Brightness Variation Images
    plt.figure(figsize=(20, 5))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.title(f'Brightness Variation (T = {i})')
    plt.show()

    # Plot Noisy Images
    plt.figure(figsize=(20, 5))
    for i in range(len(noisy_images)):
        plt.subplot(1, len(noisy_images), i + 1)
        plt.imshow(noisy_images[i])
        plt.title(f'Noisy Image (T = {i})')
    plt.show()

    # Plot Noise Samples
    plt.figure(figsize=(20, 5))
    for i in range(len(noise_samples)):
        plt.subplot(1, len(noise_samples), i + 1)
        plt.imshow(noise_samples[i], cmap='gray')
        plt.title(f'Noise Sample (T = {i+1})')
    plt.show()



def calculate_average_psnr(original_images, decompressed_images):
    psnrs = []
    for orig, dec in zip(original_images, decompressed_images):
        orig = orig.astype('float32') / 255.
        dec = dec.astype('float32') / 255.
        psnr = peak_signal_noise_ratio(orig, dec, data_range=1.0)
        psnrs.append(psnr)
    return np.mean(psnrs)



def image_preprocessing(categories, original_images, brightness_varied_images, noisy_images, labels):

    # Calculate and print the average PSNR for brightness varied images
    avg_psnr_brightness = calculate_average_psnr(original_images, brightness_varied_images)
    print(f"Average PSNR for brightness varied image: {avg_psnr_brightness} dB")

    # Calculate and print the average PSNR for noisy images
    avg_psnr_noisy = calculate_average_psnr(original_images, noisy_images)
    print(f"Average PSNR for noisy image: {avg_psnr_noisy} dB")

    images = np.array(noisy_images)
    labels = np.array(labels)
    le = LabelEncoder()
    labels_int = le.fit_transform(labels)
    class_names = le.classes_
    print(class_names)
    num_classes = len(categories)
    labels = to_categorical(labels_int, num_classes)
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25, random_state=42)

    return train_images, val_images, test_images, train_labels, val_labels, num_classes, class_names, avg_psnr_noisy, avg_psnr_brightness



def model_construction(model_name, num_classes, trainable):
    """
    Function to construct a chosen model and add a fully connected layer and a logistic layer to it.
    """
    # Choose the base model according to the model_name
    if model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False)
    elif model_name == 'VGG19':
        base_model = VGG19(weights='imagenet', include_top=False)
    elif model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False)
    elif model_name == 'MobileNet':
        base_model = MobileNet(weights='imagenet', include_top=False)
    elif model_name == 'DenseNet121':
        base_model = DenseNet121(weights='imagenet', include_top=False)
    else:
        print("Invalid model name. Choose from 'VGG16', 'VGG19', 'ResNet50', 'MobileNet', 'DenseNet121'.")
        return None

    # Add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Add a fully-connected layer
    x = Dense(1024, activation='relu')(x)

    # Add a logistic layer with the number of classes in the dataset
    predictions = Dense(num_classes, activation='softmax')(x)

    # Construct the full model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = trainable

    # Add Top 5 metrics
    top5_acc = TopKCategoricalAccuracy(k=5, name='top_5_accuracy')
    # Compile the model
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy', top5_acc])

    return model




def train_plotting(model, train_images, train_labels, val_images, val_labels):
    """
    Function to train the model and plot the accuracy and loss.
    """
    # Train the model
    history = model.fit(train_images, train_labels, batch_size=32, epochs=20, validation_data=(val_images, val_labels))

    # Prepare the epoch index
    epochs = range(1, len(history.history['accuracy']) + 1)

    # Plot the training and validation accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.plot(epochs, history.history['accuracy'], linewidth = 2)
    plt.plot(epochs, history.history['val_accuracy'], linewidth = 2)
    plt.title('Model accuracy', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.xticks(ticks=range(1,21))
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True)

    # Plot the training and validation loss
    plt.subplot(122)
    plt.plot(epochs, history.history['loss'])
    plt.plot(epochs, history.history['val_loss'])
    plt.title('Model loss', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.xticks(ticks=range(1,21))
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return history



def plot_top5_accuracy(history):
    """
    Function to plot the top 5 accuracy for training and validation data.
    It also returns the epoch number with highest validation top 5 accuracy.
    """
    # Prepare the epoch index
    epochs = range(1, len(history.history['accuracy']) + 1)

    # Plot the training and validation top 5 accuracy
    plt.figure(figsize=(5, 5))
    plt.plot(epochs, history.history['top_5_accuracy'], 'bo-', label='Training top 5 acc')
    plt.plot(epochs, history.history['val_top_5_accuracy'], 'rs-', label='Validation top 5 acc')
    plt.title('Training and validation top 5 accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12, fontweight='bold')
    plt.ylabel('Top 5 Accuracy', fontsize=12, fontweight='bold')
    plt.xticks(ticks=range(1,21))
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Get the epoch with the highest validation top 5 accuracy
    highest_val_acc_epoch = np.argmax(history.history['val_top_5_accuracy']) + 1
    highest_val_acc = np.max(history.history['val_top_5_accuracy'])

    return highest_val_acc, highest_val_acc_epoch



def visualize_feature_space(model, images, labels, class_names, resolution=100):
    # Get the output of the second-to-last layer of the model
    feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    features = feature_model.predict(images)

    # Use t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    # Use argmax to convert one-hot encoded labels back to integers
    labels_int = np.argmax(labels, axis=1)

    # Predict the labels for the images
    predicted_labels = model.predict(images)
    predicted_labels_int = np.argmax(predicted_labels, axis=1)

    # Create a scatter plot where points are colored according to their true labels
    plt.figure(figsize=(10, 10))
    for i in range(len(class_names)):
        plt.scatter(features_2d[labels_int == i, 0], features_2d[labels_int == i, 1], label=class_names[i], alpha=0.6)
    plt.legend(title="True Classes")
    plt.title("True labels")
    plt.show()

    # Create a scatter plot where points are colored according to their predicted labels
    plt.figure(figsize=(10, 10))
    for i in range(len(class_names)):
        plt.scatter(features_2d[predicted_labels_int == i, 0], features_2d[predicted_labels_int == i, 1], label=class_names[i], alpha=0.6)
    plt.legend(title="Predicted Classes")
    plt.title("Predicted labels")
    plt.show()

    # Create a scatter plot where points are colored according to whether they were correctly classified
    incorrectly_classified = (labels_int != predicted_labels_int)
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=incorrectly_classified, cmap=ListedColormap(['g', 'r']), alpha=0.6)
    plt.legend(handles=scatter.legend_elements()[0], labels=["Correct", "Incorrect"], title="Classification")
    plt.title("Classification results")
    plt.show()

    # Count misclassified instances for each class
    misclassified_counts = {}
    for i in range(len(class_names)):
        misclassified_counts[class_names[i]] = sum((labels_int == i) & (predicted_labels_int != i))

    total_misclassified = sum(misclassified_counts.values())

    return misclassified_counts, total_misclassified


def visualize_decision_boundary_knn(model, images, labels, class_names, resolution=500):
    # Get the output of the second-to-last layer of the model
    feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    features = feature_model.predict(images)

    # Use t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    # Use argmax to convert one-hot encoded labels back to integers
    labels_int = np.argmax(labels, axis=1)

    classifier = KNeighborsClassifier(n_neighbors=3)
    classifier.fit(features_2d, labels_int)

    # Generate a grid of points
    x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
    y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min) / resolution),
                         np.arange(y_min, y_max, (y_max - y_min) / resolution))

    # Predict the class labels of the grid points
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Set figure size
    plt.figure(figsize=(10, 10))

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8)

    # Plot the original data points
    for i in range(len(class_names)):
        plt.scatter(features_2d[labels_int == i, 0], features_2d[labels_int == i, 1], label=class_names[i], edgecolors='k', alpha=0.6)

    plt.legend(title="Classes")
    plt.title("Decision Boundaries (KNN)")
    plt.show()



def visualize_decision_boundary_rf(model, images, labels, class_names, resolution=500):
    # Get the output of the second-to-last layer of the model
    feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    features = feature_model.predict(images)

    # Use t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    # Use argmax to convert one-hot encoded labels back to integers
    labels_int = np.argmax(labels, axis=1)

    classifier = RandomForestClassifier(n_estimators=1000)
    classifier.fit(features_2d, labels_int)

    # Generate a grid of points
    x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
    y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min) / resolution),
                         np.arange(y_min, y_max, (y_max - y_min) / resolution))

    # Predict the class labels of the grid points
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Set figure size
    plt.figure(figsize=(10, 10))

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8)

    # Plot the original data points
    for i in range(len(class_names)):
        plt.scatter(features_2d[labels_int == i, 0], features_2d[labels_int == i, 1], label=class_names[i], edgecolors='k', alpha=0.6)

    plt.legend(title="Classes")
    plt.title("Decision Boundaries (RF)")
    plt.show()



def visualize_decision_boundary_svm(model, images, labels, class_names, resolution=500):
    # Get the output of the second-to-last layer of the model
    feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    features = feature_model.predict(images)

    # Use t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    # Use argmax to convert one-hot encoded labels back to integers
    labels_int = np.argmax(labels, axis=1)

    classifier = SVC(kernel='rbf')
    classifier.fit(features_2d, labels_int)

    # Generate a grid of points
    x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
    y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min) / resolution),
                         np.arange(y_min, y_max, (y_max - y_min) / resolution))

    # Predict the class labels of the grid points
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Set figure size
    plt.figure(figsize=(10, 10))

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8)

    # Plot the original data points
    for i in range(len(class_names)):
        plt.scatter(features_2d[labels_int == i, 0], features_2d[labels_int == i, 1], label=class_names[i], edgecolors='k', alpha=0.6)

    plt.legend(title="Classes")
    plt.title("Decision Boundaries (SVM)")
    plt.show()



def average_shift(model, original_images, decompressed_images):
    # Extract features
    feature_model = Model(inputs=model.input, outputs=model.layers[-2].output) # Assumes the last layer is the classification layer
    original_features = feature_model.predict(np.array(original_images))
    compressed_features = feature_model.predict(np.array(decompressed_images))

    # Compute Euclidean distance between original and compressed features
    distances = np.sqrt(np.sum((original_features - compressed_features)**2, axis=1))
    # Return average distance
    return np.mean(distances)



def visualize_decision_boundary_svm_shifted_topn(categories, model, images, labels, class_names, original_points, noisy_points, compressed_points, n_pairs=5, resolution=500, padding=12.0):
    # Get the output of the second-to-last layer of the model
    feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)

    # Compute features for all images
    features = feature_model.predict(images)

    # Compute features for the original, noisy, and compressed points
    original_features = [feature_model.predict(np.expand_dims(point, axis=0)) for point in original_points]
    noisy_features = [feature_model.predict(np.expand_dims(point, axis=0)) for point in noisy_points]
    compressed_features = [feature_model.predict(np.expand_dims(point, axis=0)) for point in compressed_points]

    # Combine all features for t-SNE
    all_features = np.concatenate([features] + original_features + noisy_features + compressed_features)

    # Use t-SNE to reduce dimensionality to 2D for all image features
    tsne = TSNE(n_components=2, random_state=42)
    all_features_2d = tsne.fit_transform(all_features)

    # The 2D coordinates for original_features, noisy_features, and compressed_features are the last 3n rows in features_2d
    features_2d = all_features_2d[:-3*len(original_points)]
    original_features_2d = all_features_2d[-3*len(original_points):-2*len(original_points)]
    noisy_features_2d = all_features_2d[-2*len(original_points):-len(original_points)]
    compressed_features_2d = all_features_2d[-len(original_points):]

    # Use argmax to convert one-hot encoded labels back to integers
    labels = np.array(labels)
    le = LabelEncoder()
    labels_int = le.fit_transform(labels)
    num_classes=len(categories)
    labels = to_categorical(labels_int, num_classes)
    labels_int = np.argmax(labels, axis=1)

    classifier = SVC(kernel='rbf')
    classifier.fit(features_2d, labels_int)  # We exclude the last three points (original, noisy, and compressed) when training the classifier

    # Set figure size
    plt.figure(figsize=(10, 10))

    # Generate a grid of points
    x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
    y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min) / resolution),
                         np.arange(y_min, y_max, (y_max - y_min) / resolution))

    # Predict the class labels of the grid points
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    for n in [1, n_pairs]:
        # If only 1 pair, find the most shifted one (i.e., the last one in the list)
        if n == 1:
            original_features_2d = [all_features_2d[-3*n_pairs:-2*n_pairs][-1]]
            noisy_features_2d = [all_features_2d[-2*n_pairs:-n_pairs][-1]]
            compressed_features_2d = [all_features_2d[-n_pairs:][-1]]
        else:
            # If more than 1 pair, order them in descending order
            original_features_2d = all_features_2d[-3*n_pairs:-2*n_pairs][::-1][:n]
            noisy_features_2d = all_features_2d[-2*n_pairs:-n_pairs][::-1][:n]
            compressed_features_2d = all_features_2d[-n_pairs:][::-1][:n]

        # Plot the decision boundary
        plt.figure(figsize=(10, 10))
        plt.contourf(xx, yy, Z, alpha=0.8)

        # Pre-define a list of colors for the arrows
        original_color = 'blue'
        noisy_color = 'green'
        compressed_color = 'red'
        arrow_colors = ['cyan', 'magenta', 'yellow', 'lime', 'brown', 'purple', 'orange', 'gray', 'pink', 'olive']

        legend_elements = []

        for i, (original_feature_2d, noisy_feature_2d, compressed_feature_2d) in enumerate(zip(original_features_2d, noisy_features_2d, compressed_features_2d)):
          original_class = class_names[classifier.predict(original_feature_2d.reshape(1, -1))[0]]
          noisy_class = class_names[classifier.predict(noisy_feature_2d.reshape(1, -1))[0]]
          compressed_class = class_names[classifier.predict(compressed_feature_2d.reshape(1, -1))[0]]

          plt.scatter(original_feature_2d[0], original_feature_2d[1], color=original_color, edgecolors='k', s=150)
          plt.scatter(noisy_feature_2d[0], noisy_feature_2d[1], color=noisy_color, edgecolors='k', s=150)
          plt.scatter(compressed_feature_2d[0], compressed_feature_2d[1], color=compressed_color, edgecolors='k', s=150)
          plt.arrow(original_feature_2d[0], original_feature_2d[1], noisy_feature_2d[0] - original_feature_2d[0], noisy_feature_2d[1] - original_feature_2d[1], color=arrow_colors[i], width=0.01)
          plt.arrow(noisy_feature_2d[0], noisy_feature_2d[1], compressed_feature_2d[0] - noisy_feature_2d[0], compressed_feature_2d[1] - noisy_feature_2d[1], color=arrow_colors[i+1], width=0.01)

          legend_elements.append(Line2D([0], [0], linestyle="none", marker='o', markersize=10, markerfacecolor=original_color, label=f'Original - {original_class}'))
          legend_elements.append(Line2D([0], [0], linestyle="none", marker='o', markersize=10, markerfacecolor=noisy_color, label=f'Brightness Variation - {noisy_class}'))
          legend_elements.append(Line2D([0], [0], linestyle="none", marker='o', markersize=10, markerfacecolor=compressed_color, label=f'Gaussian Noisy - {compressed_class}'))
          legend_elements.append(Line2D([0], [0], color=arrow_colors[i], lw=2, label=f'Arrow 1 - Pair {i+1}'))
          legend_elements.append(Line2D([0], [0], color=arrow_colors[i+1], lw=2, label=f'Arrow 2 - Pair {i+1}'))


        x_min_roi = min(min(pt[0] for pt in original_features_2d), min(pt[0] for pt in noisy_features_2d), min(pt[0] for pt in compressed_features_2d)) - padding
        x_max_roi = max(max(pt[0] for pt in original_features_2d), max(pt[0] for pt in noisy_features_2d), max(pt[0] for pt in compressed_features_2d)) + padding
        y_min_roi = min(min(pt[1] for pt in original_features_2d), min(pt[1] for pt in noisy_features_2d), min(pt[1] for pt in compressed_features_2d)) - padding
        y_max_roi = max(max(pt[1] for pt in original_features_2d), max(pt[1] for pt in noisy_features_2d), max(pt[1] for pt in compressed_features_2d)) + padding

        plt.xlim(x_min_roi, x_max_roi)
        plt.ylim(y_min_roi, y_max_roi)

        plt.legend(handles=legend_elements, title="Classes", loc="upper right")
        plt.title(f"Decision Boundaries (SVM) - Top {n}")
        plt.show()



def visualize_decision_boundary_svm_top10_closest_and_shifted(categories, model, images, labels_int, class_names, original_points, noisy_points, compressed_points, resolution=500, padding=7.0):
    # Get the output of the second-to-last layer of the model
    feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)

    # Compute features for all images
    features = feature_model.predict(images)

    # Compute features for the original, noisy, and compressed points
    original_features = [feature_model.predict(np.expand_dims(point, axis=0)) for point in original_points]
    noisy_features = [feature_model.predict(np.expand_dims(point, axis=0)) for point in noisy_points]
    compressed_features = [feature_model.predict(np.expand_dims(point, axis=0)) for point in compressed_points]

    # Combine all features for t-SNE
    all_features = np.concatenate([features] + original_features + noisy_features + compressed_features)

    # Use t-SNE to reduce dimensionality to 2D for all image features
    tsne = TSNE(n_components=2, random_state=42)
    all_features_2d = tsne.fit_transform(all_features)

    # The 2D coordinates for original_features, noisy_features, and compressed_features are the last 30 rows in features_2d
    features_2d = all_features_2d[:-30]
    original_features_2d = all_features_2d[-30:-20]
    noisy_features_2d = all_features_2d[-20:-10]
    compressed_features_2d = all_features_2d[-10:]

    classifier = SVC(kernel='rbf')
    classifier.fit(features_2d, labels_int)  # We exclude the last 30 points (original, noisy, and compressed) when training the classifier

    # Set figure size
    plt.figure(figsize=(10, 10))

    # Define the grid on which we will evaluate the classifier
    x_min, x_max = features_2d[:, 0].min() - padding, features_2d[:, 0].max() + padding
    y_min, y_max = features_2d[:, 1].min() - padding, features_2d[:, 1].max() + padding
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))

    # Evaluate the classifier on the grid
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8)

    # Define the colors for the points and arrows
    original_color = 'blue'
    noisy_color = 'green'
    compressed_colors = ['red', 'cyan', 'magenta', 'yellow', 'lime', 'brown', 'purple', 'orange', 'gray', 'pink', 'lightgray', 'darkorange']
    arrow_colors = ['olive', 'teal', 'navy', 'maroon', 'gold', 'silver', 'violet', 'indigo', 'coral', 'crimson', 'darkviolet', 'lightcoral']


    # Create a legend handle list
    legend_elements = []

    # Plot the original, noisy, and compressed points
    for i in range(10):
        # Get the class names for original, noisy, and compressed features
        original_class = class_names[classifier.predict(original_features_2d[i].reshape(1, -1))[0]]
        noisy_class = class_names[classifier.predict(noisy_features_2d[i].reshape(1, -1))[0]]
        compressed_class = class_names[classifier.predict(compressed_features_2d[i].reshape(1, -1))[0]]

        # Plot the original, noisy, and compressed points
        plt.scatter(original_features_2d[i, 0], original_features_2d[i, 1], color=original_color, edgecolors='k', s=150)
        plt.scatter(noisy_features_2d[i, 0], noisy_features_2d[i, 1], color=noisy_color, edgecolors='k', s=150)
        plt.scatter(compressed_features_2d[i, 0], compressed_features_2d[i, 1], color=compressed_colors[i], edgecolors='k', s=150)

        # Draw arrows from original to noisy, and from noisy to compressed
        plt.arrow(original_features_2d[i, 0], original_features_2d[i, 1], noisy_features_2d[i, 0] - original_features_2d[i, 0], noisy_features_2d[i, 1] - original_features_2d[i, 1], color=arrow_colors[i], width=0.01)
        plt.arrow(noisy_features_2d[i, 0], noisy_features_2d[i, 1], compressed_features_2d[i, 0] - noisy_features_2d[i, 0], compressed_features_2d[i, 1] - noisy_features_2d[i, 1], color=arrow_colors[i+1], width=0.01)

        # Add color information to the legend
        legend_elements.append(Line2D([0], [0], linestyle="none", marker='o', markersize=10, markerfacecolor=original_color, label=f'Original - {original_class} - Pair {i+1}'))
        legend_elements.append(Line2D([0], [0], linestyle="none", marker='o', markersize=10, markerfacecolor=noisy_color, label=f'Brightness Variation - {noisy_class} - Pair {i+1}'))
        legend_elements.append(Line2D([0], [0], linestyle="none", marker='o', markersize=10, markerfacecolor=compressed_colors[i], label=f'Gaussian Noisy - {compressed_class} - Pair {i+1}'))
        legend_elements.append(Line2D([0], [0], color=arrow_colors[i], lw=2, label=f'Arrow 1 - Pair {i+1}'))
        legend_elements.append(Line2D([0], [0], color=arrow_colors[i+1], lw=2, label=f'Arrow 2 - Pair {i+1}'))

    plt.legend(handles=legend_elements, title="Classes")
    plt.title("Decision Boundaries (SVM)")
    plt.show()



def evaluate_models(data_dir, salt_probs, pepper_probs, model_names, variation_intensity, num_iterations):
    categories = os.listdir(data_dir)
    categories = categories[:20]

    metrics = {}

    for model_name in model_names:
        metrics[model_name] = {}
        for salt_prob, pepper_prob in zip(salt_probs, pepper_probs):
            metrics[model_name][(salt_prob, pepper_prob)] = {
                'avg_psnr_noisy_list': [],
                'avg_psnr_brightness_list': [],
                'highest_val_acc_list': [],
                'lowest_val_loss_list': [],
                'highest_val_acc_top5_list': [],
                'misclassified_counts_list': [],
                'avg_noise_shift_list': [],
                'avg_brightness_shift_list': [],
                'most_shifted_distances': [],
                'least_shifted_distances': []
            }

    original_images, labels = generate_original_images(data_dir, categories)
    clean_images = original_images
    _, brightness_varied_images, labels = generate_brightness_variation_images(original_images, labels, variation_intensity, num_iterations)
    for salt_prob, pepper_prob in zip(salt_probs, pepper_probs):
        print(f"Now examining Salt and Pepper Noise with Salt Probability of {salt_prob} and Pepper Probability of {pepper_prob}")
        images_plot, noise_samples_plot, noisy_images_plot = generate_gradual_brightness_variation(data_dir, categories, variation_intensity, salt_prob, pepper_prob, num_iterations)
        plot_gradual_brightness_variation(images_plot, noise_samples_plot, noisy_images_plot)

        # Generate brightness varied and noisy images
        noisy_images, _, labels = generate_noisy_images(brightness_varied_images, labels, salt_prob, pepper_prob)
        train_images, val_images, test_images, train_labels, val_labels, num_classes, class_names, avg_psnr_noisy, avg_psnr_brightness = image_preprocessing(categories, original_images, brightness_varied_images, noisy_images, labels)

         # Iterate over each model
        for model_name in model_names:
            print(f"Now examining the model: {model_name}")

            # Construct model
            model = model_construction(model_name, num_classes, trainable = False)

            # Train the model and plot accuracy/loss
            history = train_plotting(model, train_images, train_labels, val_images, val_labels)

            # Append average PSNR for brightness varied images
            if avg_psnr_brightness == float('inf'):
                avg_psnr_brightness = 100
            metrics[model_name][(salt_prob, pepper_prob)]['avg_psnr_brightness_list'].append(avg_psnr_brightness)

            if avg_psnr_noisy == float('inf'):
                avg_psnr_noisy = 100
            metrics[model_name][(salt_prob, pepper_prob)]['avg_psnr_noisy_list'].append(avg_psnr_noisy)

            # Find the epoch with the highest validation accuracy
            highest_val_acc_epoch = np.argmax(history.history['val_accuracy']) + 1
            highest_val_acc = np.max(history.history['val_accuracy'])
            # Append highest_val_acc to list
            metrics[model_name][(salt_prob, pepper_prob)]['highest_val_acc_list'].append(highest_val_acc)
            # Find the epoch with the lowest validation loss
            lowest_val_loss_epoch = np.argmin(history.history['val_loss']) + 1
            lowest_val_loss = np.min(history.history['val_loss'])
            # Append lowest_val_loss to list
            metrics[model_name][(salt_prob, pepper_prob)]['lowest_val_loss_list'].append(lowest_val_loss)
            print(f"For {model_name}, highest validation accuracy of {highest_val_acc} was achieved at epoch {highest_val_acc_epoch}")
            print(f"For {model_name}, lowest validation loss of {lowest_val_loss} was achieved at epoch {lowest_val_loss_epoch}")

            # Plot top5 accuracy
            highest_val_acc_top5, highest_val_acc_epoch_top5 = plot_top5_accuracy(history)
            # Append highest_val_acc_top5 to list
            metrics[model_name][(salt_prob, pepper_prob)]['highest_val_acc_top5_list'].append(highest_val_acc_top5)
            print(f"For {model_name}, highest validation top 5 accuracy of {highest_val_acc_top5} was achieved at epoch {highest_val_acc_epoch_top5}")

            # Visualise true and predicted labels in feature spaces and use classifier to create decision boundaries
            misclassified_counts, total_misclassified = visualize_feature_space(model, val_images, val_labels, class_names, resolution=500)
            print(misclassified_counts)
            # Append misclassified_counts to list
            metrics[model_name][(salt_prob, pepper_prob)]['misclassified_counts_list'].append(total_misclassified)
            # visualize_decision_boundary_knn(model, val_images, val_labels, class_names, resolution=500)
            # visualize_decision_boundary_rf(model, val_images, val_labels, class_names, resolution=500)
            visualize_decision_boundary_svm(model, val_images, val_labels, class_names, resolution=500)

            # Compute average shift due to gaussian noise
            avg_shift_noisy = average_shift(model, clean_images, brightness_varied_images)
            # Append avg_shift to list
            metrics[model_name][(salt_prob, pepper_prob)]['avg_brightness_shift_list'].append(avg_shift_noisy)
            print(f"Average shift due to brightness variation for model {model_name} is {avg_shift_noisy}")

            # Compute average shift due to compression
            avg_shift = average_shift(model, clean_images, noisy_images)
            # Append avg_shift to list
            metrics[model_name][(salt_prob, pepper_prob)]['avg_noise_shift_list'].append(avg_shift)
            print(f"Average shift due to gaussian noise and bright variation for model {model_name} is {avg_shift}")


            # Compute the features for all data points
            feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
            clean_features_all = feature_model.predict(np.array(clean_images))
            brightness_varied_features_all = feature_model.predict(np.array(brightness_varied_images))
            noisy_features_all = feature_model.predict(np.array(noisy_images))

            # Compute the distance (shift due to brightness variation) for each data point
            distances_brightness_variation = np.sqrt(np.sum((clean_features_all - brightness_varied_features_all)**2, axis=1))
            # Compute the distance (shift due to noise) for each data point
            distances_noise = np.sqrt(np.sum((brightness_varied_features_all - noisy_features_all)**2, axis=1))
            # Combine the distances to get an overall shift
            overall_distances = distances_brightness_variation + distances_noise

            # Find the index of the data point with the top 5 maximum distance
            most_shifted_indices = np.argsort(overall_distances)[-5:][::-1].tolist()
            print("Top 5 distances between each pair of original and shifted points in high-dimensional space:", overall_distances[most_shifted_indices])
            # Append the maximum distance to the most_shifted_distances list
            metrics[model_name][(salt_prob, pepper_prob)]['most_shifted_distances'].append(overall_distances[most_shifted_indices[0]])

            # Now you can use `most_shifted_indices` to analyze the top 5 most severely shifted data points
            most_shifted_clean = [clean_images[i] for i in most_shifted_indices]
            most_shifted_brightness_varied = [brightness_varied_images[i] for i in most_shifted_indices]
            most_shifted_noisy = [noisy_images[i] for i in most_shifted_indices]

            # Visualize the decision boundary, original point, brightness varied point, and noisy point
            visualize_decision_boundary_svm_shifted_topn(categories, model, np.array(clean_images), labels, class_names, most_shifted_clean, most_shifted_brightness_varied, most_shifted_noisy, n_pairs=5, resolution=500, padding=12.0)

            # Find the index of the data point with the top 5 minimum distance
            least_shifted_indices = np.argsort(overall_distances)[:5].tolist()
            print("Top 5 distances between each pair of original and shifted points in high-dimensional space:", overall_distances[least_shifted_indices])
            # Append the minimum distance to the least_shifted_distances list
            metrics[model_name][(salt_prob, pepper_prob)]['least_shifted_distances'].append(overall_distances[least_shifted_indices[0]])

            # Now you can use `least_shifted_indices` to analyze the top 5 least severely shifted data points
            least_shifted_clean = [clean_images[i] for i in least_shifted_indices]
            least_shifted_brightness_varied = [brightness_varied_images[i] for i in least_shifted_indices]
            least_shifted_noisy = [noisy_images[i] for i in least_shifted_indices]

            # Visualize the decision boundary, original point, brightness varied point, and noisy point for least shifted
            visualize_decision_boundary_svm_shifted_topn(categories, model, np.array(clean_images), labels, class_names, least_shifted_clean, least_shifted_brightness_varied, least_shifted_noisy, n_pairs=5, resolution=500, padding=12.0)

            # Train SVM on all original data
            labels = np.array(labels)
            le = LabelEncoder()
            labels_int = le.fit_transform(labels)
            svm = SVC(kernel='rbf')
            svm.fit(clean_features_all, labels_int)
            # Compute the distance of each point to the decision boundary
            decision_function = svm.decision_function(clean_features_all)
            # Compute the minimum distance of each point to the decision boundary across all classes
            min_distance_to_boundary = np.min(np.abs(decision_function), axis=1)
            # Find the indices of the 10 points closest to the decision boundary
            closest_indices = np.argsort(min_distance_to_boundary)[:10]
            # Now you can use `closest_indices` to analyze the 10 points closest to the decision boundary
            closest_clean = [clean_images[i] for i in closest_indices]
            closest_brightness_varied = [brightness_varied_images[i] for i in closest_indices]
            closest_noisy = [noisy_images[i] for i in closest_indices]
            visualize_decision_boundary_svm_top10_closest_and_shifted(categories, model, np.array(clean_images), labels_int, class_names, closest_clean, closest_brightness_varied, closest_noisy, resolution=500, padding=15.0)

    return metrics



def generate_plots_for_model(model_name, metrics1, metrics2, metrics3, metrics4, metrics5, metrics6, probs):
    metric_pairs = [
        ('probs', 'avg_psnr_noisy_list', 'Probability of Salt & Pepper Noise', 'Average PSNR (Noise)'),
        ('probs', 'avg_psnr_brightness_list', 'Probability of Salt & Pepper Noise', 'Average PSNR (Brightness)'),
        ('probs', 'highest_val_acc_list', 'Probability of Salt & Pepper Noise', 'Highest Validation Accuracy'),
        ('probs', 'lowest_val_loss_list', 'Probability of Salt & Pepper Noise', 'Lowest Validation Loss'),
        ('probs', 'highest_val_acc_top5_list', 'Probability of Salt & Pepper Noise', 'Highest Top-5 Validation Accuracy'),
        ('probs', 'misclassified_counts_list', 'Probability of Salt & Pepper Noise', 'Misclassified Counts'),
        ('probs', 'avg_brightness_shift_list', 'Probability of Salt & Pepper Noise', 'Average Shift (Brightness)'),
        ('probs', 'avg_noise_shift_list', 'Probability of Salt & Pepper Noise', 'Average Shift (Noise)'),
        ('probs', 'most_shifted_distances', 'Probability of Salt & Pepper Noise', 'Most Shifted Distances'),
        ('probs', 'least_shifted_distances', 'Probability of Salt & Pepper Noise', 'Least Shifted Distances'),
        ('avg_psnr_noisy_list', 'highest_val_acc_list', 'Average PSNR (Noise)', 'Highest Validation Accuracy'),
        ('avg_psnr_noisy_list', 'lowest_val_loss_list', 'Average PSNR (Noise)', 'Lowest Validation Loss'),
        ('avg_psnr_noisy_list', 'highest_val_acc_top5_list', 'Average PSNR (Noise)', 'Highest Top-5 Validation Accuracy'),
        ('avg_noise_shift_list', 'avg_psnr_noisy_list', 'Average Shift (Noise)', 'Average PSNR (Noise)'),
        ('avg_brightness_shift_list', 'highest_val_acc_list', 'Average Shift (Brightness)', 'Highest Validation Accuracy'),
        ('avg_noise_shift_list', 'highest_val_acc_list', 'Average Shift (Noise)', 'Highest Validation Accuracy'),
    ]

    for i in range(0, len(metric_pairs), 2):
        fig, axs = plt.subplots(1, 2, figsize=(22, 10))

        for j in range(2):
            x_metric, y_metric, x_label, y_label = metric_pairs[i + j]
            axs[j].grid(True)
            axs[j].set_title(f'{y_label} vs {x_label}', fontsize=18, fontweight='bold')
            axs[j].set_xlabel(x_label, fontsize=18, fontweight='bold')
            axs[j].set_ylabel(y_label, fontsize=18, fontweight='bold')

            for metric, color, label in zip([metrics1, metrics2, metrics3, metrics4, metrics5],
                                            ['b', 'g', 'r', 'y', 'm'],
                                            ['Brightness Variation Level 0', 'Brightness Variation Level 1', 'Brightness Variation Level 3', 'Brightness Variation Level 5', 'Brightness Variation Level 7']):
                x_values = []
                y_values = []

                for prob in probs:
                    data = metric[model_name][(prob, prob)]
                    if x_metric == 'probs':
                        x_values.append(prob)
                    else:
                        x_values.append(data[x_metric][0])

                    y_values.append(data[y_metric][0])

                axs[j].scatter(x_values, y_values, color=color, s=100, label=label)
                axs[j].plot(x_values, y_values, color=color, linewidth=3)

            # x_ticks = np.arange(min(x_values), max(x_values), (max(x_values) - min(x_values)) / 10)
            axs[j].tick_params(axis='both', which='major', labelsize=16)
            axs[j].legend(fontsize=16)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, wspace=0.2)
        plt.show()


def plot_heatmap(metrics_list, model_names, probs, metric_key, title):
    brightness_variation = [0, 1, 3, 5, 7, 9]  # Adjusted from 'colour_variation'

    for model_name in model_names:
        # Create an empty 2D array to store the values
        values = []

        # Iterate through the brightness variation levels
        for metric in metrics_list:
            row = []
            for prob in probs:
                # Extract the value for this probability and brightness variation level
                value = metric[model_name][(prob, prob)][metric_key][0]
                row.append(value)
            values.append(row)

        # Convert the 2D array to a Pandas DataFrame
        df = pd.DataFrame(values, index=brightness_variation, columns=probs)

        # Create the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, annot=True, cmap="YlGnBu")
        plt.xlabel('Probability of Salt & Pepper Noise')
        plt.ylabel('Brightness Variation Level')
        plt.title(f'{title} Heatmap for {model_name}')
        plt.show()


def sensitivity_analysis(metrics_list, model_names, probs, metric_key, title):
    brightness_variation = [0, 1, 3, 5, 7, 9]  # Updated from 'colour_variation'

    # Get a color palette with 20 distinct colors
    colors = sns.color_palette("tab20", 20)

    plt.figure(figsize=(15, 10))

    color_idx = 0
    for model_name in model_names:
        for idx, metric in enumerate(metrics_list):
            y_values = []
            for prob in probs:
                value = metric[model_name][(prob, prob)][metric_key][0]
                y_values.append(value)

            plt.plot(probs, y_values, label=f'{model_name} (Brightness Variation Level = {brightness_variation[idx]})', color=colors[color_idx], linewidth=4)
            color_idx += 1

    plt.xlabel('Probability of Salt & Pepper Noise', fontsize=18, fontweight='bold')
    plt.ylabel(title, fontsize=18, fontweight='bold')
    plt.title(f'Sensitivity Analysis: {title} vs Probability', fontsize=20, fontweight='bold')
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.show()


def plot_3d_surface(metrics_list, model_names, probs, metric_key, title):
    brightness_variation = [0, 1, 3, 5, 7, 9]  # Updated for brightness variation levels

    for model_name in model_names:
        X, Y, Z = [], [], []
        for brightness_level, metrics in zip(brightness_variation, metrics_list):
            for prob in probs:
                X.append(brightness_level)
                Y.append(prob)
                Z.append(metrics[model_name][(prob, prob)][metric_key][0])

        # Convert to numpy arrays for manipulation
        X = np.array(X).reshape(len(brightness_variation), len(probs))
        Y = np.array(Y).reshape(len(brightness_variation), len(probs))
        Z = np.array(Z).reshape(len(brightness_variation), len(probs))

        # Create the 3D surface plot
        fig = go.Figure()
        fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', name='Surface'))
        fig.update_layout(
            title=dict(text=f'{title} 3D Surface Plot for {model_name}', font=dict(size=24, color='black', family="Arial, bold")),
            autosize=False,
            width=1000,
            height=800,
            scene=dict(
                xaxis_title='Brightness Variation Level',
                yaxis_title='Probability of Salt & Pepper Noise',
                zaxis_title=title,
                xaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)', showbackground=True, backgroundcolor='rgb(230, 230,230)', title_font=dict(size=18, color='black', family="Arial, bold")),
                yaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)', showbackground=True, backgroundcolor='rgb(230, 230,230)', title_font=dict(size=18, color='black', family="Arial, bold")),
                zaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)', showbackground=True, backgroundcolor='rgb(230, 230,230)', title_font=dict(size=18, color='black', family="Arial, bold"))
            ),
            font=dict(family="Arial, bold", size=16, color="black"),
            coloraxis_colorbar=dict(title='Value', title_font=dict(size=18, family="Arial, bold"), tickfont=dict(size=16, family="Arial, bold"))
        )
        fig.show()
        

def run_experiment_five_brightness_motion(data_dir):
    """
    Run an experiment involving brightness variations and motion blur.

    The function:
    1. Sets up directory, model names, and parameters.
    2. Evaluates the models under different variations of brightness and motion.
    3. Generates plots for each model.
    4. Plots heatmaps for various metrics.
    5. Conducts sensitivity analysis for the models.
    6. Plots 3D surface plots for various metrics.
    
    Parameters:
    - data_dir (str): The directory path containing the original image data.
    
    Returns:
    None (outputs are generated as plots and printed summaries).
    """

    # 1. Setup
    model_names = ['VGG16', 'MobileNet', 'DenseNet121']
    
    # 2. Evaluate models
    metrics59 = evaluate_models(data_dir, kernel_size=10, angle=45, model_names=model_names, variation_intensity=1.5, num_iterations=0, num_iterations_motion=7)
    metrics60 = evaluate_models(data_dir, kernel_size=10, angle=45, model_names=model_names, variation_intensity=1.5, num_iterations=1, num_iterations_motion=7)
    metrics61 = evaluate_models(data_dir, kernel_size=10, angle=45, model_names=model_names, variation_intensity=1.5, num_iterations=3, num_iterations_motion=7)
    metrics62 = evaluate_models(data_dir, kernel_size=10, angle=45, model_names=model_names, variation_intensity=1.5, num_iterations=5, num_iterations_motion=7)
    metrics63 = evaluate_models(data_dir, kernel_size=10, angle=45, model_names=model_names, variation_intensity=1.5, num_iterations=7, num_iterations_motion=7)
    metrics64 = evaluate_models(data_dir, kernel_size=10, angle=45, model_names=model_names, variation_intensity=1.5, num_iterations=9, num_iterations_motion=7)
    metrics_list = [metrics59, metrics60, metrics61, metrics62, metrics63, metrics64]

    # 3. Generate plots
    iterations = [str(i) for i in range(1, 8)]
    for model_name in model_names:
        print(f'Generating plots for {model_name}...')
        generate_plots_for_model(model_name, metrics59, metrics60, metrics61, metrics62, metrics63, iterations)
        print(f'Plots for {model_name} generated successfully.\n')

    # 4. Heatmaps
    plot_heatmap(metrics_list, model_names, iterations, 'avg_psnr_blurred_list', 'Average PSNR (Blurred)')
    plot_heatmap(metrics_list, model_names, iterations, 'avg_blur_shift_list', 'Average Shift (Blurred)')
    plot_heatmap(metrics_list, model_names, iterations, 'highest_val_acc_list', 'Highest Validation Accuracy')
    plot_heatmap(metrics_list, model_names, iterations, 'highest_val_acc_top5_list', 'Highest Top-5 Validation Accuracy')

    # 5. Sensitivity analysis
    sensitivity_analysis(metrics_list, model_names, iterations, 'avg_psnr_blurred_list', 'Average PSNR (Blurred)')
    sensitivity_analysis(metrics_list, model_names, iterations, 'avg_blur_shift_list', 'Average Shift (Blurred)')
    sensitivity_analysis(metrics_list, model_names, iterations, 'highest_val_acc_list', 'Highest Validation Accuracy')
    sensitivity_analysis(metrics_list, model_names, iterations, 'highest_val_acc_top5_list', 'Highest Top-5 Validation Accuracy')

    # 6. 3D Surface plots
    plot_3d_surface(metrics_list, model_names, iterations, 'avg_psnr_blurred_list', 'Average PSNR (Blurred)')
    plot_3d_surface(metrics_list, model_names, iterations, 'avg_blur_shift_list', 'Average Shift (Blurred)')
    plot_3d_surface(metrics_list, model_names, iterations, 'highest_val_acc_list', 'Highest Validation Accuracy')
    plot_3d_surface(metrics_list, model_names, iterations, 'highest_val_acc_top5_list', 'Highest Top-5 Validation Accuracy')


def run_experiment_five_brightness_salt(data_dir):
    """
    Run an experiment involving brightness variations combined with salt and pepper noise.

    The function:
    1. Sets up directory, categories, model names, and parameters.
    2. Evaluates the models under different variations of brightness and noise.
    3. Generates plots for each model.
    4. Plots heatmaps for various metrics.
    5. Conducts sensitivity analysis for the models.
    6. Plots 3D surface plots for various metrics.
    
    Parameters:
    - data_dir (str): The directory path containing the original image data.
    
    Returns:
    None (outputs are generated as plots and printed summaries).
    """
    import os
    
    # 1. Setup
    categories = os.listdir(data_dir)
    categories = categories[:20]  # Limiting to the first 20 categories for demonstration

    model_names = ['VGG16', 'MobileNet', 'DenseNet121']
    salt_probs = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    pepper_probs = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    # 2. Evaluate models
    metrics65 = evaluate_models(data_dir, salt_probs, pepper_probs, model_names, variation_intensity=1.5, num_iterations=0)
    metrics66 = evaluate_models(data_dir, salt_probs, pepper_probs, model_names, variation_intensity=1.5, num_iterations=1)
    metrics67 = evaluate_models(data_dir, salt_probs, pepper_probs, model_names, variation_intensity=1.5, num_iterations=3)
    metrics68 = evaluate_models(data_dir, salt_probs, pepper_probs, model_names, variation_intensity=1.5, num_iterations=5)
    metrics69 = evaluate_models(data_dir, salt_probs, pepper_probs, model_names, variation_intensity=1.5, num_iterations=7)
    metrics70 = evaluate_models(data_dir, salt_probs, pepper_probs, model_names, variation_intensity=1.5, num_iterations=9)
    metrics_list = [metrics65, metrics66, metrics67, metrics68, metrics69, metrics70]

    # 3. Generate plots
    probs = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for model_name in model_names:
        print(f'Generating plots for {model_name}...')
        generate_plots_for_model(model_name, metrics65, metrics66, metrics67, metrics68, metrics69, metrics70, probs)
        print(f'Plots for {model_name} generated successfully.\n')

    # 4. Heatmaps
    plot_heatmap(metrics_list, model_names, probs, 'avg_psnr_noisy_list', 'Average PSNR')
    plot_heatmap(metrics_list, model_names, probs, 'avg_noise_shift_list', 'Average Shift')
    plot_heatmap(metrics_list, model_names, probs, 'highest_val_acc_list', 'Highest Validation Accuracy')
    plot_heatmap(metrics_list, model_names, probs, 'highest_val_acc_top5_list', 'Highest Top-5 Validation Accuracy')

    # 5. Sensitivity analysis
    sensitivity_analysis(metrics_list, model_names, probs, 'avg_psnr_noisy_list', 'Average PSNR')
    sensitivity_analysis(metrics_list, model_names, probs, 'avg_noise_shift_list', 'Average Shift')
    sensitivity_analysis(metrics_list, model_names, probs, 'highest_val_acc_list', 'Highest Validation Accuracy')
    sensitivity_analysis(metrics_list, model_names, probs, 'highest_val_acc_top5_list', 'Highest Top-5 Validation Accuracy')

    # 6. 3D Surface plots
    plot_3d_surface(metrics_list, model_names, probs, 'avg_psnr_noisy_list', 'Average PSNR')
    plot_3d_surface(metrics_list, model_names, probs, 'avg_noise_shift_list', 'Average Shift')
    plot_3d_surface(metrics_list, model_names, probs, 'highest_val_acc_list', 'Highest Validation Accuracy')
    plot_3d_surface(metrics_list, model_names, probs, 'highest_val_acc_top5_list', 'Highest Top-5 Validation Accuracy')
