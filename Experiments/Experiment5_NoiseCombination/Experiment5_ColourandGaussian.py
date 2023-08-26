#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 23:02:45 2023

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
import random
import cv2
from sklearn.cluster import KMeans

# Matplotlib for color mappings
from matplotlib.colors import ListedColormap

#%%
def apply_color_variation(image, variation_level):
    color_image = image.copy()

    if variation_level == 0:
        return color_image

    elif variation_level == 1:
        channel_to_keep = random.choice([0, 1, 2])
        color_image[..., [i for i in range(3) if i != channel_to_keep]] = 0

    elif variation_level == 2:
        channel_to_remove = random.choice([0, 1, 2])
        color_image[..., channel_to_remove] = 0

    elif variation_level == 3:
        channel_to_modify = random.choice([0, 1, 2])
        color_image[..., channel_to_modify] = random.randint(0, 255)

    elif variation_level == 4:
        lab_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2LAB)
        channel_to_modify = random.choice([1, 2]) # Choose either A-channel or B-channel
        lab_image[..., channel_to_modify] = random.randint(0, 255)
        color_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

    elif variation_level == 5:
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
        hsv_image[..., 1] = random.randint(0, 255)
        color_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    elif variation_level == 6:
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
        hsv_image[..., 0] = random.randint(0, 179)
        color_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    elif variation_level == 7:
        # Scale the color values to a smaller range (e.g., 0.1 to 0.9)
        color_image = color_image * 0.3 + 0.1

    elif variation_level == 8:
        # Reshape the image to be a list of pixels
        pixels = color_image.reshape((-1, 3))

        # Perform k-means clustering to reduce the number of colors
        kmeans = KMeans(n_clusters=2, n_init=10).fit(pixels)
        labels = kmeans.predict(pixels)
        palette = kmeans.cluster_centers_

        # Reconstruct the image with the reduced color palette
        quantized_image = np.zeros_like(pixels)
        for i in range(len(pixels)):
            quantized_image[i] = palette[labels[i]]
        color_image = quantized_image.reshape(color_image.shape)

    elif variation_level == 9:
        # Novel noise: Inverting the color channels
        color_image = np.roll(color_image, shift=1, axis=2)

    return color_image



def apply_variations_to_dataset(images, variation_level):
    num_images = len(images)
    distorted_images = images.copy()  # Start with a copy of the original images

    # If variation_level is 0, return the original images
    if variation_level == 0:
        print("Variation level is 0, returning original images.")
        return images

    # Calculate the number of images for each level up to the specified variation_level
    num_per_level = num_images // variation_level
    print(f"Number of images per level: {num_per_level}")

    # Create an array of variation levels to be applied
    variations = np.concatenate([np.full(num_per_level, i) for i in range(1, variation_level + 1)])
    remaining_images = num_images - len(variations)
    variations = np.concatenate([variations, np.full(remaining_images, variation_level)])

    # Shuffle the variations
    np.random.shuffle(variations)

    # Apply the variations to the images
    for idx, var_level in enumerate(variations):
        distorted_images[idx] = apply_color_variation(distorted_images[idx], var_level)

    print(f"Total distorted images: {len(distorted_images)}")  # Debug print
    return distorted_images



def generate_color_variation_images(data_dir, categories, variation_level):
    original_images = []
    noisy_images = []
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

    # Apply the color variations to the entire dataset
    noisy_images = apply_variations_to_dataset(original_images, variation_level)

    return np.array(original_images), np.array(noisy_images), labels



def generate_gaussian_noise(image, sigma):
    # Generate Gaussian noise with mean 0 and standard deviation sigma
    gaussian_noise = np.random.normal(0, sigma, image.shape)

    # Normalize the noise for visualization purposes
    noise_to_show = (gaussian_noise - gaussian_noise.min()) / (gaussian_noise.max() - gaussian_noise.min())

    # Add the noise to the original image and clip the values to be in [0, 1]
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 1)

    return noise_to_show, noisy_image



def generate_noisy_images(color_varied_images, sigma, original_labels):
    noisy_images = []
    noise_samples = []
    labels = []

    for idx, image_array in enumerate(color_varied_images):
        # Apply Gaussian noise
        gaussian_noise, noisy_image_array = generate_gaussian_noise(image_array, sigma)

        noisy_images.append(noisy_image_array)
        noise_samples.append(gaussian_noise)
        labels.append(original_labels[idx])

    return color_varied_images, noisy_images, noise_samples, labels



def generate_noisy_images_with_color_variation(data_dir, categories, sigma, variation_level):
    """
    The `generate_noisy_images_with_color_variation` function produces noisy versions of images subjected to color variations. This procedure involves:
    1. Generating images with specified color variations.
    2. Applying Gaussian noise to these color-varied images.
    3. Calculating the total noise as the difference between the noisy image and the original image.
   
    Parameters:
    - data_dir (str): The directory path containing the original image data.
    - categories (list): List of categories to be included.
    - sigma (float): The standard deviation of the Gaussian noise.
    - variation_level (float): The desired level of color variation.
   
    Returns:
    - original_images (list): List of original images before color variation.
    - color_varied_images (list): List of images after applying color variations.
    - noisy_images (list): List of images after applying Gaussian noise to the color-varied images.
    - noise_samples (list): Samples of the applied Gaussian noise.
    - labels (list): Labels corresponding to each image.
    - total_noise (list): Difference between the noisy images and their corresponding original images.
    """
 
    # Generate original images, color-varied images, and labels using the generate_color_variation_images function
    original_images, color_varied_images, labels = generate_color_variation_images(data_dir, categories, variation_level)

    # Apply Gaussian noise to the color-varied images
    _, noisy_images, noise_samples, _ = generate_noisy_images(color_varied_images, sigma, labels)

    # Compute the total noise (difference between noisy image and original image)
    total_noise = [noisy - original for noisy, original in zip(noisy_images, original_images)]

    return original_images, color_varied_images, noisy_images, noise_samples, labels, total_noise



def plot_distortions(clean_image, sigma):
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))

    # Apply and plot the distortions for levels 0 to 4 (original and noisy)
    for i in range(0, 5):
        distorted_image = apply_color_variation(clean_image, i)  # Apply distortion
        distorted_image_array = np.array(distorted_image)  # Convert to array

        # Add Gaussian noise to the distorted image
        _, noisy_image_array = generate_gaussian_noise(distorted_image_array, sigma)

        axes[0, i].imshow(distorted_image)
        axes[0, i].set_title(f'Variation Level {i} (Distorted)')
        axes[0, i].axis('off')

        axes[2, i].imshow(noisy_image_array)
        axes[2, i].set_title(f'Variation Level {i} (Noisy)')
        axes[2, i].axis('off')

    # Apply and plot the distortions for levels 5 to 9 (original and noisy)
    for i in range(5, 10):
        distorted_image = apply_color_variation(clean_image, i)  # Apply distortion
        distorted_image_array = np.array(distorted_image)  # Convert to array

        # Add Gaussian noise to the distorted image
        _, noisy_image_array = generate_gaussian_noise(distorted_image_array, sigma)

        axes[1, i - 5].imshow(distorted_image)
        axes[1, i - 5].set_title(f'Variation Level {i} (Distorted)')
        axes[1, i - 5].axis('off')

        axes[3, i - 5].imshow(noisy_image_array)
        axes[3, i - 5].set_title(f'Variation Level {i} (Noisy)')
        axes[3, i - 5].axis('off')

    plt.show()



def calculate_average_psnr(original_images, decompressed_images):
    psnrs = []
    for orig, dec in zip(original_images, decompressed_images):
        orig = orig.astype('float32') / 255.
        dec = dec.astype('float32') / 255.
        psnr = peak_signal_noise_ratio(orig, dec, data_range=1.0)
        psnrs.append(psnr)
    return np.mean(psnrs)



def image_preprocessing(categories, original_images, color_varied_images, noisy_images, labels, sigma):
    # Plot the first original, color-varied, and noisy images for example
    plot_distortions(original_images[180], sigma)

    # Calculate and print the average PSNR for noisy images
    avg_psnr_colour = calculate_average_psnr(original_images, color_varied_images)
    print(f"Average PSNR for noisy image: {avg_psnr_colour} dB")

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

    return train_images, val_images, test_images, train_labels, val_labels, num_classes, class_names, avg_psnr_noisy, avg_psnr_colour


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
          legend_elements.append(Line2D([0], [0], linestyle="none", marker='o', markersize=10, markerfacecolor=noisy_color, label=f'Colour Variation - {noisy_class}'))
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
        legend_elements.append(Line2D([0], [0], linestyle="none", marker='o', markersize=10, markerfacecolor=noisy_color, label=f'Colour Variation - {noisy_class} - Pair {i+1}'))
        legend_elements.append(Line2D([0], [0], linestyle="none", marker='o', markersize=10, markerfacecolor=compressed_colors[i], label=f'Gaussian Noisy - {compressed_class} - Pair {i+1}'))
        legend_elements.append(Line2D([0], [0], color=arrow_colors[i], lw=2, label=f'Arrow 1 - Pair {i+1}'))
        legend_elements.append(Line2D([0], [0], color=arrow_colors[i+1], lw=2, label=f'Arrow 2 - Pair {i+1}'))

    plt.legend(handles=legend_elements, title="Classes")
    plt.title("Decision Boundaries (SVM)")
    plt.show()


def evaluate_models(data_dir, sigmas, model_names, variation_level):
    categories = os.listdir(data_dir)
    categories = categories[:20]

    metrics = {}

    for model_name in model_names:
        metrics[model_name] = {}
        for sigma in sigmas:
            metrics[model_name][sigma] = {
                'avg_psnr_noisy_list': [],
                'avg_psnr_colour_list': [],
                'highest_val_acc_list': [],
                'lowest_val_loss_list': [],
                'highest_val_acc_top5_list': [],
                'misclassified_counts_list': [],
                'avg_colour_shift_list': [],
                'avg_noise_shift_list': [],
                'most_shifted_distances': [],
                'least_shifted_distances': []
            }

    for sigma in sigmas:
        print(f"Now examining Gaussian Noise with Sigma of {sigma}")

        # Generate noisy and color-varied images
        original_images, color_varied_images, noisy_images, _, labels, _ = generate_noisy_images_with_color_variation(data_dir, categories, sigma, variation_level)

        train_images, val_images, test_images, train_labels, val_labels, num_classes, class_names, avg_psnr_noisy, avg_psnr_colour = image_preprocessing(categories, original_images, color_varied_images, noisy_images, labels, sigma)

        # Iterate over each model
        for model_name in model_names:
            print(f"Now examining the model: {model_name}")

            # Construct model
            model = model_construction(model_name, num_classes, trainable = False)

            # Train the model and plot accuracy/loss
            history = train_plotting(model, train_images, train_labels, val_images, val_labels)

            # Append average PSNR for noisy images
            if avg_psnr_colour == float('inf'):
                avg_psnr_colour = 100
            metrics[model_name][sigma]['avg_psnr_colour_list'].append(avg_psnr_colour)

            if avg_psnr_noisy == float('inf'):
                avg_psnr_noisy = 100
            metrics[model_name][sigma]['avg_psnr_noisy_list'].append(avg_psnr_noisy)

            # Find the epoch with the highest validation accuracy
            highest_val_acc_epoch = np.argmax(history.history['val_accuracy']) + 1
            highest_val_acc = np.max(history.history['val_accuracy'])
            metrics[model_name][sigma]['highest_val_acc_list'].append(highest_val_acc)

            # Find the epoch with the lowest validation loss
            lowest_val_loss_epoch = np.argmin(history.history['val_loss']) + 1
            lowest_val_loss = np.min(history.history['val_loss'])
            metrics[model_name][sigma]['lowest_val_loss_list'].append(lowest_val_loss)

            print(f"For {model_name}, highest validation accuracy of {highest_val_acc} was achieved at epoch {highest_val_acc_epoch}")
            print(f"For {model_name}, lowest validation loss of {lowest_val_loss} was achieved at epoch {lowest_val_loss_epoch}")

            # Plot top5 accuracy
            highest_val_acc_top5, highest_val_acc_epoch_top5 = plot_top5_accuracy(history)
            metrics[model_name][sigma]['highest_val_acc_top5_list'].append(highest_val_acc_top5)
            print(f"For {model_name}, highest validation top 5 accuracy of {highest_val_acc_top5} was achieved at epoch {highest_val_acc_epoch_top5}")

            # Visualise true and predicted labels in feature spaces and use classifier to create decision boundaries
            misclassified_counts, total_misclassified = visualize_feature_space(model, val_images, val_labels, class_names, resolution=500)
            print(misclassified_counts)
            metrics[model_name][sigma]['misclassified_counts_list'].append(total_misclassified)

            # visualize_decision_boundary_knn(model, val_images, val_labels, class_names, resolution=500)
            # visualize_decision_boundary_rf(model, val_images, val_labels, class_names, resolution=500)
            visualize_decision_boundary_svm(model, val_images, val_labels, class_names, resolution=500)

            # Compute average shift due to color variation
            avg_shift_variation = average_shift(model, original_images, color_varied_images)
            metrics[model_name][sigma]['avg_colour_shift_list'].append(avg_shift_variation)
            print(f"Average shift due to color variation for model {model_name} is {avg_shift_variation}")

            # Compute average shift due to colour variation and gaussian noise
            avg_shift_noisy = average_shift(model, original_images, noisy_images)
            metrics[model_name][sigma]['avg_noise_shift_list'].append(avg_shift_noisy)
            print(f"Average shift due to gaussian noise for model {model_name} is {avg_shift_noisy}")

            # Compute the features for all data points
            feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
            original_features_all = feature_model.predict(np.array(original_images))
            color_varied_features_all = feature_model.predict(np.array(color_varied_images))
            noisy_features_all = feature_model.predict(np.array(noisy_images))

            # Compute the distance (shift due to color variation) for each data point
            distances_variation = np.sqrt(np.sum((original_features_all - color_varied_features_all)**2, axis=1))
            # Compute the distance (shift due to noise) for each data point
            distances_noise = np.sqrt(np.sum((color_varied_features_all - noisy_features_all)**2, axis=1))
            # Combine the distances to get an overall shift
            overall_distances = distances_noise + distances_variation

            # Find the index of the data point with the top 5 maximum distance
            most_shifted_indices = np.argsort(overall_distances)[-5:][::-1].tolist()
            print("Top 5 distances between each pair of original and shifted points in high-dimensional space:", overall_distances[most_shifted_indices])
            metrics[model_name][sigma]['most_shifted_distances'].append(overall_distances[most_shifted_indices[0]])

            # Now you can use `most_shifted_indices` to analyze the top 5 most severely shifted data points
            most_shifted_originals = [original_images[i] for i in most_shifted_indices]
            most_shifted_color_varied = [color_varied_images[i] for i in most_shifted_indices]
            most_shifted_noisys = [noisy_images[i] for i in most_shifted_indices]

            # Visualize the decision boundary, original point, noisy point, and color-varied point
            visualize_decision_boundary_svm_shifted_topn(categories, model, np.array(original_images), labels, class_names, most_shifted_originals, most_shifted_color_varied, most_shifted_noisys, n_pairs=5, resolution=500, padding=12.0)

            # Find the index of the data point with the top 5 minimum distance
            least_shifted_indices = np.argsort(overall_distances)[:5].tolist()
            print("Top 5 distances between each pair of original and shifted points in high-dimensional space:", overall_distances[least_shifted_indices])
            metrics[model_name][sigma]['least_shifted_distances'].append(overall_distances[least_shifted_indices[0]])

            # Now you can use `least_shifted_indices` to analyze the top 5 least severely shifted data points
            least_shifted_originals = [original_images[i] for i in least_shifted_indices]
            least_shifted_color_varied = [color_varied_images[i] for i in least_shifted_indices]
            least_shifted_noisys = [noisy_images[i] for i in least_shifted_indices]

            # Visualize the decision boundary, original point, noisy point, and color-varied point for least shifted
            visualize_decision_boundary_svm_shifted_topn(categories, model, np.array(original_images), labels, class_names, least_shifted_originals, least_shifted_color_varied, least_shifted_noisys, n_pairs=5, resolution=500, padding=12.0)

            # Train SVM on all original data
            labels = np.array(labels)
            le = LabelEncoder()
            labels_int = le.fit_transform(labels)
            svm = SVC(kernel='rbf')
            svm.fit(original_features_all, labels_int)

            # Compute the distance of each point to the decision boundary
            decision_function = svm.decision_function(original_features_all)

            # Compute the minimum distance of each point to the decision boundary across all classes
            min_distance_to_boundary = np.min(np.abs(decision_function), axis=1)

            # Find the indices of the 10 points closest to the decision boundary
            closest_indices = np.argsort(min_distance_to_boundary)[:10]

            # Now you can use `closest_indices` to analyze the 10 points closest to the decision boundary
            closest_originals = [original_images[i] for i in closest_indices]
            closest_color_varied = [color_varied_images[i] for i in closest_indices]
            closest_noisys = [noisy_images[i] for i in closest_indices]

            visualize_decision_boundary_svm_top10_closest_and_shifted(categories, model, np.array(original_images), labels_int, class_names, closest_originals, closest_color_varied, closest_noisys, resolution=500, padding=15.0)

    return metrics


def generate_plots_for_model(model_name, metrics1, metrics2, metrics3, metrics4, metrics5, sigmas):
    metric_pairs = [
        ('sigma', 'avg_psnr_noisy_list', 'Sigma', 'Average PSNR (Noise)'),
        ('sigma', 'avg_psnr_colour_list', 'Sigma', 'Average PSNR (Colour)'),
        ('sigma', 'highest_val_acc_list', 'Sigma', 'Highest Validation Accuracy'),
        ('sigma', 'lowest_val_loss_list', 'Sigma', 'Lowest Validation Loss'),
        ('sigma', 'highest_val_acc_top5_list', 'Sigma', 'Highest Top-5 Validation Accuracy'),
        ('sigma', 'misclassified_counts_list', 'Sigma', 'Misclassified Counts'),
        ('sigma', 'avg_colour_shift_list', 'Sigma', 'Average Shift (Colour)'),
        ('sigma', 'avg_noise_shift_list', 'Sigma', 'Average Shift (Noise)'),
        ('sigma', 'most_shifted_distances', 'Sigma', 'Most Shifted Distances'),
        ('sigma', 'least_shifted_distances', 'Sigma', 'Least Shifted Distances'),
        ('avg_psnr_noisy_list', 'highest_val_acc_list', 'Average PSNR (Noise)', 'Highest Validation Accuracy'),
        ('avg_psnr_noisy_list', 'lowest_val_loss_list', 'Average PSNR (Noise)', 'Lowest Validation Loss'),
        ('avg_psnr_noisy_list', 'highest_val_acc_top5_list', 'Average PSNR (Noise)', 'Highest Top-5 Validation Accuracy'),
        ('avg_noise_shift_list', 'avg_psnr_noisy_list', 'Average Shift (Noise)', 'Average PSNR (Noise)'),
        ('avg_colour_shift_list', 'highest_val_acc_list', 'Average Shift (Colour)', 'Highest Validation Accuracy'),
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
                                           ['b', 'g', 'r', 'c', 'm'],
                                           ['Colour Variation Level 0', 'Colour Variation Level 3', 'Colour Variation Level 6', 'Colour Variation Level 8', 'Colour Variation Level 9']):
                x_values = []
                y_values = []

                for sigma in sigmas:
                    data = metric[model_name][sigma]
                    if x_metric == 'sigma':
                        x_values.append(sigma)
                    else:
                        x_values.append(data[x_metric][0])

                    if y_label.endswith('(Relative to Sigma 0.001)'):
                        reference_value = metric[model_name][0.001][y_metric][0]
                        y_values.append(data[y_metric][0] / reference_value)
                    else:
                        y_values.append(data[y_metric][0])

                axs[j].scatter(x_values, y_values, color=color, s=100, label=label) # Adding label here
                axs[j].plot(x_values, y_values, color=color, linewidth=3)

            axs[j].tick_params(axis='both', which='major', labelsize=16)
            axs[j].legend(fontsize=16)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, wspace=0.2)
        plt.show()



def plot_heatmap(metrics_list, model_names, sigmas, metric_key, title):
    colour_variation = [0, 3, 6, 8, 9]

    for model_name in model_names:
        # Create an empty 2D array to store the values
        values = []

        # Iterate through the compression ratios
        for metric in metrics_list:
            row = []
            for sigma in sigmas:
                # Extract the value for this sigma and compression ratio
                value = metric[model_name][sigma][metric_key][0] # Adjust as needed
                row.append(value)
            values.append(row)

        # Convert the 2D array to a Pandas DataFrame
        df = pd.DataFrame(values, index=colour_variation, columns=sigmas)

        # Create the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, annot=True, cmap="YlGnBu")
        plt.xlabel('Sigma')
        plt.ylabel('Colour Variation Level')
        plt.title(f'{title} Heatmap for {model_name}')
        plt.show()


def sensitivity_analysis(metrics_list, model_names, sigmas, metric_key, title):
    colour_variation = [0, 3, 6, 8, 9]

    # Extract the tab20 colormap and make sure you have enough colors for the combinations of models and variations
    colors = plt.cm.tab20(np.linspace(0, 1, len(model_names) * len(colour_variation)))

    plt.figure(figsize=(15, 10))

    color_idx = 0
    for model_name in model_names:
        for idx, metric in enumerate(metrics_list):
            y_values = []
            for sigma in sigmas:
                value = metric[model_name][sigma][metric_key][0]  # Adjust as needed
                y_values.append(value)

            plt.plot(sigmas, y_values, label=f'{model_name} (Colour Variation Level = {colour_variation[idx]})',
                     linewidth=4, color=colors[color_idx])
            color_idx += 1

    plt.xlabel('Sigma', fontsize=18, fontweight='bold')
    plt.ylabel(title, fontsize=18, fontweight='bold')
    plt.title(f'Sensitivity Analysis: {title} vs Sigma', fontsize=20, fontweight='bold')
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.show()
    


def plot_3d_surface(metrics_list, model_names, sigmas, metric_key, title):
    colour_variation = [0, 3, 6, 8, 9]

    for model_name in model_names:
        X, Y, Z = [], [], []
        for compression_ratio, metrics in zip(colour_variation, metrics_list):
            for sigma in sigmas:
                X.append(compression_ratio)
                Y.append(sigma)
                Z.append(metrics[model_name][sigma][metric_key][0])

        # Convert to numpy arrays for manipulation
        X = np.array(X).reshape(len(colour_variation), len(sigmas))
        Y = np.array(Y).reshape(len(colour_variation), len(sigmas))
        Z = np.array(Z).reshape(len(colour_variation), len(sigmas))

        # Define the equation for the plane (modify this to match your specific plane)
        Z_plane = Z.mean() * np.ones_like(Z)

        # Create the 3D surface plot
        fig = go.Figure()
        fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', name='Surface'))
        fig.add_trace(go.Surface(z=Z_plane, x=X, y=Y, colorscale='Greys', opacity=0.5, name='Plane', showscale=False))
        fig.update_layout(
            title=dict(text=f'{title} 3D Surface Plot for {model_name}', font=dict(size=24, color='black', family="Arial, bold")),
            autosize=False,
            width=1000,
            height=800,
            scene=dict(
                xaxis_title='Colour Variation Level',
                yaxis_title='Sigma',
                zaxis_title=title,
                xaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)', showbackground=True, backgroundcolor='rgb(230, 230,230)', title_font=dict(size=18, color='black', family="Arial, bold")),
                yaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)', showbackground=True, backgroundcolor='rgb(230, 230,230)', title_font=dict(size=18, color='black', family="Arial, bold")),
                zaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)', showbackground=True, backgroundcolor='rgb(230, 230,230)', title_font=dict(size=18, color='black', family="Arial, bold"))
            ),
            font=dict(family="Arial, bold", size=16, color="black"),
            coloraxis_colorbar=dict(title='Value', title_font=dict(size=18, family="Arial, bold"), tickfont=dict(size=16, family="Arial, bold"))
        )
        fig.show()


def run_experiment_five_colour_gaussian(data_dir):
    """
    Run the fifth experiment involving colour variations and Gaussian noise.

    The function:
    1. Sets up directory and categories
    2. Evaluates specified models with various levels of colour variations and noise
    3. Generates plots for the models
    4. Plots heatmaps for various metrics
    5. Conducts sensitivity analysis for the models
    
    Parameters:
    - data_dir (str): The directory path containing the original image data.
    
    Returns:
    None (outputs are generated as plots and printed summaries).
    """

    # 1. Set up the directory and categories
    categories = os.listdir(data_dir)
    categories = categories[:20]  # Limiting to the first 20 categories for demonstration

    model_names = ['VGG16', 'MobileNet', 'DenseNet121']
    sigmas = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

    # 2. Evaluate models
    metrics38 = evaluate_models(data_dir, sigmas, model_names, variation_level=0)
    metrics39 = evaluate_models(data_dir, sigmas, model_names, variation_level=3)
    metrics40 = evaluate_models(data_dir, sigmas, model_names, variation_level=6)
    metrics41 = evaluate_models(data_dir, sigmas, model_names, variation_level=8)
    metrics42 = evaluate_models(data_dir, sigmas, model_names, variation_level=9)

    metrics_list = [metrics38, metrics39, metrics40, metrics41, metrics42]

    # 3. Generate plots for models
    for model in model_names:
        print(f'Generating plots for {model}...')
        generate_plots_for_model(model, metrics38, metrics39, metrics40, metrics41, metrics42, sigmas)
        print(f'Plots for {model} generated successfully.\n')

    # 4. Plot heatmaps for various metrics
    plot_heatmap(metrics_list, model_names, sigmas, 'avg_psnr_noisy_list', 'Average PSNR')
    plot_heatmap(metrics_list, model_names, sigmas, 'avg_noise_shift_list', 'Average Shift')
    plot_heatmap(metrics_list, model_names, sigmas, 'highest_val_acc_list', 'Highest Validation Accuracy')
    plot_heatmap(metrics_list, model_names, sigmas, 'highest_val_acc_top5_list', 'Highest Top-5 Validation Accuracy')

    # 5. Sensitivity analysis
    sensitivity_analysis(metrics_list, model_names, sigmas, 'avg_psnr_noisy_list', 'Average PSNR')
    sensitivity_analysis(metrics_list, model_names, sigmas, 'avg_noise_shift_list', 'Average Shift')
    sensitivity_analysis(metrics_list, model_names, sigmas, 'highest_val_acc_list', 'Highest Validation Accuracy')
    sensitivity_analysis(metrics_list, model_names, sigmas, 'highest_val_acc_top5_list', 'Highest Top-5 Validation Accuracy')

