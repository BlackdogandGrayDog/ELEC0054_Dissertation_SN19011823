#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 21:56:41 2023

@author: ericwei
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
import glymur
import random
import cv2
from sklearn.cluster import KMeans

# Matplotlib for color mappings
from matplotlib.colors import ListedColormap
#%%

def apply_color_variation(image, variation_level):
    ''' 
    This function applies a specific color variation to the provided image based on the given variation level.
    
    Parameters:
    - image: The image to be modified (in RGB format).
    - variation_level: The level of variation to be applied. It ranges from 0 (no change) to 9 (maximum change).
    
    Returns:
    - color_image: The modified image after applying the specified variation.
    '''
   
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

        # Scale the values back to the original range (0 to 255)
        color_image = (color_image * 255).astype(np.uint8)

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
        color_image = (color_image * 255).astype(np.uint8)  # Convert to 8-bit values

    elif variation_level == 9:
        # Novel noise: Inverting the color channels
        color_image = np.roll(color_image, shift=1, axis=2)

    return color_image


def apply_variations_to_dataset(images, variation_level):
    '''
    This function applies color variations to a list of images based on the given variation_level.
    
    Parameters:
    - images: A list of numpy arrays representing the images.
    - variation_level: An integer value indicating the maximum type of variation level to be applied.
  
    Returns:
    - distorted_images: A list of numpy arrays representing the distorted images.
    '''
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
    '''
    This function reads images from a directory, applies color variations, and returns original and noisy images.
    
    Parameters:
    - data_dir: A string representing the path to the directory containing the images.
    - categories: A list of string labels for each category in the dataset.
    - variation_level: An integer value indicating the maximum type of variation level to be applied.

    Returns:
    - original_images, noisy_images, labels: Numpy arrays of the original images, distorted images, and their corresponding labels.
    '''
    
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




def generate_compressed_images_jpeg2000(images, labels, compression_ratio):
    '''
    This function compresses the input images using JPEG2000 compression algorithm and then decompresses them.
    
    Parameters:
    - images: A list of numpy arrays representing the images.
    - labels: A list of string labels for each image.
    - compression_ratio: An integer or float value specifying the compression ratio for JPEG2000.

    Returns:
    - decompressed_images, labels, file_sizes: Lists of decompressed images, labels, and the file sizes of the compressed images.
    '''
    
    decompressed_images = []  # list to store decompressed images
    file_sizes = []  # list to store image sizes

    # Temporary file for saving compressed images
    temp_file_jpeg2000 = "temp_compressed.jp2"

    # Loop through each image
    for image_array in images:
        # Convert the image array to PIL Image
        image = Image.fromarray((image_array * 255).astype(np.uint8))

        # Save the image as a JPEG 2000 with the specified compression ratio
        glymur.Jp2k(temp_file_jpeg2000, np.array(image), cratios=[compression_ratio])

        # Record the size of the compressed image
        file_sizes.append(os.path.getsize(temp_file_jpeg2000))

        # Open the compressed image, resulting in decompression
        decompressed_image = Image.open(temp_file_jpeg2000)
        decompressed_image_array = img_to_array(decompressed_image) / 255.0  # Save the decompressed image array

        # Add the decompressed images to the respective list
        decompressed_images.append(decompressed_image_array)

        # Remove the temporary file
        if os.path.exists(temp_file_jpeg2000):
            os.remove(temp_file_jpeg2000)

    return decompressed_images, labels, file_sizes



def generate_noisy_and_compressed_images(data_dir, categories, variation_level, compression_ratio):
    '''
    This function creates both noisy and compressed versions of images from a dataset.
    
    Parameters:
    - data_dir, categories, variation_level, compression_ratio: 
      As described in the previous functions.

    Returns:
    - original_images, noisy_images, labels, decompressed_images, file_sizes: 
      Numpy arrays/lists of original, noisy, and decompressed images, labels, and file sizes of the compressed images.
    '''
    
    # Generate original images, noisy images, noise samples, and labels using the generate_noisy_images function
    original_images, noisy_images, labels = generate_color_variation_images(data_dir, categories, variation_level)

    # Generate decompressed images, labels, and file sizes using the generate_compressed_images_jpeg2000 function
    decompressed_images, _, file_sizes = generate_compressed_images_jpeg2000(noisy_images, categories, compression_ratio)

    return original_images, noisy_images, labels, decompressed_images, file_sizes



def plot_distortions(clean_image, compression_ratio):
    '''
    This function plots the original, distorted, and compressed versions of an image for different variation levels.
    
    Parameters:
    - clean_image: A numpy array representing the original image.
    - compression_ratio: An integer or float value specifying the compression ratio for JPEG2000.
    '''
    
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))

    # Apply and plot the distortions for levels 0 to 4 (original and compressed)
    for i in range(0, 5):
        distorted_image = apply_color_variation(clean_image, i)  # Apply distortion
        distorted_image_array = np.array(distorted_image)  # Convert to array

        # Compress the distorted image using JPEG2000
        compressed_images, _, _ = generate_compressed_images_jpeg2000([distorted_image_array], [i], compression_ratio)
        compressed_image = compressed_images[0]  # Extract the compressed image

        axes[0, i].imshow(distorted_image)
        axes[0, i].set_title(f'Variation Level {i} (Noisy)')
        axes[0, i].axis('off')

        axes[2, i].imshow(compressed_image)
        axes[2, i].set_title(f'Variation Level {i} (Compressed)')
        axes[2, i].axis('off')

    # Apply and plot the distortions for levels 5 to 9 (original and compressed)
    for i in range(5, 10):
        distorted_image = apply_color_variation(clean_image, i)  # Apply distortion
        distorted_image_array = np.array(distorted_image)  # Convert to array

        # Compress the distorted image using JPEG2000
        compressed_images, _, _ = generate_compressed_images_jpeg2000([distorted_image_array], [i], compression_ratio)
        compressed_image = compressed_images[0]  # Extract the compressed image

        axes[1, i - 5].imshow(distorted_image)
        axes[1, i - 5].set_title(f'Variation Level {i} (Noisy)')
        axes[1, i - 5].axis('off')

        axes[3, i - 5].imshow(compressed_image)
        axes[3, i - 5].set_title(f'Variation Level {i} (Compressed)')
        axes[3, i - 5].axis('off')

    plt.show()




def calculate_average_size_and_bpp(file_sizes, total_pixels):
    '''
   This function calculates the average size and bits per pixel of a set of images.
   
   Parameters:
   - file_sizes: A list of integers representing the sizes of the images in bytes.
   - total_pixels: An integer value representing the total number of pixels in each image.

   Returns:
   - avg_size, avg_bpp: Average size (in KB) and average bits per pixel for the set of images.
   '''
   
    avg_size = np.mean(file_sizes) / 1024  # Convert size in bytes to kilobytes
    avg_bpp = np.mean([(size * 8) / total_pixels for size in file_sizes])  # Calculate average bits per pixel
    return avg_size, avg_bpp



def calculate_average_psnr(original_images, decompressed_images):
    '''
   This function calculates the average Peak Signal-to-Noise Ratio (PSNR) between two sets of images.
   
   Parameters:
   - original_images, decompressed_images: Lists of numpy arrays representing the original and decompressed images.

   Returns:
   - The average PSNR value between the two sets of images.
   '''
   
    psnrs = []
    for orig, dec in zip(original_images, decompressed_images):
        orig = orig.astype('float32') / 255.
        dec = dec.astype('float32') / 255.
        psnr = peak_signal_noise_ratio(orig, dec, data_range=1.0)
        psnrs.append(psnr)
    return np.mean(psnrs)



def image_preprocessing_with_compression(categories, original_images, noisy_images, labels, decompressed_images, file_sizes, compression_ratio):
    '''
 This function combines the operations of the above functions, including image distortion, compression, and visualization. 
 It also handles image preprocessing tasks such as splitting into train, validation, and test sets.
 
 Parameters:
 - categories, original_images, noisy_images, labels, decompressed_images, file_sizes, compression_ratio:
   As described in the previous functions.

 Returns:
 - train_images, val_images, test_images, train_labels, val_labels, num_classes, class_names, avg_psnr_noisy, avg_psnr, avg_size, avg_bpp:
   Processed image datasets, labels, number of classes, class names, and various metrics about the dataset.
 '''
 
    # Plot the first original, noisy, decompressed images, and noise sample for example
    plot_distortions(np.array(original_images)[85], compression_ratio)

    # Calculate and print the average PSNR for noisy images
    avg_psnr_noisy = calculate_average_psnr(original_images, noisy_images)
    print(f"Average PSNR for noisy image: {avg_psnr_noisy} dB")

    # Calculate and print the average PSNR for compressed images
    avg_psnr = calculate_average_psnr(original_images, decompressed_images)
    print(f"Average PSNR for compressed image: {avg_psnr} dB")

    # Calculate and print the average size and bits per pixel
    avg_size, avg_bpp = calculate_average_size_and_bpp(file_sizes, 224 * 224)
    print(f"Average Size: {avg_size} KB")
    print(f"Average Bits Per Pixel: {avg_bpp}")

    images = np.array(decompressed_images)
    labels = np.array(labels)
    le = LabelEncoder()
    labels_int = le.fit_transform(labels)
    class_names = le.classes_
    print(class_names)
    num_classes = len(categories)
    labels = to_categorical(labels_int, num_classes)
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25, random_state=42)

    return train_images, val_images, test_images, train_labels, val_labels, num_classes, class_names, avg_psnr_noisy, avg_psnr, avg_size, avg_bpp

#%%
def model_construction(model_name, num_classes, trainable):
    """
    Constructs the specified pre-trained model and modifies it by adding custom layers.

    Parameters:
    - model_name (str): Name of the pre-trained model.
    - num_classes (int): Number of target classes.
    - trainable (bool): Whether to freeze the base model weights or not.

    Returns:
    - The modified model.
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

#%%
def train_plotting(model, train_images, train_labels, val_images, val_labels):
    """
    Trains the given model on training data, validates it on validation data, and plots the accuracy and loss.
 
    Parameters:
    - model (Keras model): Model to be trained.
    - train_images (array): Training images.
    - train_labels (array): Training labels.
    - val_images (array): Validation images.
    - val_labels (array): Validation labels.
 
    Returns:
    - Training history.
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

#%%
def plot_top5_accuracy(history):
    """
    Plots the top 5 accuracy for training and validation data from training history.

    Parameters:
    - history (Keras History object): Contains training/validation loss and accuracy data.

    Returns:
    - Highest top 5 accuracy and its epoch number.
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

#%%
def visualize_feature_space(model, images, labels, class_names, resolution=100):
    """
    Visualize the feature space of the model using t-SNE.

    Args:
        model (Model): The trained model.
        images (ndarray): The input images.
        labels (ndarray): The true labels.
        class_names (list): List of class names.
        resolution (int, optional): Resolution of the visualization. Defaults to 100.

    Returns:
        dict: Counts of misclassified instances for each class.
    """
    
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

    return misclassified_counts

#%%
def visualize_decision_boundary_knn(model, images, labels, class_names, resolution=500):
    """
   Visualize the decision boundary using a KNN classifier on the model's feature space.

   Args:
       model (Model): The trained model.
       images (ndarray): The input images.
       labels (ndarray): The true labels.
       class_names (list): List of class names.
       resolution (int, optional): Resolution of the decision boundary. Defaults to 500.
   """
   
   
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
    
#%%
def visualize_decision_boundary_rf(model, images, labels, class_names, resolution=500):
    """
   Visualize the decision boundary using a RandomForest classifier on the model's feature space.

   Args:
       model (Model): The trained model.
       images (ndarray): The input images.
       labels (ndarray): The true labels.
       class_names (list): List of class names.
       resolution (int, optional): Resolution of the decision boundary. Defaults to 500.
   """
   
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
        
#%%
def visualize_decision_boundary_svm(model, images, labels, class_names, resolution=500):
    """
    Visualize the decision boundary using an SVM classifier on the model's feature space.

    Args:
        model (Model): The trained model.
        images (ndarray): The input images.
        labels (ndarray): The true labels.
        class_names (list): List of class names.
        resolution (int, optional): Resolution of the decision boundary. Defaults to 500.
    """
    
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

#%%
def average_shift(model, original_images, decompressed_images):
    """
    Compute the average Euclidean distance shift in feature space between the original and decompressed images.
    
    Parameters:
        - model: Trained neural network model
        - original_images: List of original images
        - decompressed_images: List of decompressed images corresponding to the original images
 
    Returns:
        - Average Euclidean distance between original and decompressed image features
   """

    # Extract features
    feature_model = Model(inputs=model.input, outputs=model.layers[-2].output) # Assumes the last layer is the classification layer
    original_features = feature_model.predict(np.array(original_images))
    compressed_features = feature_model.predict(np.array(decompressed_images))

    # Compute Euclidean distance between original and compressed features
    distances = np.sqrt(np.sum((original_features - compressed_features)**2, axis=1))
    # Return average distance
    return np.mean(distances)

#%%
def visualize_decision_boundary_svm_shifted_topn(categories, model, images, labels, class_names, original_points, noisy_points, compressed_points, n_pairs=5, resolution=500, padding=12.0):
    """
    The `visualize_decision_boundary_svm_shifted_topn` function visualizes the decision boundaries of an SVM classifier trained on features from a neural network model.
    Specifically, this function:
    - Extracts the penultimate layer of the neural network to obtain meaningful features for each image.
    - Applies t-SNE to reduce the high-dimensional features to 2D for visualization.
    - Trains an SVM classifier on the 2D t-SNE features.
    - Visualizes decision boundaries of the SVM.
    - Illustrates the shift in points (original, noisy, compressed) on this 2D space with arrows.
    
    Parameters:
    - categories (list): List of categories/classes in the dataset.
    - model (keras.Model): The trained neural network model.
    - images (np.array): Array of images.
    - labels (list): Corresponding labels for each image.
    - class_names (list): Names of the classes.
    - original_points (list): List of original images to highlight in the visualization.
    - noisy_points (list): List of noisy images corresponding to the original ones.
    - compressed_points (list): List of compressed images corresponding to the noisy ones.
    - n_pairs (int, optional): Number of pairs (original-noisy-compressed) to visualize. Defaults to 5.
    - resolution (int, optional): Resolution of the grid for plotting decision boundaries. Defaults to 500.
    - padding (float, optional): Padding around the region of interest in the plot. Defaults to 12.0.
    
    Returns:
    None. The function directly visualizes the decision boundaries and points.
    
    """

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
          legend_elements.append(Line2D([0], [0], linestyle="none", marker='o', markersize=10, markerfacecolor=noisy_color, label=f'Noisy - {noisy_class}'))
          legend_elements.append(Line2D([0], [0], linestyle="none", marker='o', markersize=10, markerfacecolor=compressed_color, label=f'Compressed - {compressed_class}'))
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


#%%
def visualize_decision_boundary_svm_top10_closest_and_shifted(categories, model, images, labels_int, class_names, original_points, noisy_points, compressed_points, resolution=500, padding=7.0):
    """
    This function visualizes the decision boundary of a trained SVM classifier over a 2D feature space.
    
    Specifically, the function performs the following:
    1. Projects high-dimensional image features onto a 2D plane using t-SNE.
    2. Trains an SVM classifier on the projected features.
    3. Plots the decision boundaries of the classifier.
    4. Overlays the original, noisy, and compressed data points onto the plot.
    5. Draws arrows indicating the shift from original to noisy and from noisy to compressed points.
    
    Parameters:
    - categories (list): List of category names.
    - model (keras.Model): Pre-trained model used to extract features.
    - images (np.array): Array of image data.
    - labels_int (list): List of integer labels corresponding to the images.
    - class_names (list): List of class names.
    - original_points (list): List of original data points.
    - noisy_points (list): List of noisy data points.
    - compressed_points (list): List of compressed data points.
    - resolution (int, optional): Grid resolution for decision boundary visualization. Default is 500.
    - padding (float, optional): Padding added to feature space before plotting. Default is 7.0.
    
    Returns:
    None. The function directly visualizes the plots.
    
    """
    
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
    classifier.fit(features_2d, labels_int)  # exclude the last 30 points (original, noisy, and compressed) when training the classifier

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
        legend_elements.append(Line2D([0], [0], linestyle="none", marker='o', markersize=10, markerfacecolor=noisy_color, label=f'Noisy - {noisy_class} - Pair {i+1}'))
        legend_elements.append(Line2D([0], [0], linestyle="none", marker='o', markersize=10, markerfacecolor=compressed_colors[i], label=f'Compressed - {compressed_class} - Pair {i+1}'))
        legend_elements.append(Line2D([0], [0], color=arrow_colors[i], lw=2, label=f'Arrow 1 - Pair {i+1}'))
        legend_elements.append(Line2D([0], [0], color=arrow_colors[i+1], lw=2, label=f'Arrow 2 - Pair {i+1}'))

    plt.legend(handles=legend_elements, title="Classes")
    plt.title("Decision Boundaries (SVM)")
    plt.show()

#%%
def evaluate_models(data_dir, variation_levels, compression_ratio, model_names):
    '''
    Evaluate multiple neural network models against images affected by varying levels of color variation, noise, and compression. 
  
    Given a directory of images, this function introduces different levels of color variations, noise, and then compresses the images. 
    After this preprocessing step, various neural network models are trained on these images. Various metrics such as accuracy, 
    loss, Peak Signal-to-Noise Ratio (PSNR), feature shift, and distances in the feature space are computed and collected for further analysis.
    
    Parameters:
    - data_dir (str): Directory containing the images to be used for the evaluation.
    - variation_levels (list): List of color variation levels to be applied to the images.
    - compression_ratio (float): Compression ratio to apply to the images.
    - model_names (list): List of neural network model names to be evaluated.
    
    Returns:
    - dict: A dictionary with model names as keys. Each model entry contains a sub-dictionary for each variation level, 
            which further includes lists for all the computed metrics.
    '''
    
    categories = os.listdir(data_dir)
    categories = categories[:20]

    metrics = {}

    for model_name in model_names:
        metrics[model_name] = {}
        for variation_level in variation_levels:
            metrics[model_name][variation_level] = {
                'avg_size_list': [],
                'avg_bpp_list': [],
                'avg_psnr_noisy_list': [],
                'avg_psnr_list': [],
                'highest_val_acc_list': [],
                'lowest_val_loss_list': [],
                'highest_val_acc_top5_list': [],
                'misclassified_counts_list': [],
                'avg_shift_list': [],
                'avg_noise_shift_list': [],
                'most_shifted_distances': [],
                'least_shifted_distances': []
            }

    for variation_level in variation_levels:
        print(f"Now examining Color Variation Level {variation_level}")
        original_images, noisy_images, labels, decompressed_images, file_sizes = generate_noisy_and_compressed_images(data_dir, categories, variation_level, compression_ratio)
        train_images, val_images, test_images, train_labels, val_labels, num_classes, class_names, avg_psnr_noisy, avg_psnr, avg_size, avg_bpp = image_preprocessing_with_compression(categories, original_images, noisy_images, labels, decompressed_images, file_sizes, compression_ratio)

        # Iterate over each model
        for model_name in model_names:
            print(f"Now examining the model: {model_name}")

            # Construct model
            model = model_construction(model_name, num_classes, trainable = False)

            # Train the model and plot accuracy/loss
            history = train_plotting(model, train_images, train_labels, val_images, val_labels)

            # Append average size, bpp and psnr to their respective lists
            metrics[model_name][variation_level]['avg_size_list'].append(avg_size)
            metrics[model_name][variation_level]['avg_bpp_list'].append(avg_bpp)

            # Check if avg_psnr is 'inf', if so replace it with 100
            if avg_psnr == float('inf'):
                avg_psnr = 100

            if avg_psnr_noisy == float('inf'):
                avg_psnr_noisy = 100
            metrics[model_name][variation_level]['avg_psnr_list'].append(avg_psnr)
            metrics[model_name][variation_level]['avg_psnr_noisy_list'].append(avg_psnr_noisy)

            # Find the epoch with the highest validation accuracy
            highest_val_acc_epoch = np.argmax(history.history['val_accuracy']) + 1
            highest_val_acc = np.max(history.history['val_accuracy'])
            # Append highest_val_acc to list
            metrics[model_name][variation_level]['highest_val_acc_list'].append(highest_val_acc)
            # Find the epoch with the lowest validation loss
            lowest_val_loss_epoch = np.argmin(history.history['val_loss']) + 1
            lowest_val_loss = np.min(history.history['val_loss'])
            # Append lowest_val_loss to list
            metrics[model_name][variation_level]['lowest_val_loss_list'].append(lowest_val_loss)
            print(f"For {model_name}, highest validation accuracy of {highest_val_acc} was achieved at epoch {highest_val_acc_epoch}")
            print(f"For {model_name}, lowest validation loss of {lowest_val_loss} was achieved at epoch {lowest_val_loss_epoch}")

            # Plot top5 accuracy
            highest_val_acc_top5, highest_val_acc_epoch_top5 = plot_top5_accuracy(history)
            # Append highest_val_acc_top5 to list
            metrics[model_name][variation_level]['highest_val_acc_top5_list'].append(highest_val_acc_top5)
            print(f"For {model_name}, highest validation top 5 accuracy of {highest_val_acc_top5} was achieved at epoch {highest_val_acc_epoch_top5}")

            # Visualise true and predicted labels in feature spaces and use classifier to create decision boundaries
            misclassified_counts, total_misclassified = visualize_feature_space(model, val_images, val_labels, class_names, resolution=500)
            print(misclassified_counts)
            # Append misclassified_counts to list
            metrics[model_name][variation_level]['misclassified_counts_list'].append(total_misclassified)
            visualize_decision_boundary_knn(model, val_images, val_labels, class_names, resolution=500)
            visualize_decision_boundary_rf(model, val_images, val_labels, class_names, resolution=500)
            visualize_decision_boundary_svm(model, val_images, val_labels, class_names, resolution=500)

            # Compute average shift due to gaussian noise
            avg_shift_noisy = average_shift(model, original_images, noisy_images)
            # Append avg_shift to list
            metrics[model_name][variation_level]['avg_noise_shift_list'].append(avg_shift_noisy)
            print(f"Average shift due to gaussian noise for model {model_name} is {avg_shift_noisy}")

            # Compute average shift due to compression
            avg_shift = average_shift(model, original_images, decompressed_images)
            # Append avg_shift to list
            metrics[model_name][variation_level]['avg_shift_list'].append(avg_shift)
            print(f"Average shift due to gaussian noise and compression for model {model_name} is {avg_shift}")


            # Compute the features for all data points
            feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
            original_features_all = feature_model.predict(np.array(original_images))
            noisy_features_all = feature_model.predict(np.array(noisy_images))
            compressed_features_all = feature_model.predict(np.array(decompressed_images))

            # Compute the distance (shift due to noise) for each data point
            distances_noise = np.sqrt(np.sum((original_features_all - noisy_features_all)**2, axis=1))
            # Compute the distance (shift due to compression) for each data point
            distances_compression = np.sqrt(np.sum((noisy_features_all - compressed_features_all)**2, axis=1))
            # Combine the distances to get an overall shift
            overall_distances = distances_noise + distances_compression

            # Find the index of the data point with the top 5 maximum distance
            most_shifted_indices = np.argsort(overall_distances)[-5:][::-1].tolist()
            print("Top 5 distances between each pair of original and shifted points in high-dimensional space:", overall_distances[most_shifted_indices])
            # Append the maximum distance to the most_shifted_distances list
            metrics[model_name][variation_level]['most_shifted_distances'].append(overall_distances[most_shifted_indices[0]])

            # use `most_shifted_indices` to analyze the top 5 most severely shifted data points
            most_shifted_originals = [original_images[i] for i in most_shifted_indices]
            most_shifted_noisys = [noisy_images[i] for i in most_shifted_indices]
            most_shifted_compresseds = [decompressed_images[i] for i in most_shifted_indices]

            # Visualize the decision boundary, original point, noisy point, and compressed point
            visualize_decision_boundary_svm_shifted_topn(categories, model, np.array(original_images), labels, class_names, most_shifted_originals, most_shifted_noisys, most_shifted_compresseds, n_pairs=5, resolution=500, padding=12.0)


            # Find the index of the data point with the top 5 minimum distance
            least_shifted_indices = np.argsort(overall_distances)[:5].tolist()
            print("Top 5 distances between each pair of original and shifted points in high-dimensional space:", overall_distances[least_shifted_indices])
            # Append the minimum distance to the least_shifted_distances list
            metrics[model_name][variation_level]['least_shifted_distances'].append(overall_distances[least_shifted_indices[0]])

            # use `least_shifted_indices` to analyze the top 5 least severely shifted data points
            least_shifted_originals = [original_images[i] for i in least_shifted_indices]
            least_shifted_noisys = [noisy_images[i] for i in least_shifted_indices]
            least_shifted_compresseds = [decompressed_images[i] for i in least_shifted_indices]

            # Visualize the decision boundary, original point, noisy point, and compressed point for least shifted
            visualize_decision_boundary_svm_shifted_topn(categories, model, np.array(original_images), labels, class_names, least_shifted_originals, least_shifted_noisys, least_shifted_compresseds, n_pairs=5, resolution=500, padding=12.0)

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
            # Use `closest_indices` to analyze the 10 points closest to the decision boundary
            closest_originals = [original_images[i] for i in closest_indices]
            closest_noisys = [noisy_images[i] for i in closest_indices]
            closest_compresseds = [decompressed_images[i] for i in closest_indices]
            visualize_decision_boundary_svm_top10_closest_and_shifted(categories, model, np.array(original_images), labels_int, class_names, closest_originals, closest_noisys, closest_compresseds, resolution=500, padding=15.0)

    return metrics

#%%
def generate_plots_for_model(model_name, metrics1, metrics2, metrics3, variation_levels):
    """
   The `generate_plots_for_model` function visualizes the relationships between various performance and quality metrics across specified color variation levels for a given model. The primary utility of this function is to analyze and understand how different metrics, such as Bit-Per-Pixel (BPP) and Peak Signal-to-Noise Ratio (PSNR), change with color variation levels, especially under different compression ratios.
   
   Parameters:
   - model_name (str): Name of the model for which the plots are to be generated.
   - metrics1 (dict): Evaluation metrics under the first compression ratio (e.g., Compression Ratio = 30).
   - metrics2 (dict): Evaluation metrics under the second compression ratio (e.g., Compression Ratio = 60).
   - metrics3 (dict): Evaluation metrics under the third compression ratio (e.g., Compression Ratio = 90).
   - variation_levels (list): Different levels of color variations to investigate.
   
   Returns:
   None. The function directly visualizes the plots using matplotlib.
   
   Usage:
   After running evaluations across different color variation levels and obtaining the metrics, call this function to generate comparative plots for insights into model behavior under varying conditions. Make sure to have the required metrics populated in the dictionaries passed as arguments.

   Note: The metrics dictionaries are expected to have a nested structure where the outer key is the model name, the next level key is the variation level, and then the metrics are stored as key-value pairs within.
   """
    metric_pairs = [
        ('variation_levels', 'avg_bpp_list', 'Color Variation Levels', 'Average Bit-Per-Pixel (BPP)'),
        ('variation_levels', 'avg_size_list', 'Color Variation Levels', 'Average Size in Kilobytes (KB)'),
        ('variation_levels', 'avg_psnr_list', 'Color Variation Levels', 'Average PSNR'),
        ('variation_levels', 'avg_psnr_noisy_list', 'Color Variation Levels', 'Average PSNR (noise)'),
        ('variation_levels', 'highest_val_acc_list', 'Color Variation Levels', 'Highest Validation Accuracy'),
        ('variation_levels', 'lowest_val_loss_list', 'Color Variation Levels', 'Lowest Validation Loss'),
        ('variation_levels', 'highest_val_acc_top5_list', 'Color Variation Levels', 'Highest Top-5 Validation Accuracy'),
        ('variation_levels', 'misclassified_counts_list', 'Color Variation Levels', 'Misclassified Counts'),
        ('variation_levels', 'avg_shift_list', 'Color Variation Levels', 'Average Shift'),
        ('variation_levels', 'avg_noise_shift_list', 'Color Variation Levels', 'Average Shift (Noise)'),
        ('variation_levels', 'most_shifted_distances', 'Color Variation Levels', 'Most Shifted Distances'),
        ('variation_levels', 'least_shifted_distances', 'Color Variation Levels', 'Least Shifted Distances'),
        ('variation_levels', 'avg_psnr_list', 'Color Variation Levels', 'Ave. PSNR (Relative to Variation Level 0)'),
        ('variation_levels', 'highest_val_acc_list', 'Color Variation Levels', 'Val. Acc. (Relative to Variation Level 0)'),
        ('avg_psnr_list', 'highest_val_acc_list', 'Average PSNR', 'Highest Validation Accuracy'),
        ('avg_psnr_list', 'lowest_val_loss_list', 'Average PSNR', 'Lowest Validation Loss'),
        ('avg_psnr_list', 'highest_val_acc_top5_list', 'Average PSNR', 'Highest Top-5 Validation Accuracy'),
        ('avg_psnr_list', 'avg_shift_list', 'Average PSNR', 'Average Shift'),
        ('avg_bpp_list', 'avg_psnr_list', 'Average Bit-Per-Pixel (BPP)', 'Average PSNR'),
        ('avg_size_list', 'avg_psnr_list', 'Average Size in Kilobytes (KB)', 'Average PSNR'),
        ('avg_shift_list', 'highest_val_acc_list', 'Average Shift', 'Highest Validation Accuracy'),
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

            for metric, color, label in zip([metrics1, metrics2, metrics3], ['b', 'g', 'r'], ['Compression Ratio = 30', 'Compression Ratio = 60', 'Compression Ratio = 90']):
                x_values = []
                y_values = []

                for variation_level in variation_levels:
                    data = metric[model_name][variation_level]
                    if x_metric == 'variation_levels':
                        x_values.append(variation_level)
                    else:
                        x_values.append(data[x_metric][0])
                    if y_label.endswith('(Relative to variation level 0)'):
                        reference_value = metric[model_name][0][y_metric][0]
                        y_values.append(data[y_metric][0] / reference_value)
                    else:
                        y_values.append(data[y_metric][0])

                axs[j].scatter(x_values, y_values, color=color, s=100, label=label)
                axs[j].plot(x_values, y_values, color=color, linewidth=3)

            axs[j].tick_params(axis='both', which='major', labelsize=16)
            axs[j].legend(fontsize=16)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, wspace=0.2)
        plt.show()

#%%
def plot_heatmap(metrics_list, model_names, variation_levels, metric_key, title):
    """
  The `plot_heatmap` function visualizes a heatmap representing the relationship between color variation levels and compression ratios for given models.
  Key Features:
  - Allows for visualizing how different color variation levels impact model metrics across various compression ratios.
  - Generates a heatmap for each model provided in the `model_names` list.
  
  Parameters:
  - metrics_list (list of dicts): A list of metric dictionaries for different scenarios.
  - model_names (list): List of model names for which heatmaps are to be generated.
  - variation_levels (list): Different levels of color variations to examine.
  - metric_key (str): The key in the metric dictionary to extract values for the heatmap.
  - title (str): The title prefix for the heatmap. The model name will be appended to this.
  
  Returns:
  None. The function directly visualizes the heatmap plots.
  
  Usage:
  Call this function, passing the appropriate parameters, after gathering metrics across various scenarios and models.
  It's recommended to use this function in a Jupyter Notebook or IPython environment for optimal visualization.
  """
  
    compression_ratios = [30, 60, 90]

    for model_name in model_names:
        # Create an empty 2D array to store the values
        values = []

        # Iterate through the scenarios
        for metric in metrics_list:
            row = []
            for level in variation_levels:
                # Extract the value for this variation level and scenario
                value = metric[model_name][level][metric_key][0] # Adjust as needed
                row.append(value)
            values.append(row)

        # Convert the 2D array to a Pandas DataFrame
        df = pd.DataFrame(values, index=compression_ratios, columns=variation_levels)

        # Create the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, annot=True, cmap="YlGnBu")
        plt.xlabel('Color Variation Levels')
        plt.ylabel('Compression Ratios')
        plt.title(f'{title} Heatmap for {model_name}')
        plt.show()

#%%
def sensitivity_analysis(metrics_list, model_names, variation_levels, metric_key, title):
    """
   The `sensitivity_analysis` function visualizes the performance sensitivity of various models across different color variation levels. It generates a plot showcasing the chosen metric's value against the color variation levels, while also accounting for different compression ratios.
   
   Parameters:
   - metrics_list (list): A list of dictionaries containing evaluation metrics for different model configurations.
   - model_names (list): Names of the models for which the plots are to be generated.
   - variation_levels (list): Different levels of color variations to examine.
   - metric_key (str): The specific metric key within the dictionary to plot.
   - title (str): The title to display above the plot and on the y-axis.
   
   Returns:
   None. The function directly visualizes the plots.
   
   Usage:
   After obtaining the evaluation metrics for different models and compression ratios, call this function, passing the appropriate parameters, to visualize the sensitivity of models to different color variations.
   """
   
    compression_ratios = [30, 60, 90]

    plt.figure(figsize=(15, 10))

    for model_name in model_names:
        for idx, metric in enumerate(metrics_list):
            y_values = []
            for level in variation_levels:
                value = metric[model_name][level][metric_key][0] # Adjust as needed
                y_values.append(value)

            plt.plot(variation_levels, y_values, label=f'{model_name} (Compression Ratio = {compression_ratios[idx]})', linewidth=4)

    plt.xlabel('Color Variation Levels', fontsize=18, fontweight='bold')
    plt.ylabel(title, fontsize=18, fontweight='bold')
    plt.title(f'Sensitivity Analysis: {title} vs Color Variation Levels', fontsize=20, fontweight='bold')
    plt.legend(fontsize=16, loc='upper right')
    plt.grid(True)
    plt.xticks(variation_levels, fontsize=16, fontweight='bold') # Ensures that x ticks show 0 to 9
    plt.yticks(fontsize=16, fontweight='bold')
    plt.show()

#%%
def plot_3d_surface(metrics_list, model_names, variation_levels, metric_key, title):
    """
    The `plot_3d_surface` function visualizes a 3D surface plot for different metrics across a variety of compression ratios and color variation levels for a given set of models.
    
    This function:
    - Iterates over different models and constructs a 3D plot for each.
    - The X-axis represents the compression ratio, the Y-axis represents color variation levels, and the Z-axis showcases the desired metric.
    - Utilizes the `plotly` library to render interactive 3D plots with aesthetic coloring.
    
    Parameters:
    - metrics_list (list of dict): A list containing evaluation metrics for each compression ratio.
    - model_names (list): List of model names for which the plots are to be generated.
    - variation_levels (list): Different levels of color variations to be plotted.
    - metric_key (str): The specific key corresponding to the metric to be plotted from the metrics dictionary.
    - title (str): Title to be used for the Z-axis and the overall plot.
    
    Returns:
    None. The function directly renders the plots.

    Usage:
    Call this function with the appropriate parameters after gathering metrics from evaluations under various conditions.
    Note: Ensure that the `plotly` library is installed and imported before using this function.
    """
    
    compression_ratios = [30, 60, 90]

    for model_name in model_names:
        X, Y, Z = [], [], []
        for compression_ratio, metrics in zip(compression_ratios, metrics_list):
            for level in variation_levels:
                X.append(compression_ratio)
                Y.append(level)
                Z.append(metrics[model_name][level][metric_key][0])

        # Convert to numpy arrays for manipulation
        X = np.array(X).reshape(len(compression_ratios), len(variation_levels))
        Y = np.array(Y).reshape(len(compression_ratios), len(variation_levels))
        Z = np.array(Z).reshape(len(compression_ratios), len(variation_levels))

        # Create the 3D surface plot
        fig = go.Figure()
        fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', name='Surface'))
        fig.update_layout(
            title=dict(text=f'{title} 3D Surface Plot for {model_name}', font=dict(size=24, color='black', family="Arial, bold")),
            autosize=False,
            width=1000,
            height=800,
            scene=dict(
                xaxis_title='Compression Ratio',
                yaxis_title='Color Variation Levels',
                zaxis_title=title,
                xaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)', showbackground=True, backgroundcolor='rgb(230, 230,230)', title_font=dict(size=18, color='black', family="Arial, bold")),
                yaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)', showbackground=True, backgroundcolor='rgb(230, 230,230)', title_font=dict(size=18, color='black', family="Arial, bold")),
                zaxis=dict(gridcolor='rgb(255, 255, 255)', zerolinecolor='rgb(255, 255, 255)', showbackground=True, backgroundcolor='rgb(230, 230,230)', title_font=dict(size=18, color='black', family="Arial, bold"))
            ),
            font=dict(family="Arial, bold", size=16, color="black"),
            coloraxis_colorbar=dict(title='Value', title_font=dict(size=18, family="Arial, bold"), tickfont=dict(size=16, family="Arial, bold"))
        )
        fig.show()

#%%
def run_experiment_four_colour(data_dir):
    """
    The function `run_experiment_four_colour` automates the evaluation and visualization of different deep learning models against color variations in a dataset.

    This function:
    - Loads the data from a specified directory.
    - Evaluates three deep learning models, VGG16, MobileNet, and DenseNet121, under different color variation levels and compression ratios.
    - Produces various visualization plots to analyze the performance metrics like PSNR, Shift, Validation Accuracy, etc.
    
    Parameters:
    - data_dir (str): Directory path to the dataset.

    Returns:
    None. The function directly prints and renders the plots.

    Usage:
    Call this function and provide the path to your dataset directory.
    """
    
    import os
    
    # Load categories
    categories = os.listdir(data_dir)
    categories = categories[:20]
    model_names = ['VGG16', 'MobileNet', 'DenseNet121']
    variation_levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Evaluate models
    metrics32 = evaluate_models(data_dir, variation_levels, compression_ratio=30, model_names=model_names)
    metrics33 = evaluate_models(data_dir, variation_levels, compression_ratio=60, model_names=model_names)
    metrics34 = evaluate_models(data_dir, variation_levels, compression_ratio=90, model_names=model_names)

    metrics_list = [metrics32, metrics33, metrics34]

    # Generate plots for each model
    for model_name in model_names:
        print(f'Generating plots for {model_name}...')
        generate_plots_for_model(model_name, metrics32, metrics33, metrics34, variation_levels)
        print(f'Plots for {model_name} generated successfully.\n')

    # Generate heatmap plots
    plot_heatmap(metrics_list, model_names, variation_levels, 'avg_psnr_list', 'Average PSNR')
    plot_heatmap(metrics_list, model_names, variation_levels, 'avg_shift_list', 'Average Shift')
    plot_heatmap(metrics_list, model_names, variation_levels, 'highest_val_acc_list', 'Highest Validation Accuracy')
    plot_heatmap(metrics_list, model_names, variation_levels, 'highest_val_acc_top5_list', 'Highest Top-5 Validation Accuracy')

    # Sensitivity analysis
    sensitivity_analysis(metrics_list, model_names, variation_levels, 'avg_psnr_list', 'Average PSNR')
    sensitivity_analysis(metrics_list, model_names, variation_levels, 'avg_shift_list', 'Average Shift')
    sensitivity_analysis(metrics_list, model_names, variation_levels, 'highest_val_acc_list', 'Highest Validation Accuracy')
    sensitivity_analysis(metrics_list, model_names, variation_levels, 'highest_val_acc_top5_list', 'Highest Top-5 Validation Accuracy')

    # 3D surface plots
    plot_3d_surface(metrics_list, model_names, variation_levels, 'avg_psnr_list', 'Average PSNR')
    plot_3d_surface(metrics_list, model_names, variation_levels, 'avg_shift_list', 'Average Shift')
    plot_3d_surface(metrics_list, model_names, variation_levels, 'highest_val_acc_list', 'Highest Validation Accuracy')
    plot_3d_surface(metrics_list, model_names, variation_levels, 'highest_val_acc_top5_list', 'Highest Top-5 Validation Accuracy')
