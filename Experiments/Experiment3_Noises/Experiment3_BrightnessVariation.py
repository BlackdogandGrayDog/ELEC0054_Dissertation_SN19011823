#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 13:57:13 2023

@author: ericwei
"""

# Standard Libraries
from matplotlib.lines import Line2D
import os

# Third-party Libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
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

# Matplotlib for color mappings
from matplotlib.colors import ListedColormap

#%%
def generate_brightness_variation(image, variation_intensity):
    """
    Modifies the brightness of an image based on the provided variation intensity using the HSV color space.

    Parameters:
    - image (numpy.ndarray): Input RGB image.
    - variation_intensity (float): Multiplier for brightness. Value >1 increases brightness, 0 < Value <1 decreases brightness.

    Returns:
    - numpy.ndarray: Image with adjusted brightness.
    """
    
    # Convert the image to the HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Apply brightness variation to the V channel
    image_hsv[:,:,2] = image_hsv[:,:,2] * variation_intensity

    # Convert back to the RGB color space
    brightened_image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    brightened_image = np.clip(brightened_image, 0, 1)  # Clip values to be in the range [0, 1]
    return brightened_image

#%%
def generate_gradual_brightness_variation(data_dir, categories, num_img, num_iterations, variation_intensity):
    """
    Generates a sequence of images with increasing brightness variations for a specific image from a data directory.
 
    Parameters:
    - data_dir (str): Directory containing the image data.
    - categories (list): Categories of images.
    - num_img (int): Image index for the variation.
    - num_iterations (int): Number of times the brightness variation should be applied.
    - variation_intensity (float): Brightness variation intensity.
 
    Returns:
    - list: List of images with gradually increasing brightness variations.
    """
   
    image_path = os.path.join(data_dir, categories[0], os.listdir(os.path.join(data_dir, categories[0]))[num_img])
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.resize((224, 224))
    original_image_array = img_to_array(image) / 255.0

    images = [original_image_array]
    for i in range(num_iterations): # Start from 0
        brightened_image = generate_brightness_variation(images[-1], variation_intensity) # Apply to the last image in the list
        images.append(brightened_image)
    return images

#%%
def plot_gradual_brightness_variation(images):
    """
    Plots the images with gradual brightness variations.
    
    Parameters:
    - images (list): List of images with brightness variations.
    
    Returns:
    None. The function directly visualizes the plots.
    """
    plt.figure(figsize=(20, 5))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.title(f'Brightness Variation (T = {i})')
    plt.show()

#%%
def generate_original_images(data_dir, categories):
    """
    Generates a list of original images and their labels from the specified data directory.
 
    Parameters:
    - data_dir (str): Directory containing the image data.
    - categories (list): Categories of images.
 
    Returns:
    - list: List of original images.
    - list: List of corresponding labels for the images.
    """
   
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

#%%
def generate_brightness_variation_images(original_images, labels, variation_intensity):
    """
    Applies brightness variation to a list of images.

    Parameters:
    - original_images (list): List of original images.
    - labels (list): List of labels corresponding to the images.
    - variation_intensity (float): Brightness variation intensity.

    Returns:
    - list: Original images.
    - list: Images with brightness variations.
    - list: Labels corresponding to the images.
    """
    
    brightness_variation_images = []

    for original_image in original_images:
        brightened_image = generate_brightness_variation(original_image, variation_intensity)
        brightness_variation_images.append(brightened_image)

    return original_images, brightness_variation_images, labels


#%%
def calculate_average_psnr(original_images, noisy_images):
    """
    Calculates the average Peak Signal-to-Noise Ratio (PSNR) between original images and their noisy versions.
    
    Parameters:
    - original_images (list): List of original images.
    - noisy_images (list): List of noisy/modified images.
    
    Returns:
    - float: Average PSNR value.
    """

    psnrs = []
    for orig, noisy in zip(original_images, noisy_images):
        orig = orig.astype('float32')
        noisy = noisy.astype('float32')
        psnr = peak_signal_noise_ratio(orig, noisy, data_range=1.0)
        psnrs.append(psnr)
    return np.mean(psnrs)

#%%
def image_preprocessing(brightness_variation_images, clean_images, labels):
    """
    Preprocesses the images by splitting them into training, validation, and test sets. Also calculates the average PSNR.
    
    Parameters:
    - brightness_variation_images (list): Images with brightness variations.
    - clean_images (list): Original clean images.
    - labels (list): Labels corresponding to the images.
    
    Returns:
    - Various arrays: Split datasets and other processed data.
    """

    # Calculate and print the average PSNR
    avg_psnr = calculate_average_psnr(np.array(clean_images), np.array(brightness_variation_images))
    print(f"Average PSNR: {avg_psnr} dB")

    images = np.array(brightness_variation_images)
    labels = np.array(labels)
    le = LabelEncoder()
    labels_int = le.fit_transform(labels)
    class_names = le.classes_
    print(class_names)
    num_classes = len(np.unique(labels))
    labels_one_hot = to_categorical(labels_int, num_classes)

    train_images, test_images, train_labels, test_labels = train_test_split(images, labels_one_hot, test_size=0.2, random_state=42)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25, random_state=42)

    return train_images, val_images, test_images, train_labels, val_labels, test_labels, num_classes, class_names, avg_psnr

#%%
def image_preprocessing_tune(brightness_variation_images, clean_images, labels):
    """
    Specially tailored preprocessing for hyperparameter tuning. Splits images differently from the generic preprocessing.
    
    Parameters:
    - brightness_variation_images (list): Images with brightness variations.
    - clean_images (list): Original clean images.
    - labels (list): Labels corresponding to the images.
    
    Returns:
    - Various arrays: Split datasets and other processed data.
    """

    # Calculate and print the average PSNR
    avg_psnr = calculate_average_psnr(np.array(clean_images), np.array(brightness_variation_images))
    print(f"Average PSNR: {avg_psnr} dB")

    original_images = np.array(clean_images)
    brightness_variation_images = np.array(brightness_variation_images)
    labels = np.array(labels)
    le = LabelEncoder()
    labels_int = le.fit_transform(labels)
    class_names = le.classes_
    print(class_names)
    num_classes = len(np.unique(labels))
    labels_one_hot = to_categorical(labels_int, num_classes)

    # Split the original images for training
    train_images, _, train_labels, _ = train_test_split(original_images, labels_one_hot, test_size=0.4, random_state=42)
    train_images, _, train_labels, _ = train_test_split(train_images, train_labels, test_size=0.25, random_state=42)

    # Split the brightness variation images for validation and test
    _, val_images, _, val_labels = train_test_split(brightness_variation_images, labels_one_hot, test_size=0.5, random_state=42)
    _, test_images, _, test_labels = train_test_split(brightness_variation_images, labels_one_hot, test_size=0.5, random_state=42)

    return train_images, val_images, test_images, train_labels, val_labels, test_labels, num_classes, class_names, avg_psnr

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
def visualize_decision_boundary_svm_shifted_topn(categories, model, images, labels, class_names, original_points, shifted_points, n_pairs=5, resolution=500, padding=7.0):
    """
   Visualize decision boundaries of a model in a 2D feature space, alongside the top 'n' original and shifted data points.
   
   Parameters:
       - categories: List of category names
       - model: Trained neural network model
       - images: List of images
       - labels: One-hot encoded labels corresponding to the images
       - class_names: List of class names
       - original_points: List of original data points
       - shifted_points: List of shifted data points
       - n_pairs: Number of top data point pairs to visualize (default is 5)
       - resolution: Resolution for the meshgrid used to visualize the decision boundary
       - padding: Padding added around the data points in the visualization for clarity
   """
   
    # Get the output of the second-to-last layer of the model
    feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)

    # Compute features for all images
    features = feature_model.predict(images)

    # Compute features for the original and shifted points
    original_features = [feature_model.predict(np.expand_dims(point, axis=0)) for point in original_points]
    shifted_features = [feature_model.predict(np.expand_dims(point, axis=0)) for point in shifted_points]

    # Combine all features for t-SNE
    all_features = np.concatenate([features] + original_features + shifted_features)

    # Use t-SNE to reduce dimensionality to 2D for all image features
    tsne = TSNE(n_components=2, random_state=42)
    all_features_2d = tsne.fit_transform(all_features)

    # The 2D coordinates for original_features and shifted_features are the last 2n rows in features_2d
    features_2d = all_features_2d[:-2*len(original_points)]
    original_features_2d = all_features_2d[-2*len(original_points):-len(original_points)]
    shifted_features_2d = all_features_2d[-len(original_points):]

    # Use argmax to convert one-hot encoded labels back to integers
    labels = np.array(labels)
    le = LabelEncoder()
    labels_int = le.fit_transform(labels)
    num_classes=len(categories)
    labels = to_categorical(labels_int, num_classes)
    labels_int = np.argmax(labels, axis=1)

    classifier = SVC(kernel='rbf')
    classifier.fit(features_2d, labels_int)  # We exclude the last two points (original and shifted) when training the classifier

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
        if n == 1:
            # If only 1 pair, find the most shifted one (i.e., the last one in the list)
            original_features_2d = [all_features_2d[-2*n_pairs:-n_pairs][-1]]
            shifted_features_2d = [all_features_2d[-n_pairs:][-1]]
        else:
            # If more than 1 pair, order them in descending order
            original_features_2d = all_features_2d[-2*n_pairs:-n_pairs][::-1][:n]
            shifted_features_2d = all_features_2d[-n_pairs:][::-1][:n]

        # Calculate the distance between each pair of original and shifted points
        distances_2d = [np.sqrt(np.sum((original_2d - shifted_2d)**2)) for original_2d, shifted_2d in zip(original_features_2d, shifted_features_2d)]

        # Print the distance for each pair
        print("Distances between each pair of original and shifted points in 2D space:", distances_2d)

        # Plot the decision boundary
        plt.figure(figsize=(10, 10))
        plt.contourf(xx, yy, Z, alpha=0.8)

        # Pre-define a list of colors for the arrows
        colors = ['red', 'green', 'yellow', 'cyan', 'magenta']

        legend_elements = []

        for i, (original_feature_2d, shifted_feature_2d) in enumerate(zip(original_features_2d, shifted_features_2d)):
            original_class = class_names[classifier.predict(original_feature_2d.reshape(1, -1))[0]]
            shifted_class = class_names[classifier.predict(shifted_feature_2d.reshape(1, -1))[0]]

            plt.scatter(original_feature_2d[0], original_feature_2d[1], color='b', edgecolors='k', s=150)
            plt.scatter(shifted_feature_2d[0], shifted_feature_2d[1], color=colors[i], edgecolors='k', s=150)
            plt.arrow(original_feature_2d[0], original_feature_2d[1], shifted_feature_2d[0] - original_feature_2d[0], shifted_feature_2d[1] - original_feature_2d[1], color=colors[i], width=0.01)

            legend_elements.append(Line2D([0], [0], linestyle="none", marker='o', markersize=10, markerfacecolor='b', label=f'Original - {original_class}'))
            legend_elements.append(Line2D([0], [0], linestyle="none", marker='o', markersize=10, markerfacecolor=colors[i], label=f'Shifted - {shifted_class}'))
            legend_elements.append(Line2D([0], [0], color=colors[i], lw=2, label=f'Arrow - Pair {i+1}'))

        x_min_roi = min(min(pt[0] for pt in original_features_2d), min(pt[0] for pt in shifted_features_2d)) - padding
        x_max_roi = max(max(pt[0] for pt in original_features_2d), max(pt[0] for pt in shifted_features_2d)) + padding
        y_min_roi = min(min(pt[1] for pt in original_features_2d), min(pt[1] for pt in shifted_features_2d)) - padding
        y_max_roi = max(max(pt[1] for pt in original_features_2d), max(pt[1] for pt in shifted_features_2d)) + padding

        plt.xlim(x_min_roi, x_max_roi)
        plt.ylim(y_min_roi, y_max_roi)

        plt.legend(handles=legend_elements, title="Classes", loc="upper right")
        plt.title(f"Decision Boundaries (SVM) - Top {n}")
        plt.show()

#%%
def visualize_decision_boundary_svm_top10_closest_and_shifted(categories, model, images, labels_int, class_names, closest_points, shifted_points, resolution=500, padding=7.0):
    """
    Visualize decision boundaries of an SVM classifier in a 2D feature space, alongside the top 10 closest and shifted data points.
    
    Parameters:
        - categories: List of category names
        - model: Trained neural network model
        - images: List of images
        - labels_int: Integer labels corresponding to the images
        - class_names: List of class names
        - closest_points: List of the top 10 closest data points
        - shifted_points: List of the top 10 shifted data points corresponding to the closest points
        - resolution: Resolution for the meshgrid used to visualize the decision boundary
        - padding: Padding added around the data points in the visualization for clarity
    """
    
    # Get the output of the second-to-last layer of the model
    feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)

    # Compute features for all images
    features = feature_model.predict(images)

    # Compute features for the closest points and shifted points
    closest_features = [feature_model.predict(np.expand_dims(point, axis=0)) for point in closest_points]
    shifted_features = [feature_model.predict(np.expand_dims(point, axis=0)) for point in shifted_points]

    # Combine all features for t-SNE
    all_features = np.concatenate([features] + closest_features + shifted_features)

    # Use t-SNE to reduce dimensionality to 2D for all image features
    tsne = TSNE(n_components=2, random_state=42)
    all_features_2d = tsne.fit_transform(all_features)

    # The 2D coordinates for closest_features and shifted_features are the last 20 rows in features_2d
    features_2d = all_features_2d[:-20]
    closest_features_2d = all_features_2d[-20:-10]
    shifted_features_2d = all_features_2d[-10:]

    classifier = SVC(kernel='rbf')
    classifier.fit(features_2d, labels_int)  # We exclude the last 20 points (closest and shifted) when training the classifier

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

    # Define the region of interest around the shifted points with some padding
    x_min_roi = min(min(pt[0] for pt in closest_features_2d), min(pt[0] for pt in shifted_features_2d)) - padding
    x_max_roi = max(max(pt[0] for pt in closest_features_2d), max(pt[0] for pt in shifted_features_2d)) + padding
    y_min_roi = min(min(pt[1] for pt in closest_features_2d), min(pt[1] for pt in shifted_features_2d)) - padding
    y_max_roi = max(max(pt[1] for pt in closest_features_2d), max(pt[1] for pt in shifted_features_2d)) + padding

    # Set the x and y limits to the region of interest
    plt.xlim(x_min_roi, x_max_roi)
    plt.ylim(y_min_roi, y_max_roi)

    # Define the colors for the shifted data points and arrows
    colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'orange', 'gray']

    # Create a legend handle list
    legend_elements = []

    # Plot the closest points and shifted points
    for i in range(10):
        # Get the class names for closest and shifted features
        closest_class = class_names[classifier.predict(closest_features_2d[i].reshape(1, -1))[0]]
        shifted_class = class_names[classifier.predict(shifted_features_2d[i].reshape(1, -1))[0]]

        # Create a label for this pair of points
        closest_label = f'Pair {i+1}: Original - {closest_class}'
        shifted_label = f'Pair {i+1}: Shifted - {shifted_class}'

        # Plot the closest and shifted points with the pair label
        plt.scatter(closest_features_2d[i, 0], closest_features_2d[i, 1], label=closest_label, color='b', edgecolors='k', s=150)
        plt.scatter(shifted_features_2d[i, 0], shifted_features_2d[i, 1], label=shifted_label, color=colors[i], edgecolors='k', s=150)

        # Draw an arrow from the closest point to the shifted point with different colors
        plt.arrow(closest_features_2d[i, 0], closest_features_2d[i, 1], shifted_features_2d[i, 0] - closest_features_2d[i, 0], shifted_features_2d[i, 1] - closest_features_2d[i, 1], color=colors[i], width=0.01)

        # Add color information to the legend
        legend_elements.append(Line2D([0], [0], linestyle="none", marker='o', markersize=10, markerfacecolor='b', label=closest_label))
        legend_elements.append(Line2D([0], [0], linestyle="none", marker='o', markersize=10, markerfacecolor=colors[i], label=shifted_label))
        legend_elements.append(Line2D([0], [0], color=colors[i], lw=2, label=f'Arrow Pair {i+1}'))

    plt.legend(handles=legend_elements, title="Classes")
    plt.title("Decision Boundaries (SVM)")
    plt.show()

#%%
def evaluate_models(data_dir, variation_intensity, num_iterations, model_names, tunable, trainable):
    '''
    Evaluates multiple models over a series of brightness variation iterations on the given dataset.
    
    The function iteratively introduces brightness variations to a dataset and evaluates how different
    models perform in terms of various metrics like validation accuracy, validation loss, and so on.
    
    Parameters:
    - data_dir (str): Directory containing the data.
    - variation_intensity (float): Intensity of brightness variation.
    - num_iterations (int): Number of brightness variation iterations.
    - model_names (list of str): Names of the models to be evaluated.
    - tunable (bool): Whether image preprocessing is tunable.
    - trainable (bool): Whether the model is trainable.
 
    Returns:
    - dict: A dictionary containing the metrics for each model across the iterations.
    '''
    categories = os.listdir(data_dir)
    categories = categories[:20]

    # Define a list of iteration labels
    iterations = [str(i) for i in range(1, num_iterations + 1)]

    metrics = {}

    # Two loops to create metrics for each model and each iteration
    for model_name in model_names:
        metrics[model_name] = {}
        for iterate in iterations:
            metrics[model_name][iterate] = {
                'avg_psnr_list': [],
                'highest_val_acc_list': [],
                'lowest_val_loss_list': [],
                'highest_val_acc_top5_list': [],
                'misclassified_counts_list': [],
                'avg_shift_list': [],
                'most_shifted_distances': [],
                'least_shifted_distances': []
            }

    original_images, labels = generate_original_images(data_dir, categories)
    clean_images = original_images

    # Loop over the number of iterations for brightness variation
    for iteration in range(num_iterations):
        iterate_label = iterations[iteration]
        print(f"Iteration {iterate_label} for Brightness Variation with Intensity {variation_intensity}")

        original_images, brightness_variation_images, labels = generate_brightness_variation_images(original_images, labels, variation_intensity)

        if tunable:
            train_images, val_images, test_images, train_labels, val_labels, test_labels, num_classes, class_names, avg_psnr = image_preprocessing_tune(brightness_variation_images, clean_images, labels)
        else:
            train_images, val_images, test_images, train_labels, val_labels, test_labels, num_classes, class_names, avg_psnr = image_preprocessing(brightness_variation_images, clean_images, labels)

        # Iterate over each model
        for model_name in model_names:
            print(f"Now examining the model: {model_name}")

            # Construct model
            model = model_construction(model_name, num_classes, trainable)

            # Train the model and plot accuracy/loss
            history = train_plotting(model, train_images, train_labels, val_images, val_labels)

            # Check if avg_psnr is 'inf', if so replace it with 100
            if avg_psnr == float('inf'):
                avg_psnr = 100
            metrics[model_name][iterate_label]['avg_psnr_list'].append(avg_psnr)

            # Find the epoch with the highest validation accuracy
            highest_val_acc_epoch = np.argmax(history.history['val_accuracy']) + 1
            highest_val_acc = np.max(history.history['val_accuracy'])
            # Append highest_val_acc to list
            metrics[model_name][iterate_label]['highest_val_acc_list'].append(highest_val_acc)
            # Find the epoch with the lowest validation loss
            lowest_val_loss_epoch = np.argmin(history.history['val_loss']) + 1
            lowest_val_loss = np.min(history.history['val_loss'])
            # Append lowest_val_loss to list
            metrics[model_name][iterate_label]['lowest_val_loss_list'].append(lowest_val_loss)
            print(f"For {model_name}, highest validation accuracy of {highest_val_acc} was achieved at epoch {highest_val_acc_epoch}")
            print(f"For {model_name}, lowest validation loss of {lowest_val_loss} was achieved at epoch {lowest_val_loss_epoch}")

            # Plot top5 accuracy
            highest_val_acc_top5, highest_val_acc_epoch_top5 = plot_top5_accuracy(history)
            # Append highest_val_acc_top5 to list
            metrics[model_name][iterate_label]['highest_val_acc_top5_list'].append(highest_val_acc_top5)
            print(f"For {model_name}, highest validation top 5 accuracy of {highest_val_acc_top5} was achieved at epoch {highest_val_acc_epoch_top5}")

            # Visualise true and predicted labels in feature spaces and use classifier to create decision boundaries
            misclassified_counts, total_misclassified = visualize_feature_space(model, val_images, val_labels, class_names, resolution=500)
            print(misclassified_counts)
            # Append misclassified_counts to list
            metrics[model_name][iterate_label]['misclassified_counts_list'].append(total_misclassified)
            visualize_decision_boundary_knn(model, val_images, val_labels, class_names, resolution=500)
            visualize_decision_boundary_rf(model, val_images, val_labels, class_names, resolution=500)
            visualize_decision_boundary_svm(model, val_images, val_labels, class_names, resolution=500)

            # Compute average shift due to compression
            avg_shift = average_shift(model, clean_images, brightness_variation_images)
            # Append avg_shift to list
            metrics[model_name][iterate_label]['avg_shift_list'].append(avg_shift)
            print(f"Average shift due to motion blur for model {model_name} is {avg_shift}")

            # Compute the features for all data points
            feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
            original_features_all = feature_model.predict(np.array(clean_images))
            noisy_features_all = feature_model.predict(np.array(brightness_variation_images))
            # Compute the distance (shift due to compression) for each data point
            distances = np.sqrt(np.sum((original_features_all - noisy_features_all)**2, axis=1))


            # Find the index of the data point with the top 5 maximum distance
            most_shifted_indices = np.argsort(distances)[-5:][::-1].tolist()
            print("Top 5 distances between each pair of original and shifted points in high-dimensional space:", distances[most_shifted_indices])
            # Append the maximum distance to the most_shifted_distances list
            metrics[model_name][iterate_label]['most_shifted_distances'].append(distances[most_shifted_indices[0]])
            # Now you can use `most_shifted_indices` to analyze the top 5 most severely shifted data points
            most_shifted_originals = [clean_images[i] for i in most_shifted_indices]
            most_shifted_compresseds = [brightness_variation_images[i] for i in most_shifted_indices]
            # Visualize the decision boundary, original point, and shifted point
            visualize_decision_boundary_svm_shifted_topn(categories, model, np.array(clean_images), labels, class_names, most_shifted_originals, most_shifted_compresseds, n_pairs=5, resolution=500, padding=7.0)


            # Find the index of the data point with the top 5 minimum distance
            least_shifted_indices = np.argsort(distances)[:5].tolist()
            print("Top 5 distances between each pair of original and shifted points in high-dimensional space:", distances[least_shifted_indices])
            # Append the minimum distance to the least_shifted_distances list
            metrics[model_name][iterate_label]['least_shifted_distances'].append(distances[least_shifted_indices[0]])
            # Use `least_shifted_indices` to analyze the top 5 least severely shifted data points
            least_shifted_originals = [clean_images[i] for i in least_shifted_indices]
            least_shifted_compresseds = [brightness_variation_images[i] for i in least_shifted_indices]
            # Visualize the decision boundary, original point, and shifted point for least shifted
            visualize_decision_boundary_svm_shifted_topn(categories, model, np.array(clean_images), labels, class_names, least_shifted_originals, least_shifted_compresseds, n_pairs=5, resolution=500, padding=7.0)


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
            closest_originals = [clean_images[i] for i in closest_indices]
            closest_compresseds = [brightness_variation_images[i] for i in closest_indices]
            visualize_decision_boundary_svm_top10_closest_and_shifted(categories, model, np.array(clean_images), labels_int, class_names, closest_originals, closest_compresseds, resolution=500, padding=15.0)

        # Update original_images to be the blurred_images for the next iteration
        original_images = brightness_variation_images

    return metrics

#%%
def generate_plot_for_model_single(model_name, metric, iterations):
    """
    The `generate_plot_for_model_single` function visualizes relationships between different metrics over 
    specified iterations for a single model configuration.
    
    Features:
    - This function examines metrics such as Average PSNR, Highest Validation Accuracy, etc.
    - It provides both the absolute values and values relative to the first iteration.
    - Metric relationships are plotted over iterations, showing how they evolve as the model progresses.
    
    Parameters:
    - model_name (str): Name of the model for which the plots will be generated.
    - metric (dict): Dictionary containing evaluation metrics of the model for different iterations.
    - iterations (list): List of iteration numbers to be used for the x-axis of the plots.
    
    Returns:
    None. The function displays plots directly.
    """
    
    metric_pairs = [
        ('iterations', 'avg_psnr_list', 'Iterations', 'Average PSNR'),
        ('iterations', 'highest_val_acc_list', 'Iterations', 'Highest Validation Accuracy'),
        ('iterations', 'lowest_val_loss_list', 'Iterations', 'Lowest Validation Loss'),
        ('iterations', 'highest_val_acc_top5_list', 'Iterations', 'Highest Top-5 Validation Accuracy'),
        ('iterations', 'misclassified_counts_list', 'Iterations', 'Misclassified Counts'),
        ('iterations', 'avg_shift_list', 'Iterations', 'Average Shift'),
        ('iterations', 'most_shifted_distances', 'Iterations', 'Most Shifted Distances'),
        ('iterations', 'least_shifted_distances', 'Iterations', 'Least Shifted Distances'),
        ('iterations', 'avg_psnr_list', 'Iterations', 'Average PSNR (Relative to Iteration 1)'),
        ('iterations', 'highest_val_acc_list', 'Iterations', 'Highest Validation Accuracy (Relative to Iteration 1)'),
        ('avg_psnr_list', 'highest_val_acc_list', 'Average PSNR', 'Highest Validation Accuracy'),
        ('avg_psnr_list', 'lowest_val_loss_list', 'Average PSNR', 'Lowest Validation Loss'),
        ('avg_psnr_list', 'highest_val_acc_top5_list', 'Average PSNR', 'Highest Top-5 Validation Accuracy'),
        ('avg_psnr_list', 'avg_shift_list', 'Average PSNR', 'Average Shift'),
    ]

    for i in range(0, len(metric_pairs), 2):
        fig, axs = plt.subplots(1, 2, figsize=(22, 10))

        for j in range(2):
            x_metric, y_metric, x_label, y_label = metric_pairs[i + j]
            axs[j].grid(True)
            axs[j].set_title(f'{y_label} vs {x_label}', fontsize=18, fontweight='bold')
            axs[j].set_xlabel(x_label, fontsize=18, fontweight='bold')
            axs[j].set_ylabel(y_label, fontsize=18, fontweight='bold')

            x_values = []
            y_values = []

            for iterate in iterations:
                data = metric[model_name][iterate]
                if x_metric == 'iterations':
                  x_values.append(int(iterate))
                else:
                  x_values.append(data[x_metric][0])

                if y_label.endswith('(Relative to Iteration 1)'):
                    reference_value = metric[model_name]['1'][y_metric][0]
                    y_values.append(data[y_metric][0] / reference_value)
                else:
                    y_values.append(data[y_metric][0])

            axs[j].scatter(x_values, y_values, color='b', s=100)
            axs[j].plot(x_values, y_values, color='r', linewidth=3)

            x_ticks = np.arange(min(x_values), max(x_values) + 1, (max(x_values) - min(x_values)) / 10)
            axs[j].set_xticks(np.round(x_ticks).astype(int))
            axs[j].tick_params(axis='both', which='major', labelsize=16)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, wspace=0.2)
        plt.show()

#%%
def generate_plot_for_model(model_name, metric1, metric2, metric3, iterations):
    """
    The `generate_plot_for_model` function creates visual plots to compare different metrics across various iterations for a given model. Specifically:
    - The function considers multiple metrics like Average PSNR, Validation Accuracy, etc.
    - Metrics can be plotted in terms of their absolute values or relative to the first iteration (i.e., relative to 'Iteration 1').
    - Allows comparing metrics for different model configurations like pre-trained, noisy, and clean setups.
    
    Parameters:
    - model_name (str): The name of the model under evaluation.
    - metric1 (dict): Metrics for the first model configuration, typically the pre-trained setup.
    - metric2 (dict): Metrics for the second model configuration, which could be using noisy data.
    - metric3 (dict): Metrics for the third model configuration, possibly with clean data.
    - iterations (list): List of iteration numbers under consideration.
    
    Returns:
    None. The function outputs plots directly for visual comparison.
    
    """
    
    metric_pairs = [
        ('iterations', 'avg_psnr_list', 'Iterations', 'Average PSNR'),
        ('iterations', 'highest_val_acc_list', 'Iterations', 'Highest Validation Accuracy'),
        ('iterations', 'lowest_val_loss_list', 'Iterations', 'Lowest Validation Loss'),
        ('iterations', 'highest_val_acc_top5_list', 'Iterations', 'Highest Top-5 Validation Accuracy'),
        ('iterations', 'misclassified_counts_list', 'Iterations', 'Misclassified Counts'),
        ('iterations', 'avg_shift_list', 'Iterations', 'Average Shift'),
        ('iterations', 'most_shifted_distances', 'Iterations', 'Most Shifted Distances'),
        ('iterations', 'least_shifted_distances', 'Iterations', 'Least Shifted Distances'),
        ('iterations', 'avg_psnr_list', 'Iterations', 'Average PSNR (Relative to Iteration 1)'),
        ('iterations', 'highest_val_acc_list', 'Iterations', 'Highest Validation Accuracy (Relative to Iteration 1)'),
        ('avg_psnr_list', 'highest_val_acc_list', 'Average PSNR', 'Highest Validation Accuracy'),
        ('avg_psnr_list', 'lowest_val_loss_list', 'Average PSNR', 'Lowest Validation Loss'),
        ('avg_psnr_list', 'highest_val_acc_top5_list', 'Average PSNR', 'Highest Top-5 Validation Accuracy'),
        ('avg_psnr_list', 'avg_shift_list', 'Average PSNR', 'Average Shift'),
    ]

    for i in range(0, len(metric_pairs), 2):
        fig, axs = plt.subplots(1, 2, figsize=(22, 10))

        for j in range(2):
            x_metric, y_metric, x_label, y_label = metric_pairs[i + j]
            axs[j].grid(True)
            axs[j].set_title(f'{y_label} vs {x_label}', fontsize=18, fontweight='bold')
            axs[j].set_xlabel(x_label, fontsize=18, fontweight='bold')
            axs[j].set_ylabel(y_label, fontsize=18, fontweight='bold')

            for metric, color, label in zip([metric1, metric2, metric3], ['b', 'g', 'r'], ['W_pre-trained', 'W_Noisy', 'W_clean']):
                x_values = []
                y_values = []

                for iteration in iterations:
                    data = metric[model_name][iteration]
                    if x_metric == 'iterations':
                        x_values.append(int(iteration))
                    else:
                        x_values.append(data[x_metric][0])
                    if y_label.endswith('(Relative to Iteration 1)'):
                        reference_value = metric[model_name]['1'][y_metric][0]
                        y_values.append(data[y_metric][0] / reference_value)
                    else:
                        y_values.append(data[y_metric][0])

                axs[j].scatter(x_values, y_values, color=color, s=100, label=label)
                axs[j].plot(x_values, y_values, color=color, linewidth=3)

            # x_ticks = np.arange(min(x_values), max(x_values), (max(x_values) - min(x_values)) / 10)
            axs[j].tick_params(axis='both', which='major', labelsize=16)
            axs[j].legend(fontsize=16)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, wspace=0.2)
        plt.show()

#%%
def run_experiment_three_brightness(data_dir):
    """
    This function conducts three brightness variation experiments, evaluates different model configurations on the 
    given data directory, and generates visual plots for comparison. 
    
    Parameters:
    - data_dir (str): The directory containing the dataset.
    
    Returns:
    None. Directly outputs plots and prints statements indicating the progress of the experiments.
    """
    
    # Brightness variation experiments
    categories = os.listdir(data_dir)
    categories = categories[:20]
    
    plot_gradual_brightness_variation(generate_gradual_brightness_variation(data_dir, categories, num_img=15, num_iterations=4, variation_intensity=1.5))
    plot_gradual_brightness_variation(generate_gradual_brightness_variation(data_dir, categories, num_img=17, num_iterations=4, variation_intensity=1.5))
    plot_gradual_brightness_variation(generate_gradual_brightness_variation(data_dir, categories, num_img=25, num_iterations=4, variation_intensity=1.5))
    
    # Model evaluations
    model_names = ['VGG16', 'VGG19', 'ResNet50', 'MobileNet', 'DenseNet121']
    
    metrics20 = evaluate_models(data_dir, variation_intensity=1.5, num_iterations=10, model_names=model_names, tunable=False, trainable=False)
    metrics21 = evaluate_models(data_dir, variation_intensity=1.5, num_iterations=10, model_names=model_names, tunable=False, trainable=True)
    metrics22 = evaluate_models(data_dir, variation_intensity=1.5, num_iterations=10, model_names=model_names, tunable=True, trainable=True)
    
    # Plot generation using `generate_plot_for_model_single`
    num_iterations = 10
    iterations = [str(i) for i in range(1, num_iterations + 1)]
    for model_name in model_names:
        print(f'Generating plots for {model_name}...')
        generate_plot_for_model_single(model_name, metrics20, iterations)
        print(f'Plots for {model_name} generated successfully.\n')
    
    # Plot generation using `generate_plot_for_model`
    for model_name in model_names:
        print(f'Generating plots for {model_name}...')
        generate_plot_for_model(model_name, metrics20, metrics21, metrics22, iterations)
        print(f'Plots for {model_name} generated successfully.\n')
