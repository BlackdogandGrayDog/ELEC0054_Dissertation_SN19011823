#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 13:12:54 2023

@author: ericwei
"""

# Standard Libraries
from matplotlib.lines import Line2D
import os

# Third-party Libraries
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.cluster import KMeans
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
def apply_color_variation(image, variation_level):
    """
    Apply various color variations to an image.
    
    Parameters:
    - image (np.array): A 3-channel image where each value is in the range [0, 255].
    - variation_level (int): The type of color variation to apply (ranges from 0 to 9).
    
    Returns:
    - color_image (np.array): The modified image.
    """
    
    color_image = image.copy()

    # Level 0: Original image (no variation applied).
    if variation_level == 0:
        return color_image

    # Level 1: Randomly keep only one of the RGB channels, setting others to 0.
    elif variation_level == 1:
        channel_to_keep = random.choice([0, 1, 2])
        color_image[..., [i for i in range(3) if i != channel_to_keep]] = 0

    # Level 2: Randomly remove one of the RGB channels (set to 0).
    elif variation_level == 2:
        channel_to_remove = random.choice([0, 1, 2])
        color_image[..., channel_to_remove] = 0

    # Level 3: Randomly select one RGB channel and set its values to a random constant.
    elif variation_level == 3:
        channel_to_modify = random.choice([0, 1, 2])
        color_image[..., channel_to_modify] = random.randint(0, 255)

    # Level 4: In LAB color space, randomly modify A-channel or B-channel values.
    elif variation_level == 4:
        lab_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2LAB)
        channel_to_modify = random.choice([1, 2])
        lab_image[..., channel_to_modify] = random.randint(0, 255)
        color_image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)

    # Level 5: In HSV color space, modify the saturation values.
    elif variation_level == 5:
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
        hsv_image[..., 1] = random.randint(0, 255)
        color_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    # Level 6: In HSV color space, modify the hue values.
    elif variation_level == 6:
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2HSV)
        hsv_image[..., 0] = random.randint(0, 179)
        color_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    # Level 7: Scale the color values to a smaller range and then back to the original range.
    elif variation_level == 7:
        color_image = color_image * 0.3 + 0.1
        color_image = (color_image * 255).astype(np.uint8)

    # Level 8: Reduce the image to 2 colors using K-means clustering.
    elif variation_level == 8:
        pixels = color_image.reshape((-1, 3))
        kmeans = KMeans(n_clusters=2, n_init=10).fit(pixels)
        labels = kmeans.predict(pixels)
        palette = kmeans.cluster_centers_
        quantized_image = np.zeros_like(pixels)
        for i in range(len(pixels)):
            quantized_image[i] = palette[labels[i]]
        color_image = quantized_image.reshape(color_image.shape)
        color_image = (color_image * 255).astype(np.uint8)

    # Level 9: Invert the RGB channels.
    elif variation_level == 9:
        color_image = np.roll(color_image, shift=1, axis=2)

    return color_image

#%%
def apply_variations_to_dataset(images, variation_level):
    '''
    This function applies color variations to a given dataset of images. The level 
    of variation is determined by the specified variation_level parameter.
    
    Parameters:
    - images (list of arrays): A list of image arrays to which the variations will be applied.
    - variation_level (int): The level of color variation to be applied to the dataset.
    
    Returns:
    - distorted_images (list of arrays): A list of image arrays after the color variations are applied.
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

#%%
def plot_distortions(clean_image):
    '''
   This function displays plots for distorted versions of a given image.
   
   Parameters:
   - clean_image (array): The original image that will be distorted and then plotted.
   
   Returns:
   None.
   '''
   
    fig1, axes1 = plt.subplots(1, 5, figsize=(20, 4))
    fig2, axes2 = plt.subplots(1, 5, figsize=(20, 4))

    # Apply and plot the distortions for levels 1 to 5
    for i in range(0, 5):
        distorted_image = apply_color_variation(clean_image, i)
        axes1[i].imshow(distorted_image)
        axes1[i].set_title(f'Variation Level {i}')
        axes1[i].axis('off')

    # Apply and plot the distortions for levels 6 to 10
    for i in range(5, 10):
        distorted_image = apply_color_variation(clean_image, i)
        axes2[i - 5].imshow(distorted_image)
        axes2[i - 5].set_title(f'Variation Level {i}')
        axes2[i - 5].axis('off')

    plt.show()

#%%
def calculate_average_psnr(original_images, noisy_images):
    '''
    Calculate the average Peak Signal-to-Noise Ratio (PSNR) between the original and noisy images.
    
    Parameters:
    - original_images (list of arrays): List of original image arrays.
    - noisy_images (list of arrays): List of distorted image arrays.
    
    Returns:
    - average_psnr (float): The average PSNR value.
    '''
    
    psnrs = []
    for orig, noisy in zip(original_images, noisy_images):
        orig = orig.astype('float32')
        noisy = noisy.astype('float32')
        psnr = peak_signal_noise_ratio(orig, noisy, data_range=1.0)
        psnrs.append(psnr)
    return np.mean(psnrs)

#%%
def generate_color_variation_images(data_dir, categories, variation_level):
    '''
    This function generates distorted images with color variations based on the given variation level.
    
    Parameters:
    - data_dir (str): Directory containing the dataset.
    - categories (list of str): List of categories to consider from the dataset.
    - variation_level (int): The level of color variation to be applied.
    
    Returns:
    - original_images (array): Array of the original images.
    - noisy_images (array): Array of the distorted images.
    - labels (list of str): List of corresponding labels for each image.
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

#%%
def image_preprocessing(data_dir, categories, variation_level):
    '''
    Preprocess the images with the specified color variation level, then split them 
    into training, validation, and test datasets.
    
    Parameters:
    - data_dir (str): Directory containing the dataset.
    - categories (list of str): List of categories to consider from the dataset.
    - variation_level (int): The level of color variation to be applied.
    
    Returns:
    Multiple arrays representing training, validation, and test datasets, along with 
    other associated data.
    '''
    
    original_images, noisy_images, labels = generate_color_variation_images(data_dir, categories, variation_level)

    # Plot the first original and distorted images for example
    plot_distortions(np.array(original_images)[7])

    # Calculate and print the average PSNR
    avg_psnr = calculate_average_psnr(original_images, noisy_images)
    print(f"Average PSNR: {avg_psnr} dB")

    # Convert labels to integers and then to one-hot encoding
    labels_int = LabelEncoder().fit_transform(labels)
    num_classes = len(categories)
    labels_one_hot = to_categorical(labels_int, num_classes)

    # Split the dataset into training, validation, and test sets
    train_images, test_images, train_labels, test_labels = train_test_split(noisy_images, labels_one_hot, test_size=0.2, random_state=42)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25, random_state=42)

    class_names = categories

    return train_images, val_images, test_images, train_labels, val_labels, num_classes, class_names, original_images, noisy_images, labels, avg_psnr


#%%
def image_preprocessing_tune(data_dir, categories, variation_level):
    '''
    This function is similar to the image_preprocessing function but tunes the distribution 
    of the images for another use case. Images are preprocessed and split differently.
    
    Parameters:
    - data_dir (str): Directory containing the dataset.
    - categories (list of str): List of categories to consider from the dataset.
    - variation_level (int): The level of color variation to be applied.
    
    Returns:
    Multiple arrays representing training, validation, and test datasets, along with 
    other associated data.
    '''
    
    original_images, noisy_images, labels = generate_color_variation_images(data_dir, categories, variation_level)

    # Plot the first original and distorted images for example
    plot_distortions(np.array(original_images)[7])

    # Calculate and print the average PSNR
    avg_psnr = calculate_average_psnr(original_images, noisy_images)
    print(f"Average PSNR: {avg_psnr} dB")

    # Convert labels to integers and then to one-hot encoding
    labels_int = LabelEncoder().fit_transform(labels)
    num_classes = len(categories)
    labels_one_hot = to_categorical(labels_int, num_classes)

    # Convert to NumPy arrays
    original_images = np.array(original_images)
    noisy_images = np.array(noisy_images)

    # Split the original images for training
    train_images, _, train_labels, _ = train_test_split(original_images, labels_one_hot, test_size=0.2, random_state=42)
    train_images, _, train_labels, _ = train_test_split(train_images, train_labels, test_size=0.25, random_state=42)

    # Split the noisy images for validation and test
    _, val_images, _, val_labels = train_test_split(noisy_images, labels_one_hot, test_size=0.25, random_state=42)
    _, test_images, _, test_labels = train_test_split(noisy_images, labels_one_hot, test_size=0.2, random_state=42)

    class_names = categories

    return train_images, val_images, test_images, train_labels, val_labels, num_classes, class_names, original_images, noisy_images, labels, avg_psnr

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
def evaluate_models(data_dir, variation_levels, model_names, tunable, trainable):
    """
    This function automates the process of evaluating different deep learning models under various levels of color variations.
    Specifically, this function:
    - Loads the data, categorizes it, and prepares it based on the desired level of color variation.
    - Trains multiple deep learning models on this data.
    - Computes and captures metrics such as PSNR, validation accuracy, validation loss, and shifts in feature space due to the color variations.
    - Analyzes features of images in high-dimensional space, to assess how color variation affects the feature distributions.
    - Trains SVM on the feature space of original images and finds the closest data points to the decision boundary.
    - Provides visual insights through various plots such as decision boundaries in feature space, misclassified examples, etc.
    
    Parameters:
    - data_dir (str): The directory path containing the data.
    - variation_levels (list): Different levels of color variations to examine.
    - model_names (list): Names of models to evaluate.
    - tunable (bool): Determines if the preprocessing should be tunable.
    - trainable (bool): Flag to decide if the constructed model should be trainable.
    
    Returns:
    - metrics (dict): Dictionary containing evaluation metrics for each model at each variation level.
    """

    categories = os.listdir(data_dir)
    categories = categories[:20]

    metrics = {}

    for model_name in model_names:
        metrics[model_name] = {}
        for variation_level in variation_levels:
            metrics[model_name][variation_level] = {
                'avg_psnr_list': [],
                'highest_val_acc_list': [],
                'lowest_val_loss_list': [],
                'highest_val_acc_top5_list': [],
                'misclassified_counts_list': [],
                'avg_shift_list': [],
                'most_shifted_distances': [],
                'least_shifted_distances': []
            }

    for variation_level in variation_levels:
        print(f"Now examining Color Variation with Level of {variation_level}")
        if tunable:
            train_images, val_images, test_images, train_labels, val_labels, num_classes, class_names, original_images, noisy_images, labels, avg_psnr = image_preprocessing_tune(data_dir, categories, variation_level)
        else:
            train_images, val_images, test_images, train_labels, val_labels, num_classes, class_names, original_images, noisy_images, labels, avg_psnr = image_preprocessing(data_dir, categories, variation_level)

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
            metrics[model_name][variation_level]['avg_psnr_list'].append(avg_psnr)

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

            # Compute average shift due to color variation
            avg_shift = average_shift(model, original_images, noisy_images)
            # Append avg_shift to list
            metrics[model_name][variation_level]['avg_shift_list'].append(avg_shift)
            print(f"Average shift due to color variation for model {model_name} is {avg_shift}")

            # Compute the features for all data points
            feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
            original_features_all = feature_model.predict(np.array(original_images))
            noisy_features_all = feature_model.predict(np.array(noisy_images))
            # Compute the distance (shift due to compression) for each data point
            distances = np.sqrt(np.sum((original_features_all - noisy_features_all)**2, axis=1))


            # Find the index of the data point with the top 5 maximum distance
            most_shifted_indices = np.argsort(distances)[-5:][::-1].tolist()
            print("Top 5 distances between each pair of original and shifted points in high-dimensional space:", distances[most_shifted_indices])
            # Append the maximum distance to the most_shifted_distances list
            metrics[model_name][variation_level]['most_shifted_distances'].append(distances[most_shifted_indices[0]])
            # Use `most_shifted_indices` to analyze the top 5 most severely shifted data points
            most_shifted_originals = [original_images[i] for i in most_shifted_indices]
            most_shifted_compresseds = [noisy_images[i] for i in most_shifted_indices]
            # Visualize the decision boundary, original point, and shifted point
            visualize_decision_boundary_svm_shifted_topn(categories, model, np.array(original_images), labels, class_names, most_shifted_originals, most_shifted_compresseds, n_pairs=5, resolution=500, padding=7.0)


            # Find the index of the data point with the top 5 minimum distance
            least_shifted_indices = np.argsort(distances)[:5].tolist()
            print("Top 5 distances between each pair of original and shifted points in high-dimensional space:", distances[least_shifted_indices])
            # Append the minimum distance to the least_shifted_distances list
            metrics[model_name][variation_level]['least_shifted_distances'].append(distances[least_shifted_indices[0]])
            # Use `least_shifted_indices` to analyze the top 5 least severely shifted data points
            least_shifted_originals = [original_images[i] for i in least_shifted_indices]
            least_shifted_compresseds = [noisy_images[i] for i in least_shifted_indices]
            # Visualize the decision boundary, original point, and shifted point for least shifted
            visualize_decision_boundary_svm_shifted_topn(categories, model, np.array(original_images), labels, class_names, least_shifted_originals, least_shifted_compresseds, n_pairs=5, resolution=500, padding=7.0)


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
            closest_compresseds = [noisy_images[i] for i in closest_indices]
            visualize_decision_boundary_svm_top10_closest_and_shifted(categories, model, np.array(original_images), labels_int, class_names, closest_originals, closest_compresseds, resolution=500, padding=15.0)

    return metrics

#%%
def generate_plot_for_model_single(model_name, metric, variation_levels):
    """
    The `generate_plot_for_model_single` function visualizes the performance metrics of a given model across different levels of color variations.
    Specifically, this function:
    - Generates scatter plots for various metrics, both relative to color variation levels and relative to one another.
    - Plots metrics such as Average PSNR, Validation Accuracy, Misclassification, and Shifts in high dimensional space due to color variations.
    - Creates side-by-side scatter plots to allow for easy comparison and visualization.
 
    Parameters:
    - model_name (str): Name of the model for which metrics need to be visualized.
    - metric (dict): Dictionary containing evaluation metrics for the model at each variation level.
    - variation_levels (list): Different levels of color variations.
 
    """
   
    metric_pairs = [
        ('variation_levels', 'avg_psnr_list', 'Color Variation Levels', 'Average PSNR'),
        ('variation_levels', 'highest_val_acc_list', 'Color Variation Levels', 'Highest Validation Accuracy'),
        ('variation_levels', 'lowest_val_loss_list', 'Color Variation Levels', 'Lowest Validation Loss'),
        ('variation_levels', 'highest_val_acc_top5_list', 'Color Variation Levels', 'Highest Top-5 Validation Accuracy'),
        ('variation_levels', 'misclassified_counts_list', 'Color Variation Levels', 'Misclassified Counts'),
        ('variation_levels', 'avg_shift_list', 'Color Variation Levels', 'Average Shift'),
        ('variation_levels', 'most_shifted_distances', 'Color Variation Levels', 'Most Shifted Distances'),
        ('variation_levels', 'least_shifted_distances', 'Color Variation Levels', 'Least Shifted Distances'),
        ('variation_levels', 'avg_psnr_list', 'Color Variation Levels', 'Average PSNR (Relative to Iteration 1)'),
        ('variation_levels', 'highest_val_acc_list', 'Color Variation Levels', 'Highest Validation Accuracy (Relative to Iteration 1)'),
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

            for variation_level in variation_levels:
                data = metric[model_name][variation_level]
                if x_metric == 'variation_levels':
                  x_values.append(int(variation_level))
                else:
                  x_values.append(data[x_metric][0])

                if y_label.endswith('(Relative to variation level 0)'):
                    reference_value = metric[model_name][0][y_metric][0]
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
def generate_plot_for_model(model_name, metric1, metric2, metric3, variation_levels):
    """
    The `generate_plot_for_model` function creates visual plots to compare different metrics across various color variation levels for a given model.
    Specifically, this function:
    - Iterates over pairs of metrics and plots their relationship.
    - Supports both absolute metric values and those relative to a base value (i.e., iteration 1).
    - Allows comparing metrics for different model configurations (e.g., pre-trained, noisy, clean).
    
    Parameters:
    - model_name (str): Name of the model for which the plots are to be generated.
    - metric1 (dict): Evaluation metrics for the first model configuration (e.g., pre-trained).
    - metric2 (dict): Evaluation metrics for the second model configuration (e.g., with noisy data).
    - metric3 (dict): Evaluation metrics for the third model configuration (e.g., with clean data).
    - variation_levels (list): Different levels of color variations to examine.
    
    Returns:
    None. The function directly visualizes the plots.
    
    """
    
    metric_pairs = [
        ('variation_levels', 'avg_psnr_list', 'Color Variation Levels', 'Average PSNR'),
        ('variation_levels', 'highest_val_acc_list', 'Color Variation Levels', 'Highest Validation Accuracy'),
        ('variation_levels', 'lowest_val_loss_list', 'Color Variation Levels', 'Lowest Validation Loss'),
        ('variation_levels', 'highest_val_acc_top5_list', 'Color Variation Levels', 'Highest Top-5 Validation Accuracy'),
        ('variation_levels', 'misclassified_counts_list', 'Color Variation Levels', 'Misclassified Counts'),
        ('variation_levels', 'avg_shift_list', 'Color Variation Levels', 'Average Shift'),
        ('variation_levels', 'most_shifted_distances', 'Color Variation Levels', 'Most Shifted Distances'),
        ('variation_levels', 'least_shifted_distances', 'Color Variation Levels', 'Least Shifted Distances'),
        ('variation_levels', 'avg_psnr_list', 'Color Variation Levels', 'Average PSNR (Relative to Iteration 1)'),
        ('variation_levels', 'highest_val_acc_list', 'Color Variation Levels', 'Highest Validation Accuracy (Relative to Iteration 1)'),
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

                for variation_level in variation_levels:
                    data = metric[model_name][variation_level]
                    if x_metric == 'variation_levels':
                        x_values.append(variation_level)
                    else:
                        x_values.append(data[x_metric][0])
                    if y_label.endswith('(Relative to variation level 0)'):
                        reference_value = metric[model_name][0.01, 0.01][y_metric][0]
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
def run_experiment_three_colour(data_dir):
    """
    This function conducts the third experiment which compares different neural network architectures 
    on their performance across various color variation levels.
    
    Steps:
    - Lists the first 20 categories from the provided data directory.
    - Evaluates multiple popular architectures such as VGG16, VGG19, ResNet50, MobileNet, and DenseNet121 
      under three different configurations:
      1. Neither tunable nor trainable.
      2. Non-tunable but trainable.
      3. Both tunable and trainable.
    - Visualizes the results using two plotting functions.
    
    Parameters:
    data_dir (str): The directory path where the dataset resides.
    
    Returns:
    None. The function directly outputs the evaluation results and visualizes them.
    """

    # Listing the first 20 categories from the provided data directory
    categories = os.listdir(data_dir)
    categories = categories[:20]

    # Define model names and variation levels
    model_names = ['VGG16', 'VGG19', 'ResNet50', 'MobileNet', 'DenseNet121']
    variation_levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Evaluating models under three configurations
    metrics17 = evaluate_models(data_dir, variation_levels, model_names, tunable = False, trainable = False)
    metrics18 = evaluate_models(data_dir, variation_levels, model_names, tunable = False, trainable = True)
    metrics19 = evaluate_models(data_dir, variation_levels, model_names, tunable = True, trainable = True)

    # Generating plots for each model with a single metric set
    for model_name in model_names:
        print(f'Generating plots for {model_name}...')
        generate_plot_for_model_single(model_name, metrics17, variation_levels)
        print(f'Plots for {model_name} generated successfully.\n')

    # Generating plots comparing all three metric sets
    for model_name in model_names:
        print(f'Generating plots for {model_name}...')
        generate_plot_for_model(model_name, metrics17, metrics18, metrics19, variation_levels)
        print(f'Plots for {model_name} generated successfully.\n')