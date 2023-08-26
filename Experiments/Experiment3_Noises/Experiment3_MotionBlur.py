#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 02:49:09 2023

@author: ericwei
"""

from matplotlib.lines import Line2D
import os

# Third-party Libraries
import numpy as np
import matplotlib.pyplot as plt
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
from scipy.signal import convolve
from scipy.ndimage import rotate

# Matplotlib for color mappings
from matplotlib.colors import ListedColormap

#%%

def generate_motion_blur_noise(image, kernel_size, angle):
    """
   Generates a motion-blurred image using a linear blur kernel.
   
   Parameters:
   - image (numpy.ndarray): Input image.
   - kernel_size (int): Size of the kernel used for blurring.
   - angle (int): Angle of motion blur.

   Returns:
   - tuple: Motion blur kernel and motion-blurred image.
   """
   
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[kernel_size // 2, :] = np.ones(kernel_size)
    kernel = rotate(kernel, angle, reshape=False)
    kernel /= kernel.sum()  # Normalize the kernel

    # Apply convolution to each channel separately
    noisy_image = np.zeros_like(image)
    for i in range(3):  # Assuming 3 channels for RGB
        noisy_image[..., i] = convolve(image[..., i], kernel, mode='same', method='direct')

    noisy_image = np.clip(noisy_image, 0, 1)  # Clip values to be in the range [0, 1]
    return kernel, noisy_image

#%%
def generate_gradual_motion_blur(data_dir, categories, kernel_size, num_iterations=4, angle=45):
    """
    Generates a sequence of gradually blurred images.
    
    Parameters:
    - data_dir (str): Directory containing the dataset.
    - categories (list): List of categories for dataset.
    - kernel_size (int): Size of the kernel used for blurring.
    - num_iterations (int): Number of blurring iterations.
    - angle (int): Angle of motion blur.

    Returns:
    - list: List of blurred images and their kernels.
    """
    
    # Load an example image
    image_path = os.path.join(data_dir, categories[0], os.listdir(os.path.join(data_dir, categories[0]))[20])
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.resize((224, 224))
    original_image_array = img_to_array(image) / 255.0

    images = [original_image_array]
    kernels = []
    for i in range(num_iterations):
        kernel, blurred_image = generate_motion_blur_noise(images[-1], kernel_size, angle)
        images.append(blurred_image)
        kernels.append(kernel)
    return images, kernels


#%%
def plot_gradual_motion_blur(images, kernels):
    """
   Plots blurred images and their corresponding blur kernels.
   
   Parameters:
   - images (list): List of blurred images.
   - kernels (list): List of motion blur kernels.
   """
   
    # Plot Blurred Images
    plt.figure(figsize=(20, 5))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.title(f'Blurred Image (T = {i})')
    plt.show()

    # Plot Kernels
    plt.figure(figsize=(20, 5))
    for i in range(len(kernels)):
        plt.subplot(1, len(kernels), i + 1)
        plt.imshow(kernels[i], cmap='gray')
        plt.title(f'Kernel (T = {i+1})')
    plt.show()


#%%
def generate_original_images(data_dir, categories):
    """
    Load and preprocess original images from the dataset.
    
    Parameters:
    - data_dir (str): Directory containing the dataset.
    - categories (list): List of categories for dataset.

    Returns:
    - tuple: List of original images and their corresponding labels.
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
def generate_blurred_images(original_images, labels, kernel_size, angle):
    """
   Generate motion-blurred images from original images.
   
   Parameters:
   - original_images (list): List of original images.
   - labels (list): Corresponding labels for the images.
   - kernel_size (int): Size of the kernel used for blurring.
   - angle (int): Angle of motion blur.

   Returns:
   - tuple: List of original images, blurred images, kernels, and their corresponding labels.
   """
   
    blurred_images = []
    kernels = []

    for original_image in original_images:
        kernel, blurred_image = generate_motion_blur_noise(original_image, kernel_size, angle)
        blurred_images.append(blurred_image)
        kernels.append(kernel)

    return original_images, blurred_images, kernels, labels



#%%
def calculate_average_psnr(original_images, noisy_images):
    """
   Calculate the average PSNR between original and noisy images.
   
   Parameters:
   - original_images (list): List of original images.
   - noisy_images (list): List of noisy (blurred) images.

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
def image_preprocessing(blurred_images, clean_images, labels):
    """
    Preprocess images for training, validation, and testing.
    
    Parameters:
    - blurred_images (list): List of blurred images.
    - clean_images (list): List of clean (original) images.
    - labels (list): Corresponding labels for the images.

    Returns:
    - tuple: Training, validation, and test images and labels, number of classes, class names, and average PSNR.
    """
    
    # Calculate and print the average PSNR
    avg_psnr = calculate_average_psnr(clean_images, blurred_images)
    print(f"Average PSNR: {avg_psnr} dB")

    images = np.array(blurred_images)
    labels = np.array(labels)
    le = LabelEncoder()
    labels_int = le.fit_transform(labels)
    class_names = le.classes_
    print(class_names)
    num_classes = len(np.unique(labels))
    labels = to_categorical(labels_int, num_classes)
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25, random_state=42)

    return train_images, val_images, test_images, train_labels, val_labels, test_labels, num_classes, class_names, avg_psnr

#%%
def image_preprocessing_tune(blurred_images, clean_images, labels):
    """
    Preprocess images for fine-tuning a model.
    
    Parameters:
    - blurred_images (list): List of blurred images.
    - clean_images (list): List of clean (original) images.
    - labels (list): Corresponding labels for the images.
 
    Returns:
    - tuple: Training, validation, and test images and labels, number of classes, class names, and average PSNR.
   """
   
    # Calculate and print the average PSNR
    avg_psnr = calculate_average_psnr(clean_images, blurred_images)
    print(f"Average PSNR: {avg_psnr} dB")

    original_images = np.array(clean_images)
    blurred_images = np.array(blurred_images)
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

    # Split the blurred images for validation and test
    _, val_images, _, val_labels = train_test_split(blurred_images, labels_one_hot, test_size=0.5, random_state=42)
    _, test_images, _, test_labels = train_test_split(blurred_images, labels_one_hot, test_size=0.5, random_state=42)

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
def evaluate_models(data_dir, kernel_size, angle, num_iterations, model_names, tunable, trainable):
    """
    Evaluates different models on data with motion blur applied iteratively.

    Args:
    - data_dir (str): Directory path containing the data.
    - kernel_size (int): Size of the kernel for motion blur.
    - angle (float): Angle for motion blur.
    - num_iterations (int): Number of times motion blur is applied iteratively.
    - model_names (list): List of model names to evaluate.
    - tunable (bool): Whether to tune the image preprocessing.
    - trainable (bool): Whether the model layers are trainable.

    Returns:
    - dict: Metrics for each model and each iteration.
    """

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
    # Loop over the number of iterations for motion blur
    for iteration in range(num_iterations):
        iterate_label = iterations[iteration]
        print(f"Iteration {iterate_label} for Motion Blur with Kernel Size {kernel_size}, Angle {angle}")

        original_images, blurred_images, kernels, labels = generate_blurred_images(original_images, labels, kernel_size, angle)

        if tunable:
          train_images, val_images, test_images, train_labels, val_labels, test_labels, num_classes, class_names, avg_psnr = image_preprocessing_tune(blurred_images, clean_images, labels)
        else:
          train_images, val_images, test_images, train_labels, val_labels, test_labels, num_classes, class_names, avg_psnr = image_preprocessing(blurred_images, clean_images, labels)

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
            avg_shift = average_shift(model, clean_images, blurred_images)
            # Append avg_shift to list
            metrics[model_name][iterate_label]['avg_shift_list'].append(avg_shift)
            print(f"Average shift due to motion blur for model {model_name} is {avg_shift}")

            # Compute the features for all data points
            feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
            original_features_all = feature_model.predict(np.array(clean_images))
            noisy_features_all = feature_model.predict(np.array(blurred_images))
            # Compute the distance (shift due to compression) for each data point
            distances = np.sqrt(np.sum((original_features_all - noisy_features_all)**2, axis=1))


            # Find the index of the data point with the top 5 maximum distance
            most_shifted_indices = np.argsort(distances)[-5:][::-1].tolist()
            print("Top 5 distances between each pair of original and shifted points in high-dimensional space:", distances[most_shifted_indices])
            # Append the maximum distance to the most_shifted_distances list
            metrics[model_name][iterate_label]['most_shifted_distances'].append(distances[most_shifted_indices[0]])
            # use `most_shifted_indices` to analyze the top 5 most severely shifted data points
            most_shifted_originals = [clean_images[i] for i in most_shifted_indices]
            most_shifted_compresseds = [blurred_images[i] for i in most_shifted_indices]
            # Visualize the decision boundary, original point, and shifted point
            visualize_decision_boundary_svm_shifted_topn(categories, model, np.array(clean_images), labels, class_names, most_shifted_originals, most_shifted_compresseds, n_pairs=5, resolution=500, padding=7.0)


            # Find the index of the data point with the top 5 minimum distance
            least_shifted_indices = np.argsort(distances)[:5].tolist()
            print("Top 5 distances between each pair of original and shifted points in high-dimensional space:", distances[least_shifted_indices])
            # Append the minimum distance to the least_shifted_distances list
            metrics[model_name][iterate_label]['least_shifted_distances'].append(distances[least_shifted_indices[0]])
            # use `least_shifted_indices` to analyze the top 5 least severely shifted data points
            least_shifted_originals = [clean_images[i] for i in least_shifted_indices]
            least_shifted_compresseds = [blurred_images[i] for i in least_shifted_indices]
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
            # use `closest_indices` to analyze the 10 points closest to the decision boundary
            closest_originals = [clean_images[i] for i in closest_indices]
            closest_compresseds = [blurred_images[i] for i in closest_indices]
            visualize_decision_boundary_svm_top10_closest_and_shifted(categories, model, np.array(clean_images), labels_int, class_names, closest_originals, closest_compresseds, resolution=500, padding=15.0)

        # Update original_images to be the blurred_images for the next iteration
        original_images = blurred_images

    return metrics

#%%
def generate_plot_for_model_single(model_name, metric, iterations):
    """
    Generate plots for a single metric for a given model.
    
    Parameters:
    - model_name: (str) The name of the model.
    - metric: (dict) Dictionary containing metrics data for the model.
    - iterations: (list) List of iterations for which metrics are available.
    
    This function will produce a series of plots for the metrics vs. iterations. 
    Each plot shows a specific metric (e.g., average PSNR, validation accuracy, etc.) against the iteration.
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
   Generate plots for three metric dictionaries for a given model.
   
   Parameters:
   - model_name: (str) The name of the model.
   - metric1, metric2, metric3: (dicts) Dictionaries containing metrics data for the model for three scenarios.
   - iterations: (list) List of iterations for which metrics are available.
   
   This function will produce a series of plots for the metrics vs. iterations. 
   Each plot shows a specific metric (e.g., average PSNR, validation accuracy, etc.) against the iteration 
   for three different metric dictionaries (e.g., for three different preprocessing or training scenarios).
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

            for metric, color, label in zip([metric1, metric2, metric3], ['b', 'g', 'r'], ['W_pre-trained', 'W_Blurred', 'W_clean']):
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

                axs[j].scatter(x_values, y_values, color=color, s=100, label=label)
                axs[j].plot(x_values, y_values, color=color, linewidth=3)

            x_ticks = np.arange(min(x_values), max(x_values) + 1, (max(x_values) - min(x_values)) / 10)
            axs[j].set_xticks(np.round(x_ticks).astype(int))
            axs[j].tick_params(axis='both', which='major', labelsize=16)
            axs[j].legend(fontsize=16)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, wspace=0.2)
        plt.show()

#%%
def run_experiment_three_motion(data_dir):
    """
    Run an experiment on motion blur for multiple models.

    Parameters:
    - data_dir: (str) Directory path to the dataset.

    This function:
    1. Loads a subset of categories from the provided directory.
    2. Generates and plots gradually blurred images.
    3. Evaluates multiple models using three different configurations on the blurred images.
    4. Plots the results.
    """
    # Load categories
    categories = os.listdir(data_dir)
    categories = categories[:20]
    
    # Generate and plot blurred images
    images, kernels = generate_gradual_motion_blur(data_dir, categories, kernel_size = 10)
    plot_gradual_motion_blur(images, kernels)

    # List of model names to evaluate
    model_names = ['VGG16', 'VGG19', 'ResNet50', 'MobileNet', 'DenseNet121']
    
    # Evaluate models using different configurations
    metrics11 = evaluate_models(data_dir, kernel_size=10, angle=45, num_iterations=10, model_names=model_names, tunable=False, trainable=False)
    metrics12 = evaluate_models(data_dir, kernel_size=10, angle=45, num_iterations=10, model_names=model_names, tunable=False, trainable=True)
    metrics13 = evaluate_models(data_dir, kernel_size=10, angle=45, num_iterations=10, model_names=model_names, tunable=True, trainable=True)

    # Define the number of iterations and list of iteration strings
    num_iterations = 10
    iterations = [str(i) for i in range(1, num_iterations + 1)]
    
    # Plot results for each model using single metric
    for model_name in model_names:
        print(f'Generating plots for {model_name}...')
        generate_plot_for_model_single(model_name, metrics11, iterations)
        print(f'Plots for {model_name} generated successfully.\n')

    # Plot results for each model using multiple metrics
    for model_name in model_names:
        print(f'Generating plots for {model_name}...')
        generate_plot_for_model(model_name, metrics11, metrics12, metrics13, iterations)
        print(f'Plots for {model_name} generated successfully.\n')

