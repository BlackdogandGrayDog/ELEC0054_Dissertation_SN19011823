#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 18:07:48 2023

@author: ericwei
"""

# Standard Library
import os

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import glymur
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# TensorFlow and Keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import TopKCategoricalAccuracy
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, MobileNet, DenseNet121

#%%
def generate_labels_and_images(data_dir, categories, compression_ratio, compression):
    """
    Generates labels and images from a specified directory, and applies a specified compression to each image.

    Parameters:
    - data_dir (str): Path to the data directory.
    - categories (list): List of categories present in the dataset.
    - compression_ratio (int): Compression ratio to be applied on images.
    - compression (str): Type of compression to be applied. Either 'JPEG' or 'JPEG2000'.

    Returns:
    - Tuple containing original images, decompressed images, labels, and file sizes.
    """
    
    original_images = []   # list to store original images
    decompressed_images = []  # list to store decompressed images
    labels = []  # list to store labels
    file_sizes = []  # list to store image sizes
    # total_pixels = 224 * 224  # total pixels in an image

    # Temporary files for saving compressed images
    temp_file_jpeg = "temp_compressed.jpeg"
    temp_file_jpeg2000 = "temp_compressed.jp2"

    # Loop through each category
    for category in categories:
        
        if category == '.DS_Store':
            continue
        
        # Get the list of image files in the category
        image_files = os.listdir(os.path.join(data_dir, category))

        # Loop through each image file
        for image_file in image_files:
            # Open, convert and resize the image
            image = Image.open(os.path.join(data_dir, category, image_file))
            image = image.convert('RGB')
            image = image.resize((224, 224))
            original_image_array = img_to_array(image) / 255.0  # Save the original image array

            # Compress the image using either JPEG or JPEG 2000
            if compression == 'JPEG':
                # Save the image as a JPEG with the specified quality
                image.save(temp_file_jpeg, "JPEG", quality=int(compression_ratio))
                temp_file = temp_file_jpeg  # set the temp file to the JPEG file

            elif compression == 'JPEG2000':
                # Save the image as a JPEG 2000 with the specified compression ratio
                image_array = np.array(image)
                glymur.Jp2k(temp_file_jpeg2000, image_array, cratios=[compression_ratio])
                temp_file = temp_file_jpeg2000  # set the temp file to the JPEG2000 file

            else:
                raise ValueError("Invalid compression type. Choose 'JPEG' or 'JPEG2000'.")

            # Record the size of the compressed image
            file_sizes.append(os.path.getsize(temp_file))

            # Open the compressed image, resulting in decompression
            decompressed_image = Image.open(temp_file)
            decompressed_image_array = img_to_array(decompressed_image) / 255.0  # Save the decompressed image array

            # Add the original and decompressed images and their label to the respective lists
            original_images.append(original_image_array)
            decompressed_images.append(decompressed_image_array)
            labels.append(category)

            # Remove the temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file)

    return original_images, decompressed_images, labels, file_sizes

#%%
def plot_images(image_array, decompressed_image_array):
    """
   Plots original and decompressed images side by side for comparison.

   Parameters:
   - image_array (array): Original image array.
   - decompressed_image_array (array): Decompressed image array.

   Returns:
   - None.
   """
   
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_array)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(decompressed_image_array)
    plt.title('Decompressed Image')
    plt.show()

#%%
def calculate_average_size_and_bpp(file_sizes, total_pixels):
    """
    Calculates the average size (in KB) and bits per pixel (bpp) for a given list of file sizes.

    Parameters:
    - file_sizes (list): List containing the sizes of each file in bytes.
    - total_pixels (int): Total number of pixels in an image.

    Returns:
    - Tuple containing average size in KB and average bpp.
    """
    
    avg_size = np.mean(file_sizes) / 1024  # Convert size in bytes to kilobytes
    avg_bpp = np.mean([(size * 8) / total_pixels for size in file_sizes])  # Calculate average bits per pixel
    return avg_size, avg_bpp

#%%
def calculate_average_psnr(original_images, decompressed_images):
    """
    Calculates the average Peak Signal-to-Noise Ratio (PSNR) between original and decompressed images.

    Parameters:
    - original_images (list): List of original images.
    - decompressed_images (list): List of decompressed images.

    Returns:
    - Average PSNR value.
    """
    
    psnrs = []
    for orig, dec in zip(original_images, decompressed_images):
        orig = orig.astype('float32') / 255.
        dec = dec.astype('float32') / 255.
        psnr = peak_signal_noise_ratio(orig, dec, data_range=1.0)
        psnrs.append(psnr)
    return np.mean(psnrs)

#%%
def image_preprocessing(categories, original_images, decompressed_images, labels, file_sizes):
    """
    Preprocesses the original and decompressed images. Computes PSNR and plots examples. 
    Also splits data into training, validation, and test sets.

    Parameters:
    - categories (list): List of image categories.
    - original_images (list): List of original images.
    - decompressed_images (list): List of decompressed images.
    - labels (list): List of image labels.
    - file_sizes (list): List of image file sizes.

    Returns:
    - Training, validation, and test images and labels, number of classes, class names, average size, 
      average bits per pixel, and average PSNR.
    """
    
    # Plot the first original and decompressed images for example
    plot_images(original_images[0], decompressed_images[0])

    # Calculate and print the average PSNR
    avg_psnr = calculate_average_psnr(original_images, decompressed_images)
    print(f"Average PSNR: {avg_psnr} dB")

    avg_size, avg_bpp = calculate_average_size_and_bpp(file_sizes, 224*224)

    images = np.array(decompressed_images)
    labels = np.array(labels)
    le = LabelEncoder()
    labels_int = le.fit_transform(labels)
    class_names = le.classes_
    print(class_names)
    num_classes=len(categories)
    labels = to_categorical(labels_int, num_classes)
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25, random_state=42)

    return train_images, val_images, test_images, train_labels, val_labels, num_classes, class_names, avg_size, avg_bpp, avg_psnr

#%%
def image_preprocessing_tune(categories, original_images, decompressed_images, labels, file_sizes):
    """
    Preprocesses the original and decompressed images specifically for tuning. Computes PSNR and plots examples. 
    Also splits data into training, validation, and test sets.

    Parameters:
    - categories (list): List of image categories.
    - original_images (list): List of original images.
    - decompressed_images (list): List of decompressed images.
    - labels (list): List of image labels.
    - file_sizes (list): List of image file sizes.

    Returns:
    - Training, validation, and test images and labels, number of classes, class names, average size, 
      average bits per pixel, and average PSNR.
    """
    
    # Plot the first original and decompressed images for example
    plot_images(original_images[0], decompressed_images[0])

    # Calculate and print the average PSNR
    avg_psnr = calculate_average_psnr(original_images, decompressed_images)
    print(f"Average PSNR: {avg_psnr} dB")

    avg_size, avg_bpp = calculate_average_size_and_bpp(file_sizes, 224*224)

    original_images = np.array(original_images)
    decompressed_images = np.array(decompressed_images)
    labels = np.array(labels)
    le = LabelEncoder()
    labels_int = le.fit_transform(labels)
    class_names = le.classes_
    print(class_names)
    num_classes = len(categories)
    labels_one_hot = to_categorical(labels_int, num_classes)

    # Split the original images for training
    train_images, test_images, train_labels, _ = train_test_split(original_images, labels_one_hot, test_size=0.2, random_state=42)
    train_images, val_images, train_labels, _ = train_test_split(train_images, train_labels, test_size=0.25, random_state=42)

    # Split the decompressed images for validation and test
    _, val_images_decompressed, _, val_labels = train_test_split(decompressed_images, labels_one_hot, test_size=0.25, random_state=42)
    _, test_images_decompressed, _, test_labels = train_test_split(decompressed_images, labels_one_hot, test_size=0.2, random_state=42)

    return train_images, val_images_decompressed, test_images_decompressed, train_labels, val_labels, num_classes, class_names, avg_size, avg_bpp, avg_psnr

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
def evaluate_models(data_dir, jpeg_qualities, model_names, compression, tunable, trainable):
    """
    Evaluate a set of models on data compressed at various qualities.
 
    This function evaluates the performance of different models on compressed images, 
    calculating metrics such as accuracy, loss, feature shifts due to compression, etc. 
    It also provides visualizations to understand the effects of compression on the feature space 
    and decision boundaries of models.
 
    Parameters:
    - data_dir (str): Directory containing the data organized into categories.
    - jpeg_qualities (list[int]): List of jpeg compression quality values to be tested.
    - model_names (list[str]): Names of models to be evaluated.
    - compression (str): Compression algorithm/type to be used.
    - tunable (bool): If True, applies tunable preprocessing on images. If False, applies standard preprocessing.
    - trainable (bool): If True, the model will be trained. Otherwise, the model will be used as-is.
 
    Returns:
    - metrics (dict): A dictionary containing various metrics computed for each model and each jpeg quality.
 
    Notes:
    The metrics computed for each model and jpeg quality combination include:
    - Average size (avg_size_list)
    - Bits per pixel (avg_bpp_list)
    - PSNR (avg_psnr_list)
    - Highest validation accuracy (highest_val_acc_list)
    - Lowest validation loss (lowest_val_loss_list)
    - Highest top-5 validation accuracy (highest_val_acc_top5_list)
    - Count of misclassified samples (misclassified_counts_list)
    - Average shift due to compression (avg_shift_list)
    - Most shifted distances (most_shifted_distances)
    - Least shifted distances (least_shifted_distances)
 
    The function prints out a lot of intermediate information, useful for tracking the 
    progress of evaluations, understanding model performance and behavior with compression. 
    It also provides visualization to give insights on decision boundaries and feature shifts.
    """

    categories = os.listdir(data_dir)
    categories = categories[:20]

    metrics = {}

    for model_name in model_names:
        metrics[model_name] = {}
        for quality in jpeg_qualities:
            metrics[model_name][quality] = {
                'avg_size_list': [],
                'avg_bpp_list': [],
                'avg_psnr_list': [],
                'highest_val_acc_list': [],
                'lowest_val_loss_list': [],
                'highest_val_acc_top5_list': [],
                'misclassified_counts_list': [],
                'avg_shift_list': [],
                'most_shifted_distances': [],
                'least_shifted_distances': []
            }

    for quality in jpeg_qualities:
        print(f"Now examining {compression} Quality of {quality}")
        original_images, decompressed_images, labels, file_sizes = generate_labels_and_images(data_dir, categories, compression_ratio = quality, compression = compression)
        if tunable:
          train_images, val_images, test_images, train_labels, val_labels, num_classes, class_names, avg_size, avg_bpp, avg_psnr = image_preprocessing_tune(categories, original_images, decompressed_images, labels, file_sizes)
        else:
          train_images, val_images, test_images, train_labels, val_labels, num_classes, class_names, avg_size, avg_bpp, avg_psnr = image_preprocessing(categories, original_images, decompressed_images, labels, file_sizes)

        # Iterate over each model
        for model_name in model_names:
            print(f"Now examining the model: {model_name}")

            # Construct model
            model = model_construction(model_name, num_classes, trainable)

            # Train the model and plot accuracy/loss
            history = train_plotting(model, train_images, train_labels, val_images, val_labels)

            # Append average size, bpp and psnr to their respective lists
            metrics[model_name][quality]['avg_size_list'].append(avg_size)
            metrics[model_name][quality]['avg_bpp_list'].append(avg_bpp)
            # Check if avg_psnr is 'inf', if so replace it with 100
            if avg_psnr == float('inf'):
                avg_psnr = 100
            metrics[model_name][quality]['avg_psnr_list'].append(avg_psnr)

            # Find the epoch with the highest validation accuracy
            highest_val_acc_epoch = np.argmax(history.history['val_accuracy']) + 1
            highest_val_acc = np.max(history.history['val_accuracy'])
            # Append highest_val_acc to list
            metrics[model_name][quality]['highest_val_acc_list'].append(highest_val_acc)
            # Find the epoch with the lowest validation loss
            lowest_val_loss_epoch = np.argmin(history.history['val_loss']) + 1
            lowest_val_loss = np.min(history.history['val_loss'])
            # Append lowest_val_loss to list
            metrics[model_name][quality]['lowest_val_loss_list'].append(lowest_val_loss)
            print(f"For {model_name}, highest validation accuracy of {highest_val_acc} was achieved at epoch {highest_val_acc_epoch}")
            print(f"For {model_name}, lowest validation loss of {lowest_val_loss} was achieved at epoch {lowest_val_loss_epoch}")

            # Plot top5 accuracy
            highest_val_acc_top5, highest_val_acc_epoch_top5 = plot_top5_accuracy(history)
            # Append highest_val_acc_top5 to list
            metrics[model_name][quality]['highest_val_acc_top5_list'].append(highest_val_acc_top5)
            print(f"For {model_name}, highest validation top 5 accuracy of {highest_val_acc_top5} was achieved at epoch {highest_val_acc_epoch_top5}")

            # Visualise true and predicted labels in feature spaces and use classifier to create decision boundaries
            misclassified_counts, total_misclassified = visualize_feature_space(model, val_images, val_labels, class_names, resolution=500)
            print(misclassified_counts)
            # Append misclassified_counts to list
            metrics[model_name][quality]['misclassified_counts_list'].append(total_misclassified)
            visualize_decision_boundary_knn(model, val_images, val_labels, class_names, resolution=500)
            visualize_decision_boundary_rf(model, val_images, val_labels, class_names, resolution=500)
            visualize_decision_boundary_svm(model, val_images, val_labels, class_names, resolution=500)

            # Compute average shift due to compression
            avg_shift = average_shift(model, original_images, decompressed_images)
            # Append avg_shift to list
            metrics[model_name][quality]['avg_shift_list'].append(avg_shift)
            print(f"Average shift due to compression for model {model_name} is {avg_shift}")

            # Compute the features for all data points
            feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
            original_features_all = feature_model.predict(np.array(original_images))
            compressed_features_all = feature_model.predict(np.array(decompressed_images))
            # Compute the distance (shift due to compression) for each data point
            distances = np.sqrt(np.sum((original_features_all - compressed_features_all)**2, axis=1))


            # Find the index of the data point with the top 5 maximum distance
            most_shifted_indices = np.argsort(distances)[-5:][::-1].tolist()
            print("Top 5 distances between each pair of original and shifted points in high-dimensional space:", distances[most_shifted_indices])
            # Append the maximum distance to the most_shifted_distances list
            metrics[model_name][quality]['most_shifted_distances'].append(distances[most_shifted_indices[0]])
            # Use `most_shifted_indices` to analyze the top 5 most severely shifted data points
            most_shifted_originals = [original_images[i] for i in most_shifted_indices]
            most_shifted_compresseds = [decompressed_images[i] for i in most_shifted_indices]
            # Visualize the decision boundary, original point, and shifted point
            visualize_decision_boundary_svm_shifted_topn(categories, model, np.array(original_images), labels, class_names, most_shifted_originals, most_shifted_compresseds, n_pairs=5, resolution=500, padding=7.0)


            # Find the index of the data point with the top 5 minimum distance
            least_shifted_indices = np.argsort(distances)[:5].tolist()
            print("Top 5 distances between each pair of original and shifted points in high-dimensional space:", distances[least_shifted_indices])
            # Append the minimum distance to the least_shifted_distances list
            metrics[model_name][quality]['least_shifted_distances'].append(distances[least_shifted_indices[0]])
            # Use `least_shifted_indices` to analyze the top 5 least severely shifted data points
            least_shifted_originals = [original_images[i] for i in least_shifted_indices]
            least_shifted_compresseds = [decompressed_images[i] for i in least_shifted_indices]
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
            closest_compresseds = [decompressed_images[i] for i in closest_indices]
            visualize_decision_boundary_svm_top10_closest_and_shifted(categories, model, np.array(original_images), labels_int, class_names, closest_originals, closest_compresseds, resolution=500, padding=15.0)

    return metrics

#%%
def generate_plots_for_model_single(model_name, metrics, jpeg_qualities, use_jpeg2000):
    """
    This function generates a series of comparative plots for a given model based on various metrics. 
    The main aim is to visualize how various JPEG/JPEG2000 qualities (or compression ratios) impact different metrics such as 
    bit-per-pixel, size, PSNR, validation accuracy, etc., for the provided model.

    Parameters:
    - model_name: The name of the neural network model under evaluation.
    - metrics: A dictionary containing lists of metrics for different JPEG/JPEG2000 qualities.
    - jpeg_qualities: A list of JPEG/JPEG2000 qualities (or compression ratios) under evaluation.
    - use_jpeg2000: A boolean that indicates whether JPEG2000 is used. If False, standard JPEG is assumed.

    Returns:
    - None. But it displays the plots.
    """
    
    # Define pairs of metrics to be plotted against each other.
    # Each tuple contains: the x-axis metric, the y-axis metric, the x-axis label, and the y-axis label.
    
    metric_pairs = [
        ('quality', 'avg_bpp_list', 'Quality', 'Average Bit-Per-Pixel (BPP)'),
        ('quality', 'avg_size_list', 'Quality', 'Average Size in Kilobytes (KB)'),
        ('avg_bpp_list', 'avg_size_list', 'Average Bit-Per-Pixel (BPP)', 'Average Size in Kilobytes (KB)'),
        ('avg_bpp_list', 'avg_psnr_list', 'Average Bit-Per-Pixel (BPP)', 'Average PSNR'),
        ('quality', 'avg_psnr_list', 'Quality', 'Average PSNR'),
        ('avg_size_list', 'avg_psnr_list', 'Average Size in Kilobytes (KB)', 'Average PSNR'),
        ('avg_bpp_list', 'highest_val_acc_list', 'Average Bit-Per-Pixel (BPP)', 'Highest Validation Accuracy'),
        ('avg_size_list', 'highest_val_acc_list', 'Average Size in Kilobytes (KB)', 'Highest Validation Accuracy'),
        ('quality', 'highest_val_acc_list', 'Quality', 'Highest Validation Accuracy'),
        ('avg_bpp_list', 'lowest_val_loss_list', 'Average Bit-Per-Pixel (BPP)', 'Lowest Validation Loss'),
        ('avg_bpp_list', 'highest_val_acc_top5_list', 'Average Bit-Per-Pixel (BPP)', 'Highest Top-5 Validation Accuracy'),
        ('avg_bpp_list', 'misclassified_counts_list', 'Average Bit-Per-Pixel (BPP)', 'Misclassified Counts'),
        ('avg_bpp_list', 'avg_shift_list', 'Average Bit-Per-Pixel (BPP)', 'Average Shift'),
        ('avg_bpp_list', 'most_shifted_distances', 'Average Bit-Per-Pixel (BPP)', 'Most Shifted Distances'),
        ('avg_bpp_list', 'least_shifted_distances', 'Average Bit-Per-Pixel (BPP)', 'Least Shifted Distances'),
        ('avg_psnr_list', 'highest_val_acc_list', 'Average PSNR', 'Highest Validation Accuracy'),
        ('quality', 'avg_psnr_list', 'Quality', 'Average PSNR (Relative to Quality 100)'),
        ('quality', 'highest_val_acc_list', 'Quality', 'Highest Validation Accuracy (Relative to Quality 100)'),
    ]
    
    # Loop through each pair of metrics and generate plots.
    for i in range(0, len(metric_pairs), 2):
        fig, axs = plt.subplots(1, 2, figsize=(22, 10))

        for j in range(2):
            x_metric, y_metric, x_label, y_label = metric_pairs[i + j]
            axs[j].grid(True)
            axs[j].set_title(f'{y_label} vs {x_label}', fontsize=18, fontweight='bold')

            if use_jpeg2000 and x_label == 'Quality':
                x_label = 'Compression Ratio'

            axs[j].set_xlabel(x_label, fontsize=18, fontweight='bold')
            axs[j].set_ylabel(y_label, fontsize=18, fontweight='bold')

            x_values = []
            y_values = []
            
            # Extract metric values for each quality.
            for quality in jpeg_qualities:
                data = metrics[model_name][quality]

                if x_metric == 'quality':
                    x_values.append(quality)
                else:
                    x_values.append(data[x_metric][0])

                if y_label.endswith('(Relative to Quality 100)'):
                    reference_quality = 1 if use_jpeg2000 else 100
                    y_values.append(data[y_metric][0] / metrics[model_name][reference_quality][y_metric][0])
                else:
                    y_values.append(data[y_metric][0])

            axs[j].scatter(x_values, y_values, color='b', s=100)
            axs[j].plot(x_values, y_values, color='r', linewidth=3)

            x_ticks = np.arange(min(x_values), max(x_values)+1, (max(x_values) - min(x_values)) / 10)
            axs[j].set_xticks(np.round(x_ticks).astype(int))
            axs[j].tick_params(axis='both', which='major', labelsize=16)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, wspace=0.2)
        plt.show()

#%%
def generate_plots_for_model(model_name, metric1, metric2, metric3, jpeg_qualities, use_jpeg2000):
    """
    This function generates a series of comparative plots for a given model based on various metrics.
    The main aim is to visualize how different JPEG/JPEG2000 qualities (or compression ratios) impact metrics like
    bit-per-pixel, size, PSNR, validation accuracy, etc., for the provided model. This function extends the capability
    to visualize comparisons between three different metrics simultaneously.

    Parameters:
    - model_name: The name of the neural network model under evaluation.
    - metric1, metric2, metric3: Three dictionaries each containing lists of metrics for different JPEG/JPEG2000 qualities.
    - jpeg_qualities: A list of JPEG/JPEG2000 qualities (or compression ratios) under evaluation.
    - use_jpeg2000: A boolean indicating if JPEG2000 is being used. If False, standard JPEG is assumed.

    Returns:
    - None. Plots are displayed.
    """
    # Define pairs of metrics for comparison.
    metric_pairs = [
        ('quality', 'avg_bpp_list', 'Quality', 'Average Bit-Per-Pixel (BPP)'),
        ('quality', 'avg_size_list', 'Quality', 'Average Size in Kilobytes (KB)'),
        ('avg_bpp_list', 'avg_size_list', 'Average Bit-Per-Pixel (BPP)', 'Average Size in Kilobytes (KB)'),
        ('avg_bpp_list', 'avg_psnr_list', 'Average Bit-Per-Pixel (BPP)', 'Average PSNR'),
        ('quality', 'avg_psnr_list', 'Quality', 'Average PSNR'),
        ('avg_size_list', 'avg_psnr_list', 'Average Size in Kilobytes (KB)', 'Average PSNR'),
        ('avg_bpp_list', 'highest_val_acc_list', 'Average Bit-Per-Pixel (BPP)', 'Highest Validation Accuracy'),
        ('avg_size_list', 'highest_val_acc_list', 'Average Size in Kilobytes (KB)', 'Highest Validation Accuracy'),
        ('quality', 'highest_val_acc_list', 'Quality', 'Highest Validation Accuracy'),
        ('avg_bpp_list', 'lowest_val_loss_list', 'Average Bit-Per-Pixel (BPP)', 'Lowest Validation Loss'),
        ('avg_bpp_list', 'highest_val_acc_top5_list', 'Average Bit-Per-Pixel (BPP)', 'Highest Top-5 Validation Accuracy'),
        ('avg_bpp_list', 'misclassified_counts_list', 'Average Bit-Per-Pixel (BPP)', 'Misclassified Counts'),
        ('avg_bpp_list', 'avg_shift_list', 'Average Bit-Per-Pixel (BPP)', 'Average Shift'),
        ('avg_bpp_list', 'most_shifted_distances', 'Average Bit-Per-Pixel (BPP)', 'Most Shifted Distances'),
        ('avg_bpp_list', 'least_shifted_distances', 'Average Bit-Per-Pixel (BPP)', 'Least Shifted Distances'),
        ('avg_psnr_list', 'highest_val_acc_list', 'Average PSNR', 'Highest Validation Accuracy'),
        ('quality', 'avg_psnr_list', 'Quality', 'Average PSNR (Relative to Quality 100)'),
        ('quality', 'highest_val_acc_list', 'Quality', 'Highest Validation Accuracy (Relative to Quality 100)'),
    ]
    
    # Loop through metric pairs and generate the plots.
    for i in range(0, len(metric_pairs), 2):
        fig, axs = plt.subplots(1, 2, figsize=(22, 10))

        for j in range(2):
            x_metric, y_metric, x_label, y_label = metric_pairs[i + j]
            axs[j].grid(True)
            axs[j].set_title(f'{y_label} vs {x_label}', fontsize=18, fontweight='bold')

            if use_jpeg2000 and x_label == 'Quality':
                x_label = 'Compression Ratio'

            axs[j].set_xlabel(x_label, fontsize=18, fontweight='bold')
            axs[j].set_ylabel(y_label, fontsize=18, fontweight='bold')

            # A loop to iterate through the three metrics and plot them
            for metric, color in zip([metric1, metric2, metric3], ['b', 'g', 'r']):
                x_values = []
                y_values = []

                for quality in jpeg_qualities:
                    data = metric[model_name][quality]

                    if x_metric == 'quality':
                        x_values.append(quality)
                    else:
                        x_values.append(data[x_metric][0])

                    if y_label.endswith('(Relative to Quality 100)'):
                        reference_quality = 1 if use_jpeg2000 else 100
                        y_values.append(data[y_metric][0] / metric[model_name][reference_quality][y_metric][0])
                    else:
                        y_values.append(data[y_metric][0])

                axs[j].scatter(x_values, y_values, color=color, s=100)
                axs[j].plot(x_values, y_values, color=color, linewidth=3)

            x_ticks = np.arange(min(x_values), max(x_values)+1, (max(x_values) - min(x_values)) / 10)
            axs[j].set_xticks(np.round(x_ticks).astype(int))
            axs[j].tick_params(axis='both', which='major', labelsize=16)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, wspace=0.2)
        plt.show()
        
#%%
def run_experiment_two(data_dir):
    """
    This function runs an experiment to evaluate multiple models under different conditions of tunability and trainability 
    for both JPEG and JPEG2000 compression formats. Following the evaluation, it generates comparative plots for each model.
    
    Parameters:
    - data_dir: The directory path where the dataset is located.
    
    Returns:
    - None. Plots are displayed.
    """
    
    jpeg_qualities = [0,1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,60,70,80,90,100]
    model_names = ['VGG16', 'VGG19', 'ResNet50', 'MobileNet', 'DenseNet121']

    metrics5 = evaluate_models(data_dir, jpeg_qualities, model_names, 'JPEG', tunable = False, trainable = False)
    metrics6 = evaluate_models(data_dir, jpeg_qualities, model_names, 'JPEG', tunable = False, trainable = True)
    metrics7 = evaluate_models(data_dir, jpeg_qualities, model_names, 'JPEG', tunable = True, trainable = True)

    metrics8 = evaluate_models(data_dir, jpeg_qualities, model_names, 'JPEG2000', tunable = False, trainable = False)
    metrics9 = evaluate_models(data_dir, jpeg_qualities, model_names, 'JPEG2000', tunable = False, trainable = True)
    metrics10 = evaluate_models(data_dir, jpeg_qualities, model_names, 'JPEG2000', tunable = True, trainable = True)

    # Generating plots for JPEG format
    for model in model_names:
        print(f'Generating plots for {model} (JPEG)...')
        generate_plots_for_model_single(model, metrics5, jpeg_qualities, use_jpeg2000 = False)
        print(f'Plots for {model} (JPEG) generated successfully.\n')

    # Generating plots for JPEG2000 format
    for model in model_names:
        print(f'Generating plots for {model} (JPEG2000)...')
        generate_plots_for_model_single(model, metrics8, jpeg_qualities, use_jpeg2000 = True)
        print(f'Plots for {model} (JPEG2000) generated successfully.\n')

    # Generating comparative plots for JPEG format with different settings
    for model in model_names:
        print(f'Generating comparative plots for {model} (JPEG)...')
        generate_plots_for_model(model, metrics5, metrics6, metrics7, jpeg_qualities, use_jpeg2000 = False)
        print(f'Comparative plots for {model} (JPEG) generated successfully.\n')

    # Generating comparative plots for JPEG2000 format with different settings
    for model in model_names:
        print(f'Generating comparative plots for {model} (JPEG2000)...')
        generate_plots_for_model(model, metrics8, metrics9, metrics10, jpeg_qualities, use_jpeg2000 = True)
        print(f'Comparative plots for {model} (JPEG2000) generated successfully.\n')

