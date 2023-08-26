#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 20:23:14 2023

@author: ericwei
"""

# Standard Libraries
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
from scipy.signal import convolve
from scipy.ndimage import rotate

# Matplotlib for color mappings
from matplotlib.colors import ListedColormap

#%%
def generate_motion_blur_noise(image, kernel_size, angle):
    '''
    Generate a motion blur noise on the given image.

    Parameters:
    - image (numpy array): The input image on which motion blur is to be applied.
    - kernel_size (int): The size of the convolution kernel.
    - angle (float): The angle at which motion blur is to be applied.

    Returns:
    - kernel (numpy array): The generated motion blur kernel.
    - noisy_image (numpy array): Image with applied motion blur.
    
    Usage:
    kernel, blurred_image = generate_motion_blur_noise(image, 5, 45)
    '''
    
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
def generate_original_images(data_dir, categories):
    '''
    Generate original images and their labels from the given directory.

    Parameters:
    - data_dir (str): The directory where the image datasets are located.
    - categories (list): List of category names (subdirectories).

    Returns:
    - original_images (list): List of loaded and processed images.
    - labels (list): List of labels corresponding to the loaded images.

    Usage:
    images, image_labels = generate_original_images('./dataset', ['cat', 'dog'])
    '''
    
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
    '''
    Generate motion blurred images using the provided original images.

    Parameters:
    - original_images (list): List of original images to be blurred.
    - labels (list): Corresponding labels of the images.
    - kernel_size (int): The size of the convolution kernel for motion blur.
    - angle (float): The angle at which motion blur is to be applied.

    Returns:
    - original_images, blurred_images, kernels, labels

    Usage:
    orig, blurred, kerns, labs = generate_blurred_images(images, labels, 5, 45)
    '''
    
    blurred_images = []
    kernels = []

    for original_image in original_images:
        kernel, blurred_image = generate_motion_blur_noise(original_image, kernel_size, angle)
        blurred_images.append(blurred_image)
        kernels.append(kernel)

    return original_images, blurred_images, kernels, labels


#%%
def generate_compressed_images_jpeg2000(images, labels, compression_ratio):
    '''
    Compress the provided images using JPEG 2000 algorithm and then decompress them.

    Parameters:
    - images (list): List of images to be compressed.
    - labels (list): Labels corresponding to the images.
    - compression_ratio (int): Desired compression ratio for JPEG 2000.

    Returns:
    - decompressed_images, labels, file_sizes

    Usage:
    decomp_imgs, labs, sizes = generate_compressed_images_jpeg2000(images, labels, 10)
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


#%%
def generate_blurred_and_compressed_images(original_images, labels, kernel_size, angle, compression_ratio):
    '''
    Generate motion blurred images and then compress-decompress them using JPEG 2000.
 
    Parameters are same as previous functions combined.
 
    Usage:
    orig, blurred, kerns, labs, decomp, sizes = generate_blurred_and_compressed_images(images, labels, 5, 45, 10)
   '''

    # Generate blurred images and kernels using the generate_blurred_images function
    original_images, blurred_images, kernels, labels = generate_blurred_images(original_images, labels, kernel_size, angle)

    # Generate decompressed images, labels, and file sizes using the generate_compressed_images_jpeg2000 function
    decompressed_images, _, file_sizes = generate_compressed_images_jpeg2000(blurred_images, labels, compression_ratio)

    return original_images, blurred_images, kernels, labels, decompressed_images, file_sizes


#%%
def generate_gradual_motion_blur(data_dir, categories, kernel_size, compression_ratio, num_iterations=4, angle=45):
    '''
    Gradually apply motion blur and compression-decompression to an image.

    Parameters:
    - data_dir (str): Directory where the images are stored.
    - categories (list): List of category names.
    ... (rest of the parameters are similar to previous functions)

    Returns:
    - images, noise_samples, decompressed_images

    Usage:
    imgs, noises, decomp_imgs = generate_gradual_motion_blur('./dataset', ['cat'], 5, 10, 4, 45)
    '''
    
    # Load an example image
    image_path = os.path.join(data_dir, categories[0], os.listdir(os.path.join(data_dir, categories[0]))[20])
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.resize((224, 224))
    original_image_array = img_to_array(image) / 255.0

    images = [original_image_array]
    noise_samples = []
    decompressed_images = [original_image_array]
    for i in range(num_iterations):
        _, blurred_image = generate_motion_blur_noise(images[-1], kernel_size, angle)
        # Compress and decompress the blurred image
        compressed_image, _, _ = generate_compressed_images_jpeg2000([blurred_image], [categories[0]], compression_ratio)
        decompressed_image = compressed_image[0]

        # Calculate noise sample (difference between decompressed and original)
        noise_sample = decompressed_image - images[-1]
        noise_sample = (noise_sample - noise_sample.min()) / (noise_sample.max() - noise_sample.min()) # Normalize to [0, 1]

        # Append to the lists
        images.append(blurred_image)
        noise_samples.append(noise_sample)
        decompressed_images.append(decompressed_image)

    return images, noise_samples, decompressed_images


#%%
def plot_gradual_motion_blur(images, noise_samples, decompressed_images):
    '''
    Plot the images that have been gradually blurred.

    Parameters:
    - images (list): List of gradually blurred images.
    - noise_samples (list): List of noise patterns.
    - decompressed_images (list): List of decompressed images after compression.

    Usage:
    plot_gradual_motion_blur(imgs, noises, decomp_imgs)
    '''
    
    # Plot Blurred Images
    plt.figure(figsize=(20, 5))
    for i in range(len(images)):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.title(f'Blurred Image (T = {i})')
    plt.show()

    # Plot Decompressed Images
    plt.figure(figsize=(20, 5))
    for i in range(len(decompressed_images)):
        plt.subplot(1, len(decompressed_images), i + 1)
        plt.imshow(decompressed_images[i])
        plt.title(f'Decompressed Image (T = {i})')
    plt.show()

    # Plot Noise Samples
    plt.figure(figsize=(20, 5))
    for i in range(len(noise_samples)):
        plt.subplot(1, len(noise_samples), i + 1)
        plt.imshow(noise_samples[i], cmap='gray')
        plt.title(f'Noise Sample (T = {i+1})')
    plt.show()


#%%
def calculate_average_size_and_bpp(file_sizes, total_pixels):
    '''
    Calculate average size in KB and bits per pixel (bpp) for given images.

    Parameters:
    - file_sizes (list): Sizes of image files in bytes.
    - total_pixels (int): Total number of pixels in each image.

    Returns:
    - avg_size, avg_bpp

    Usage:
    average_size, bits_per_pixel = calculate_average_size_and_bpp(sizes, 224*224)
    '''
    
    avg_size = np.mean(file_sizes) / 1024  # Convert size in bytes to kilobytes
    avg_bpp = np.mean([(size * 8) / total_pixels for size in file_sizes])  # Calculate average bits per pixel
    return avg_size, avg_bpp


#%%
def calculate_average_psnr(original_images, decompressed_images):
    '''
    Calculate the average Peak Signal-to-Noise Ratio (PSNR) between original and decompressed images.

    Parameters:
    - original_images (list): List of original images.
    - decompressed_images (list): List of decompressed images.

    Returns:
    - average_psnr

    Usage:
    avg_psnr_value = calculate_average_psnr(orig_images, decomp_images)
    '''
    
    psnrs = []
    for orig, dec in zip(original_images, decompressed_images):
        orig = orig.astype('float32') / 255.
        dec = dec.astype('float32') / 255.
        psnr = peak_signal_noise_ratio(orig, dec, data_range=1.0)
        psnrs.append(psnr)
    return np.mean(psnrs)


#%%
def image_preprocessing_with_compression(categories, clean_images, blurred_images, labels, decompressed_images, file_sizes):
    '''
    Complete image preprocessing pipeline that calculates various metrics and prepares the dataset for training.
    
    Parameters:
    ... (Parameters from the inputs of the function)
    
    Returns:
    - train_images, val_images, test_images, train_labels, val_labels, num_classes, class_names, avg_psnr_noisy, avg_psnr, avg_size, avg_bpp
    
    Usage:
    train_imgs, val_imgs, test_imgs, train_lbls, val_lbls, classes, class_names, avg_psnr_n, avg_psnr_v, avg_size_v, avg_bpp_v = image_preprocessing_with_compression(categories_list, clean, blurred, lbls, decomp, sizes)
    '''


    # Calculate and print the average PSNR for noisy images
    avg_psnr_noisy = calculate_average_psnr(clean_images, blurred_images)
    print(f"Average PSNR for noisy image: {avg_psnr_noisy} dB")

    # Calculate and print the average PSNR for compressed images
    avg_psnr = calculate_average_psnr(clean_images, decompressed_images)
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
def evaluate_models(data_dir, kernel_size, angle, compression_ratio, num_iterations, model_names):
    '''
    This function evaluates a list of models based on their performance against images that undergo iterative motion blur, noise introduction, and compression.
    
    Parameters:
    - data_dir: Directory containing the original image dataset.
    - kernel_size: Size of the kernel used for motion blur.
    - angle: Angle for the motion blur effect.
    - compression_ratio: Ratio for image compression.
    - num_iterations: Number of iterations to apply the motion blur.
    - model_names: List of model names to evaluate.
    
    Returns:
    - metrics: A dictionary containing various evaluation metrics (e.g., PSNR, accuracy) for each model after each iteration of motion blur.
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

    original_images, labels = generate_original_images(data_dir, categories)
    clean_images = original_images
    # Loop over the number of iterations for motion blur
    for iteration in range(num_iterations):
        iterate_label = iterations[iteration]
        print(f"Iteration {iterate_label} for Motion Blur with Kernel Size {kernel_size}, Angle {angle}")

        # Generate noisy and compressed images
        original_images, blurred_images, noise_samples, labels, decompressed_images, file_sizes = generate_blurred_and_compressed_images(original_images, labels, kernel_size, angle, compression_ratio)
        train_images, val_images, test_images, train_labels, val_labels, num_classes, class_names, avg_psnr_noisy, avg_psnr, avg_size, avg_bpp = image_preprocessing_with_compression(categories, clean_images, blurred_images, labels, decompressed_images, file_sizes)

        # Iterate over each model
        for model_name in model_names:
            print(f"Now examining the model: {model_name}")

            # Construct model
            model = model_construction(model_name, num_classes, trainable = False)

            # Train the model and plot accuracy/loss
            history = train_plotting(model, train_images, train_labels, val_images, val_labels)

            # Append average size, bpp and psnr to their respective lists
            metrics[model_name][iterate_label]['avg_size_list'].append(avg_size)
            metrics[model_name][iterate_label]['avg_bpp_list'].append(avg_bpp)

            # Check if avg_psnr is 'inf', if so replace it with 100
            if avg_psnr == float('inf'):
                avg_psnr = 100

            if avg_psnr_noisy == float('inf'):
                avg_psnr_noisy = 100
            metrics[model_name][iterate_label]['avg_psnr_list'].append(avg_psnr)
            metrics[model_name][iterate_label]['avg_psnr_noisy_list'].append(avg_psnr_noisy)

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
            # visualize_decision_boundary_knn(model, val_images, val_labels, class_names, resolution=500)
            # visualize_decision_boundary_rf(model, val_images, val_labels, class_names, resolution=500)
            visualize_decision_boundary_svm(model, val_images, val_labels, class_names, resolution=500)

            # Compute average shift due to gaussian noise
            avg_shift_noisy = average_shift(model, clean_images, blurred_images)
            # Append avg_shift to list
            metrics[model_name][iterate_label]['avg_noise_shift_list'].append(avg_shift_noisy)
            print(f"Average shift due to gaussian noise for model {model_name} is {avg_shift_noisy}")

            # Compute average shift due to compression
            avg_shift = average_shift(model, clean_images, decompressed_images)
            # Append avg_shift to list
            metrics[model_name][iterate_label]['avg_shift_list'].append(avg_shift)
            print(f"Average shift due to gaussian noise and compression for model {model_name} is {avg_shift}")


            # Compute the features for all data points
            feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)
            original_features_all = feature_model.predict(np.array(clean_images))
            noisy_features_all = feature_model.predict(np.array(blurred_images))
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
            metrics[model_name][iterate_label]['most_shifted_distances'].append(overall_distances[most_shifted_indices[0]])

            # use `most_shifted_indices` to analyze the top 5 most severely shifted data points
            most_shifted_originals = [clean_images[i] for i in most_shifted_indices]
            most_shifted_blurs = [blurred_images[i] for i in most_shifted_indices]
            most_shifted_compresseds = [decompressed_images[i] for i in most_shifted_indices]

            # Visualize the decision boundary, original point, noisy point, and compressed point
            visualize_decision_boundary_svm_shifted_topn(categories, model, np.array(clean_images), labels, class_names, most_shifted_originals, most_shifted_blurs, most_shifted_compresseds, n_pairs=5, resolution=500, padding=12.0)


            # Find the index of the data point with the top 5 minimum distance
            least_shifted_indices = np.argsort(overall_distances)[:5].tolist()
            print("Top 5 distances between each pair of original and shifted points in high-dimensional space:", overall_distances[least_shifted_indices])
            # Append the minimum distance to the least_shifted_distances list
            metrics[model_name][iterate_label]['least_shifted_distances'].append(overall_distances[least_shifted_indices[0]])

            # use `least_shifted_indices` to analyze the top 5 least severely shifted data points
            least_shifted_originals = [clean_images[i] for i in least_shifted_indices]
            least_shifted_blurs = [blurred_images[i] for i in least_shifted_indices]
            least_shifted_compresseds = [decompressed_images[i] for i in least_shifted_indices]

            # Visualize the decision boundary, original point, noisy point, and compressed point for least shifted
            visualize_decision_boundary_svm_shifted_topn(categories, model, np.array(clean_images), labels, class_names, least_shifted_originals, least_shifted_blurs, least_shifted_compresseds, n_pairs=5, resolution=500, padding=12.0)

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
            closest_blurs = [blurred_images[i] for i in closest_indices]
            closest_compresseds = [decompressed_images[i] for i in closest_indices]
            visualize_decision_boundary_svm_top10_closest_and_shifted(categories, model, np.array(clean_images), labels_int, class_names, closest_originals, closest_blurs, closest_compresseds, resolution=500, padding=15.0)

        # Update original_images to be the blurred_images for the next iteration
        original_images = blurred_images

    return metrics

#%%
def generate_plots_for_model(model_name, metrics1, metrics2, metrics3, iterations):
    """
    The `generate_plots_for_model` function visualizes multiple plots to compare various metrics over different iterations for a specific model. 
    Key Features:
    - Generates comparative plots between different metrics across various iterations.
    - Enables visualization of both absolute metric values and those relative to a base iteration.
    - Supports comparison of metrics under different compression ratios (e.g., 30, 60, 90).
    
    Parameters:
    - model_name (str): Name of the model for which the plots are to be produced.
    - metrics1 (dict): Evaluation metrics for the first compression ratio (e.g., Compression Ratio = 30).
    - metrics2 (dict): Evaluation metrics for the second compression ratio (e.g., Compression Ratio = 60).
    - metrics3 (dict): Evaluation metrics for the third compression ratio (e.g., Compression Ratio = 90).
    - iterations (list): List of iterations to be examined.
    
    Returns:
    None. The function directly visualizes the plots.
    
    Usage:
    To utilize this function, run your model evaluations and gather the metrics for the desired compression ratios. Then, invoke this function passing the gathered metrics and desired iterations as parameters.
    """
    
    metric_pairs = [
        ('iterations', 'avg_bpp_list', 'Iterations', 'Average Bit-Per-Pixel (BPP)'),
        ('iterations', 'avg_size_list', 'Iterations', 'Average Size in Kilobytes (KB)'),
        ('iterations', 'avg_psnr_list', 'Iterations', 'Average PSNR'),
        ('iterations', 'avg_psnr_noisy_list', 'Iterations', 'Average PSNR (noise)'),
        ('iterations', 'highest_val_acc_list', 'Iterations', 'Highest Validation Accuracy'),
        ('iterations', 'lowest_val_loss_list', 'Iterations', 'Lowest Validation Loss'),
        ('iterations', 'highest_val_acc_top5_list', 'Iterations', 'Highest Top-5 Validation Accuracy'),
        ('iterations', 'misclassified_counts_list', 'Iterations', 'Misclassified Counts'),
        ('iterations', 'avg_shift_list', 'Iterations', 'Average Shift'),
        ('iterations', 'avg_noise_shift_list', 'Iterations', 'Average Shift (Noise)'),
        ('iterations', 'most_shifted_distances', 'Iterations', 'Most Shifted Distances'),
        ('iterations', 'least_shifted_distances', 'Iterations', 'Least Shifted Distances'),
        ('iterations', 'avg_psnr_list', 'Iterations', 'Average PSNR (Relative to Iteration 1)'),
        ('iterations', 'highest_val_acc_list', 'Iterations', 'Highest Validation Accuracy (Relative to Iteration 1)'),
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

            # x_ticks = np.arange(min(x_values), max(x_values) + 1, (max(x_values) - min(x_values)) / 10)
            # axs[j].set_xticks(np.round(x_ticks).astype(int))
            axs[j].tick_params(axis='both', which='major', labelsize=16)
            axs[j].legend(fontsize=16)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, wspace=0.2)
        plt.show()

#%%
def plot_heatmap(metrics_list, model_names, iterations, metric_key, title):
    """
    The `plot_heatmap` function visualizes the relationship between different iterations and compression ratios for a specific metric across multiple model configurations. The function:
    - Creates heatmaps based on provided metrics, allowing users to quickly discern patterns and relationships.
    - Supports multiple models, iterating over each one and plotting its heatmap.
    - Uses Seaborn to generate visually appealing heatmaps with annotated values.
    
    Parameters:
    - metrics_list (list of dicts): List of evaluation metrics dictionaries for different scenarios.
    - model_names (list of strs): Names of the models for which the heatmaps are to be generated.
    - iterations (list): Different iterations to examine.
    - metric_key (str): The specific metric within the metrics dictionary to plot.
    - title (str): Base title for the heatmap, specific to the metric being plotted.
    
    Returns:
    None. The function directly visualizes the heatmaps.
    
    """
    compression_ratios = [30, 60, 90]

    for model_name in model_names:
        # Create an empty 2D array to store the values
        values = []

        # Iterate through the scenarios
        for metric in metrics_list:
            row = []
            for iterate in iterations:
                # Extract the value for this iteration and scenario
                value = metric[model_name][iterate][metric_key][0] # Adjust as needed
                row.append(value)
            values.append(row)

        # Convert the 2D array to a Pandas DataFrame
        df = pd.DataFrame(values, index=compression_ratios, columns=iterations)

        # Create the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df, annot=True, cmap="YlGnBu")
        plt.xlabel('Iterations')
        plt.ylabel('Compression Ratio')
        plt.title(f'{title} Heatmap for {model_name}')
        plt.show()

#%%
def sensitivity_analysis(metrics_list, model_names, iterations, metric_key, title):
    """
    The `sensitivity_analysis` function creates visual plots to showcase the sensitivity of different model variants 
    against varied iterations. The function is especially catered for scenarios involving different compression ratios. 
    This assists in understanding how different models react to compression over successive iterations.
    
    Parameters:
    - metrics_list (list of dicts): A list of evaluation metrics, each dict pertaining to a specific compression ratio.
    - model_names (list): Names of the models to be plotted.
    - iterations (list): Iteration numbers to be plotted on the x-axis.
    - metric_key (str): Key to extract the desired metric value from the metrics dictionary.
    - title (str): The label for the y-axis and part of the plot's main title.
    
    Returns:
    None. The function directly visualizes the plots.
    
    """
  
    compression_ratios = [30, 60, 90]

    plt.figure(figsize=(15, 10))

    for model_name in model_names:
        for idx, metric in enumerate(metrics_list):
            y_values = []
            for iterate in iterations:
                value = metric[model_name][iterate][metric_key][0] # Adjust as needed
                y_values.append(value)

            plt.plot(iterations, y_values, label=f'{model_name} (Compression Ratio = {compression_ratios[idx]})', linewidth=4)

    plt.xlabel('Iterations', fontsize=18, fontweight='bold')
    plt.ylabel(title, fontsize=18, fontweight='bold')
    plt.title(f'Sensitivity Analysis: {title} vs Iterations', fontsize=20, fontweight='bold')
    plt.legend(fontsize=16, loc='upper right')
    plt.grid(True)
    plt.xticks(fontsize=16, fontweight='bold')
    plt.yticks(fontsize=16, fontweight='bold')
    plt.show()

#%%
def plot_3d_surface(metrics_list, model_names, iterations, metric_key, title):
    """
    The `plot_3d_surface` function visualizes a 3D surface plot for given metrics across various compression ratios and iterations.
    Key features of this function include:
    - Generation of 3D plots using the `plotly` library to visually represent the relationship between compression ratios, iterations, and a specific metric.
    - The function supports multiple model names, iterating over each to produce individual plots.
    - Compression ratios are predefined at levels: 30, 60, and 90.
    
    Parameters:
    - metrics_list (list of dicts): List of metrics for different compression ratios.
    - model_names (list of str): Names of the models to be plotted.
    - iterations (list): List of iterations for which metrics are available.
    - metric_key (str): The key of the metric within the metrics dictionary to be plotted.
    - title (str): The title to be displayed on the z-axis and overall plot title.
    
    Returns:
    None. The function directly displays the 3D surface plots for each model.
    """
  
    compression_ratios = [30, 60, 90]

    for model_name in model_names:
        X, Y, Z = [], [], []
        for compression_ratio, metrics in zip(compression_ratios, metrics_list):
            for iterate in iterations:
                X.append(compression_ratio)
                Y.append(iterate)
                Z.append(metrics[model_name][iterate][metric_key][0])

        # Convert to numpy arrays for manipulation
        X = np.array(X).reshape(len(compression_ratios), len(iterations))
        Y = np.array(Y).reshape(len(compression_ratios), len(iterations))
        Z = np.array(Z).reshape(len(compression_ratios), len(iterations))

        # Create the 3D surface plot
        fig = go.Figure()
        fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', name='Surface'))
        fig.update_layout(
            title=dict(text=f'{title} 3D Surface Plot for {model_name}', font=dict(size=24, color='black', family="Arial, bold")),
            autosize=False,
            width=1000,
            height=800,
            scene=dict(
                xaxis_title='compression Ratio',
                yaxis_title='Iterations',
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
def run_experiment_four_motion(data_dir):
    """
    Executes experiment four, focusing on the impact of motion blur on different image compression levels. 
    This function generates and visualizes the gradual motion blur, evaluates models under different conditions, 
    and visualizes the results through various plots and heatmaps.
    
    Parameters:
    - data_dir (str): The directory path where the dataset is located.
    
    Returns:
    None. All visualizations are displayed during the function's execution.
    """
    
    categories = os.listdir(data_dir)
    categories = categories[:20]
    model_names = ['VGG16', 'MobileNet', 'DenseNet121']
    
    # Compression Ratio: 30
    images, noise_samples, decompressed_images = generate_gradual_motion_blur(data_dir, categories, kernel_size = 10, compression_ratio = 30)
    plot_gradual_motion_blur(images, noise_samples, decompressed_images)
    metrics26 = evaluate_models(data_dir, kernel_size = 10, angle = 45, compression_ratio = 30, num_iterations = 10, model_names = model_names)

    # Compression Ratio: 60
    images, noise_samples, decompressed_images = generate_gradual_motion_blur(data_dir, categories, kernel_size = 10, compression_ratio = 60)
    plot_gradual_motion_blur(images, noise_samples, decompressed_images)
    metrics27 = evaluate_models(data_dir, kernel_size = 10, angle = 45, compression_ratio = 60, num_iterations = 10, model_names = model_names)

    # Compression Ratio: 90
    images, noise_samples, decompressed_images = generate_gradual_motion_blur(data_dir, categories, kernel_size = 10, compression_ratio = 90)
    plot_gradual_motion_blur(images, noise_samples, decompressed_images)
    metrics28 = evaluate_models(data_dir, kernel_size = 10, angle = 45, compression_ratio = 90, num_iterations = 10, model_names = model_names)
    
    num_iterations = 10
    iterations = [str(i) for i in range(1, num_iterations + 1)]
    
    for model_name in model_names:
        print(f'Generating plots for {model_name}...')
        generate_plots_for_model(model_name, metrics26, metrics27, metrics28, iterations)
        print(f'Plots for {model_name} generated successfully.\n')
    
    metrics_list = [metrics26, metrics27, metrics28]
    
    # Generating heatmaps
    plot_heatmap(metrics_list, model_names, iterations, 'avg_psnr_list', 'Average PSNR')
    plot_heatmap(metrics_list, model_names, iterations, 'avg_shift_list', 'Average Shift')
    plot_heatmap(metrics_list, model_names, iterations, 'highest_val_acc_list', 'Highest Validation Accuracy')
    plot_heatmap(metrics_list, model_names, iterations, 'highest_val_acc_top5_list', 'Highest Top-5 Validation Accuracy')
    
    # Running sensitivity analysis
    sensitivity_analysis(metrics_list, model_names, iterations, 'avg_psnr_list', 'Average PSNR')
    sensitivity_analysis(metrics_list, model_names, iterations, 'avg_shift_list', 'Average Shift')
    sensitivity_analysis(metrics_list, model_names, iterations, 'highest_val_acc_list', 'Highest Validation Accuracy')
    sensitivity_analysis(metrics_list, model_names, iterations, 'highest_val_acc_top5_list', 'Highest Top-5 Validation Accuracy')
    
    # Generating 3D surface plots
    plot_3d_surface(metrics_list, model_names, iterations, 'avg_psnr_list', 'Average PSNR')
    plot_3d_surface(metrics_list, model_names, iterations, 'avg_shift_list', 'Average Shift')
    plot_3d_surface(metrics_list, model_names, iterations, 'highest_val_acc_list', 'Highest Validation Accuracy')
    plot_3d_surface(metrics_list, model_names, iterations, 'highest_val_acc_top5_list', 'Highest Top-5 Validation Accuracy')
