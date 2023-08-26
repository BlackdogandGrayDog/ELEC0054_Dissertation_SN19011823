#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 21:27:10 2023

@author: ericwei
"""

# Standard Libraries
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

# Matplotlib for color mappings
from matplotlib.colors import ListedColormap

#%%
def generate_gaussian_noise(image, sigma):
    """
    Generate Gaussian noise with specified sigma (standard deviation) and 
    add it to the given image.
    
    Parameters:
        image (ndarray): The input image to which noise should be added.
        sigma (float): The standard deviation for the Gaussian noise.
        
    Returns:
        noise_to_show (ndarray): Noise normalized for visualization.
        noisy_image (ndarray): The original image with added Gaussian noise.
    """
    
    # Generate Gaussian noise with mean 0 and standard deviation sigma
    gaussian_noise = np.random.normal(0, sigma, image.shape)

    # Normalize the noise for visualization purposes
    noise_to_show = (gaussian_noise - gaussian_noise.min()) / (gaussian_noise.max() - gaussian_noise.min())

    # Add the noise to the original image and clip the values to be in [0, 1]
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 1)

    return noise_to_show, noisy_image


#%%
def generate_noisy_images(data_dir, categories, sigma):
    """
    Generates Gaussian noisy images for all the images in the specified categories.
    
    Parameters:
        data_dir (str): Directory path where the images are located.
        categories (list): List of categories to process.
        sigma (float): Standard deviation for Gaussian noise.
        
    Returns:
        original_images (list): List of original images.
        noisy_images (list): List of images with Gaussian noise added.
        noise_samples (list): List of Gaussian noise samples.
        labels (list): Corresponding labels for the images.
    """
    
    original_images = []
    noisy_images = []
    noise_samples = []
    labels = []

    for category in categories:
        
        if category == '.DS_Store':
            continue
        
        # Get the list of image files in the category
        image_files = os.listdir(os.path.join(data_dir, category))
        for image_file in image_files:
            image = Image.open(os.path.join(data_dir, category, image_file))
            image = image.convert('RGB')
            image = image.resize((224, 224))
            original_image_array = img_to_array(image) / 255.0
            gaussian_noise, noisy_image_array = generate_gaussian_noise(original_image_array, sigma)

            original_images.append(original_image_array)
            noisy_images.append(noisy_image_array)
            noise_samples.append(gaussian_noise)
            labels.append(category)

    return original_images, noisy_images, noise_samples, labels


#%%
def plot_images(image_array, noisy_image_array, noise_sample):
    """
    Plots the original, noise sample, and noisy images side by side.
    
    Parameters:
        image_array (ndarray): Original image.
        noisy_image_array (ndarray): Noisy image.
        noise_sample (ndarray): Gaussian noise sample.
    """
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image_array)
    plt.title('Original Image')
    plt.subplot(1, 3, 2)
    plt.imshow(noise_sample)
    plt.title('Gaussian Noise')
    plt.subplot(1, 3, 3)
    plt.imshow(noisy_image_array)
    plt.title('Noisy Image')
    plt.show()

#%%
def calculate_average_psnr(original_images, noisy_images):
    """
    Calculate the average Peak Signal-to-Noise Ratio (PSNR) for a set of images.
    
    Parameters:
        original_images (list): List of original images.
        noisy_images (list): List of noisy images.
        
    Returns:
        float: The average PSNR for the image sets.
    """
    
    psnrs = []
    for orig, noisy in zip(original_images, noisy_images):
        orig = orig.astype('float32')
        noisy = noisy.astype('float32')
        psnr = peak_signal_noise_ratio(orig, noisy, data_range=1.0)
        psnrs.append(psnr)
    return np.mean(psnrs)

#%%
def image_preprocessing(data_dir, categories, sigma):
    """
    Process images by adding Gaussian noise, calculating average PSNR,
    and preparing training, validation, and test datasets.
    
    Parameters:
        data_dir (str): Directory path containing the images.
        categories (list): List of categories to be processed.
        sigma (float): Standard deviation for Gaussian noise.
        
    Returns:
        tuple: Processed train, validation, test datasets, and other related info.
    """
    
    original_images, noisy_images, noise_samples, labels = generate_noisy_images(data_dir, categories, sigma)

    # Plot the first original and decompressed images for example
    plot_images(original_images[10], noisy_images[10], noise_samples[10])

    # Calculate and print the average PSNR
    avg_psnr = calculate_average_psnr(original_images, noisy_images)
    print(f"Average PSNR: {avg_psnr} dB")

    images = np.array(noisy_images)
    labels = np.array(labels)
    le = LabelEncoder()
    labels_int = le.fit_transform(labels)
    class_names = le.classes_
    print(class_names)
    num_classes=len(categories)
    labels = to_categorical(labels_int, num_classes)
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.25, random_state=42)

    return train_images, val_images, test_images, train_labels, val_labels, num_classes, class_names, avg_psnr

#%%
def image_preprocessing_tune(data_dir, categories, sigma):
    """
    This function is similar to `image_preprocessing`, but with a different 
    way of splitting data for tuning purposes.
    
    Parameters:
        data_dir (str): Directory path containing the images.
        categories (list): List of categories to be processed.
        sigma (float): Standard deviation for Gaussian noise.
        
    Returns:
        tuple: Processed train, validation, test datasets, and other related info.
    """
    
    original_images, noisy_images, noise_samples, labels = generate_noisy_images(data_dir, categories, sigma)

    # Plot the first original and decompressed images for example
    plot_images(original_images[10], noisy_images[10], noise_samples[10])

    # Calculate and print the average PSNR
    avg_psnr = calculate_average_psnr(original_images, noisy_images)
    print(f"Average PSNR: {avg_psnr} dB")

    images = np.array(original_images)
    noisy_images = np.array(noisy_images)
    labels = np.array(labels)
    le = LabelEncoder()
    labels_int = le.fit_transform(labels)
    class_names = le.classes_
    print(class_names)
    num_classes = len(categories)
    labels_one_hot = to_categorical(labels_int, num_classes)

    # Split the original images for training
    train_images, _, train_labels, _ = train_test_split(images, labels_one_hot, test_size=0.2, random_state=42)
    train_images, _, train_labels, _ = train_test_split(train_images, train_labels, test_size=0.25, random_state=42)

    # Split the noisy images for validation and test
    _, val_images, _, val_labels = train_test_split(noisy_images, labels_one_hot, test_size=0.25, random_state=42)
    _, test_images, _, test_labels = train_test_split(noisy_images, labels_one_hot, test_size=0.2, random_state=42)

    return train_images, val_images, test_images, train_labels, val_labels, num_classes, class_names, avg_psnr

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
def evaluate_models(data_dir, sigmas, model_names, tunable, trainable):
    """
    This function takes a directory of images, a list of sigmas (representing noise levels),
    model names, and flags indicating if a model is tunable or trainable. It applies
    Gaussian noise with different sigmas to the images and evaluates various metrics
    for specified models on the noisy images.
    
    Parameters:
    - data_dir (str): Directory where images are stored, sorted by category.
    - sigmas (list): List of noise levels.
    - model_names (list): List of model names to evaluate.
    - tunable (bool): Flag to decide if the preprocessing function to use is 'image_preprocessing' (True) or 'image_preprocessing_tune' (False).
    - trainable (bool): Flag to decide if the model weights are trainable.
    
    Returns:
    - metrics (dict): Dictionary containing various evaluation metrics for each model and sigma value.
    
    """

    categories = os.listdir(data_dir)
    categories = categories[:20]

    metrics = {}

    for model_name in model_names:
        metrics[model_name] = {}
        for sigma in sigmas:
            metrics[model_name][sigma] = {
                'avg_psnr_list': [],
                'highest_val_acc_list': [],
                'lowest_val_loss_list': [],
                'highest_val_acc_top5_list': [],
                'misclassified_counts_list': [],
                'avg_shift_list': [],
                'most_shifted_distances': [],
                'least_shifted_distances': []
            }

    for sigma in sigmas:
        print(f"Now examining Gaussion Noise with Sigma of {sigma}")
        original_images, noisy_images, noise_samples, labels = generate_noisy_images(data_dir, categories, sigma)
        if tunable:
          train_images, val_images, test_images, train_labels, val_labels, num_classes, class_names, avg_psnr = image_preprocessing(data_dir, categories, sigma)
        else:
          train_images, val_images, test_images, train_labels, val_labels, num_classes, class_names, avg_psnr = image_preprocessing_tune(data_dir, categories, sigma)

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
            metrics[model_name][sigma]['avg_psnr_list'].append(avg_psnr)

            # Find the epoch with the highest validation accuracy
            highest_val_acc_epoch = np.argmax(history.history['val_accuracy']) + 1
            highest_val_acc = np.max(history.history['val_accuracy'])
            # Append highest_val_acc to list
            metrics[model_name][sigma]['highest_val_acc_list'].append(highest_val_acc)
            # Find the epoch with the lowest validation loss
            lowest_val_loss_epoch = np.argmin(history.history['val_loss']) + 1
            lowest_val_loss = np.min(history.history['val_loss'])
            # Append lowest_val_loss to list
            metrics[model_name][sigma]['lowest_val_loss_list'].append(lowest_val_loss)
            print(f"For {model_name}, highest validation accuracy of {highest_val_acc} was achieved at epoch {highest_val_acc_epoch}")
            print(f"For {model_name}, lowest validation loss of {lowest_val_loss} was achieved at epoch {lowest_val_loss_epoch}")

            # Plot top5 accuracy
            highest_val_acc_top5, highest_val_acc_epoch_top5 = plot_top5_accuracy(history)
            # Append highest_val_acc_top5 to list
            metrics[model_name][sigma]['highest_val_acc_top5_list'].append(highest_val_acc_top5)
            print(f"For {model_name}, highest validation top 5 accuracy of {highest_val_acc_top5} was achieved at epoch {highest_val_acc_epoch_top5}")

            # Visualise true and predicted labels in feature spaces and use classifier to create decision boundaries
            misclassified_counts, total_misclassified = visualize_feature_space(model, val_images, val_labels, class_names, resolution=500)
            print(misclassified_counts)
            # Append misclassified_counts to list
            metrics[model_name][sigma]['misclassified_counts_list'].append(total_misclassified)
            visualize_decision_boundary_knn(model, val_images, val_labels, class_names, resolution=500)
            visualize_decision_boundary_rf(model, val_images, val_labels, class_names, resolution=500)
            visualize_decision_boundary_svm(model, val_images, val_labels, class_names, resolution=500)

            # Compute average shift due to compression
            avg_shift = average_shift(model, original_images, noisy_images)
            # Append avg_shift to list
            metrics[model_name][sigma]['avg_shift_list'].append(avg_shift)
            print(f"Average shift due to compression for model {model_name} is {avg_shift}")

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
            metrics[model_name][sigma]['most_shifted_distances'].append(distances[most_shifted_indices[0]])
            # Use `most_shifted_indices` to analyze the top 5 most severely shifted data points
            most_shifted_originals = [original_images[i] for i in most_shifted_indices]
            most_shifted_compresseds = [noisy_images[i] for i in most_shifted_indices]
            # Visualize the decision boundary, original point, and shifted point
            visualize_decision_boundary_svm_shifted_topn(categories, model, np.array(original_images), labels, class_names, most_shifted_originals, most_shifted_compresseds, n_pairs=5, resolution=500, padding=7.0)


            # Find the index of the data point with the top 5 minimum distance
            least_shifted_indices = np.argsort(distances)[:5].tolist()
            print("Top 5 distances between each pair of original and shifted points in high-dimensional space:", distances[least_shifted_indices])
            # Append the minimum distance to the least_shifted_distances list
            metrics[model_name][sigma]['least_shifted_distances'].append(distances[least_shifted_indices[0]])
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
def generate_plot_for_model_single(model_name, metric, sigmas):
    """
    Generate Plot for Single Model:
    
    This function creates a series of scatter plots for various evaluation metrics
    of a single model across different noise levels (sigmas). The plots visually 
    represent how each metric varies with sigma.
    
    Parameters:
    - model_name (str): Name of the model for which the plots are being generated.
    - metric (dict): Dictionary containing evaluation metrics for each model and sigma value.
    - sigmas (list): List of noise levels.
    
    """
    # Pairs of metrics to be plotted against each other
    metric_pairs = [
        ('sigma', 'avg_psnr_list', 'Sigma', 'Average PSNR'),
        ('sigma', 'highest_val_acc_list', 'Sigma', 'Highest Validation Accuracy'),
        ('sigma', 'lowest_val_loss_list', 'Sigma', 'Lowest Validation Loss'),
        ('sigma', 'highest_val_acc_top5_list', 'Sigma', 'Highest Top-5 Validation Accuracy'),
        ('sigma', 'misclassified_counts_list', 'Sigma', 'Misclassified Counts'),
        ('sigma', 'avg_shift_list', 'Sigma', 'Average Shift'),
        ('sigma', 'most_shifted_distances', 'Sigma', 'Most Shifted Distances'),
        ('sigma', 'least_shifted_distances', 'Sigma', 'Least Shifted Distances'),
        ('sigma', 'avg_psnr_list', 'Sigma', 'Average PSNR (Relative to Sigma 0.001)'),
        ('sigma', 'highest_val_acc_list', 'Sigma', 'Highest Validation Accuracy (Relative to Sigma 0.001)'),
        ('avg_psnr_list', 'highest_val_acc_list', 'Average PSNR', 'Highest Validation Accuracy'),
        ('avg_psnr_list', 'lowest_val_loss_list', 'Average PSNR', 'Lowest Validation Loss'),
        ('avg_psnr_list', 'highest_val_acc_top5_list', 'Average PSNR', 'Highest Top-5 Validation Accuracy'),
        ('avg_psnr_list', 'avg_shift_list', 'Average PSNR', 'Average Shift'),
    ]
    
    # Iterate over metric pairs with a stride of 2
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

            # Fetch metric values for each sigma
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

            axs[j].scatter(x_values, y_values, color='b', s=100) # Blue dots
            axs[j].plot(x_values, y_values, color='r', linewidth=3) # Red line

            axs[j].tick_params(axis='both', which='major', labelsize=16)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, wspace=0.2)
        plt.show()
        
#%%
def generate_plots_for_model(model_name, metrics1, metrics2, metrics3, sigmas):
    """
    Generate Plots for a Single Model across Multiple Metrics:
    
    This function creates scatter plots for various evaluation metrics 
    of a single model across different noise levels (sigmas). The plots 
    visually represent how each metric varies with sigma for different model versions.
    
    Parameters:
    - model_name (str): Name of the model for which the plots are being generated.
    - metrics1, metrics2, metrics3 (dict): Dictionaries containing evaluation metrics 
        for each model version and sigma value. Correspond to 'W_pre-trained', 'W_Noisy', 
        and 'W_clean' respectively.
    - sigmas (list): List of noise levels (sigma values).
    
    """
    
    # Define pairs of metrics that are to be plotted against each other
    metric_pairs = [
        ('sigma', 'avg_psnr_list', 'Sigma', 'Average PSNR'),
        ('sigma', 'highest_val_acc_list', 'Sigma', 'Highest Validation Accuracy'),
        ('sigma', 'lowest_val_loss_list', 'Sigma', 'Lowest Validation Loss'),
        ('sigma', 'highest_val_acc_top5_list', 'Sigma', 'Highest Top-5 Validation Accuracy'),
        ('sigma', 'misclassified_counts_list', 'Sigma', 'Misclassified Counts'),
        ('sigma', 'avg_shift_list', 'Sigma', 'Average Shift'),
        ('sigma', 'most_shifted_distances', 'Sigma', 'Most Shifted Distances'),
        ('sigma', 'least_shifted_distances', 'Sigma', 'Least Shifted Distances'),
        ('sigma', 'avg_psnr_list', 'Sigma', 'Average PSNR (Relative to Sigma 0.001)'),
        ('sigma', 'highest_val_acc_list', 'Sigma', 'Highest Validation Accuracy (Relative to Sigma 0.001)'),
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

            for metric, color, label in zip([metrics1, metrics2, metrics3], ['b', 'g', 'r'], ['W_pre-trained', 'W_Noisy', 'W_clean']):
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

            x_ticks = np.arange(min(x_values), max(x_values)+1, (max(x_values) - min(x_values)) / 10)
            axs[j].set_xticks(np.round(x_ticks).astype(int))
            axs[j].tick_params(axis='both', which='major', labelsize=16)
            axs[j].legend(fontsize=16)

        plt.tight_layout()
        plt.subplots_adjust(top=0.9, wspace=0.2)
        plt.show()

#%%
def run_experiment_three_gaussian(data_dir):
    """
    Run Experiment for Three Gaussian Noise Levels:

    This function evaluates and generates plots for multiple models under various noise 
    levels. It evaluates three versions of models: pre-trained, noisy, and clean. 
    The results are plotted for each model and each version.

    Parameters:
    - data_dir (str): Directory path containing the dataset.

    """

    # List of noise levels (sigmas) and model names to evaluate
    sigmas = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    model_names = ['VGG16', 'VGG19', 'ResNet50', 'MobileNet', 'DenseNet121']

    # Evaluate the models for different versions
    metrics1 = evaluate_models(data_dir, sigmas, model_names, tunable = False, trainable = False)
    metrics2 = evaluate_models(data_dir, sigmas, model_names, tunable = False, trainable = True)
    metrics3 = evaluate_models(data_dir, sigmas, model_names, tunable = True, trainable = True)

    # Generate and display plots for the first version of models
    for model in model_names:
        print(f'Generating plots for {model}...')
        generate_plot_for_model_single(model, metrics1, sigmas)
        print(f'Plots for {model} generated successfully.\n')

    # Generate and display plots comparing all versions of models
    for model in model_names:
        print(f'Generating plots for {model}...')
        generate_plots_for_model(model, metrics1, metrics2, metrics3, sigmas)
        print(f'Plots for {model} generated successfully.\n')
