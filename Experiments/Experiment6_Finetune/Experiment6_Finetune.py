#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 01:23:34 2023

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

import cv2
import numpy as np
import os
import imageio
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tabulate import tabulate
import glymur
from keras.models import load_model
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from scipy.signal import convolve
from scipy.ndimage import rotate
from matplotlib.ticker import MaxNLocator
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, MobileNet, DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import TopKCategoricalAccuracy
#%%
def generate_brightness_variation(image, variation_intensity):
    # Convert the image to the HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Apply brightness variation to the V channel
    image_hsv[:,:,2] = image_hsv[:,:,2] * variation_intensity

    # Convert back to the RGB color space
    brightened_image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    brightened_image = np.clip(brightened_image, 0, 1)  # Clip values to be in the range [0, 1]
    return brightened_image



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



def generate_gaussian_noise(image, sigma):
    # Generate Gaussian noise with mean 0 and standard deviation sigma
    gaussian_noise = np.random.normal(0, sigma, image.shape)

    # Normalize the noise for visualization purposes
    noise_to_show = (gaussian_noise - gaussian_noise.min()) / (gaussian_noise.max() - gaussian_noise.min())

    # Add the noise to the original image and clip the values to be in [0, 1]
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 1)

    return noise_to_show, noisy_image



def generate_motion_blur_noise(image, kernel_size, angle):
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



def generate_multiple_compressed_images_jpeg2000(images, labels, compression_ratios=[10, 20, 30, 40, 50, 60, 70, 80, 90]):
    decompressed_images = []  # list to store decompressed images for all compression ratios
    decompressed_labels = []  # list to store the corresponding labels for all decompressed images

    # Temporary file for saving compressed images
    temp_file_jpeg2000 = "temp_compressed.jp2"

    # Loop through each image
    for idx, image_array in enumerate(images):
        # Choose a random compression ratio for the current image
        cratio = random.choice(compression_ratios)

        # Convert the image array to PIL Image
        image = Image.fromarray((image_array * 255).astype(np.uint8))

        # Save the image as a JPEG 2000 with the chosen compression ratio
        glymur.Jp2k(temp_file_jpeg2000, np.array(image), cratios=[cratio])

        # Open the compressed image, resulting in decompression
        decompressed_image = Image.open(temp_file_jpeg2000)
        decompressed_image_array = img_to_array(decompressed_image) / 255.0  # Convert the decompressed image to array

        # Add the decompressed images and its label to the respective lists
        decompressed_images.append(decompressed_image_array)
        decompressed_labels.append(labels[idx])

        # Remove the temporary file
        if os.path.exists(temp_file_jpeg2000):
            os.remove(temp_file_jpeg2000)

    return decompressed_images, decompressed_labels




def generate_gaussian_noisy_images(images, labels, sigma_values):
    noisy_images = []

    for image in images:
        # Randomly choose a sigma value from the provided sigma_values
        sigma = np.random.choice(sigma_values)

        _, noisy_image_array = generate_gaussian_noise(image, sigma)
        noisy_images.append(noisy_image_array)

    return noisy_images, labels



def generate_blurred_images(images, labels, kernel_size, angle, max_iterations=7):
    blurred_images = []

    for image in images:
        current_image = image.copy()
        chosen_iterations = random.randint(1, max_iterations)  # Randomly choose a number of iterations
        for _ in range(chosen_iterations):
            _, current_image = generate_motion_blur_noise(current_image, kernel_size, angle)
        blurred_images.append(current_image)

    return blurred_images, labels



def generate_salt_pepper_noisy_images(original_images, original_labels, salt_probs, pepper_probs):
    noisy_images = []
    noise_patterns = []

    for original_image_array in original_images:
        # Randomly choose salt and pepper probabilities for each image
        chosen_salt_prob = random.choice(salt_probs)
        chosen_pepper_prob = random.choice(pepper_probs)
        noisy_image_array, noise_pattern = generate_salt_and_pepper_noise(original_image_array, chosen_salt_prob, chosen_pepper_prob)

        noisy_images.append(noisy_image_array)
        noise_patterns.append(noise_pattern)

    return noisy_images, original_labels, noise_patterns




def generate_colour_variations_images(images, labels, variation_level):
    distorted_images = images.copy()  # Start with a copy of the original images

    # If variation_level is 0, return the original images and labels
    if variation_level == 0:
        print("Variation level is 0, returning original images.")
        return images, labels

    num_images = len(images)

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
    return distorted_images, labels





def generate_brightness_variation_images(original_images, original_labels, iteration_range, variation_intensity):
    brightness_variation_images = []

    for original_image in original_images:
        # Randomly choose an iteration count for each image
        chosen_iterations = random.choice(iteration_range)

        varied_image = original_image.copy()
        for _ in range(chosen_iterations):
            varied_image = generate_brightness_variation(varied_image, variation_intensity)
        brightness_variation_images.append(varied_image)

    return brightness_variation_images, original_labels





def generate_compressed_images_jpeg2000(images, labels, compression_ratio):
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





def generate_combined_noisy_images(original_images, labels, variation_intensity=1.5):
    """
    Generates combined noisy images by introducing brightness variation and salt & pepper noise.

    This function creates synthetic images by varying the brightness level of the 
    original images iteratively and adding salt & pepper noise. The brightness iterations
    and noise levels are chosen randomly.

    Parameters:
    - original_images (list): A list of images to be modified.
    - labels (list): Corresponding labels for each image.
    - variation_intensity (float): The intensity of brightness variation.

    Returns:
    - list: A list of combined noisy images.
    - list: Labels corresponding to the noisy images.
    - list: Brightness levels used for each image.
    - list: Salt probabilities used for each image.
    - list: Pepper probabilities used for each image.
    """
    
    brightness_iterations = [0, 1, 2, 3]
    salt_probs = [0, 0.1, 0.15, 0.2]
    pepper_probs = [0, 0.1, 0.15, 0.2]

    combined_noisy_images = []
    brightness_levels_used = []
    salt_probs_used = []
    pepper_probs_used = []
    combined_labels = []

    for idx, original_image in enumerate(original_images):
        chosen_brightness_iter = random.choice(brightness_iterations)
        chosen_salt_prob = random.choice(salt_probs)
        chosen_pepper_prob = random.choice(pepper_probs)

        brightened_image = original_image
        for _ in range(chosen_brightness_iter):
            brightened_image = generate_brightness_variation(brightened_image, variation_intensity)
        noisy_image, _ = generate_salt_and_pepper_noise(brightened_image, chosen_salt_prob, chosen_pepper_prob)

        brightness_levels_used.append(chosen_brightness_iter)
        salt_probs_used.append(chosen_salt_prob)
        pepper_probs_used.append(chosen_pepper_prob)

        combined_noisy_images.append(noisy_image)
        combined_labels.append(labels[idx])

    return combined_noisy_images, combined_labels, brightness_levels_used, salt_probs_used, pepper_probs_used




def generate_compressed_combined_noisy_images(original_images, labels, variation_intensity=1.5):
    """
   Generates compressed noisy images with brightness variation, salt & pepper noise, 
   and JPEG2000 compression.

   This function first introduces brightness variations and salt & pepper noise to 
   the original images and then compresses them using JPEG2000 with randomly chosen
   compression ratios.

   Parameters:
   - original_images (list): A list of images to be modified.
   - labels (list): Corresponding labels for each image.
   - variation_intensity (float): The intensity of brightness variation.

   Returns:
   - list: A list of compressed noisy images.
   - list: Labels corresponding to the noisy images.
   - list: Brightness levels used for each image.
   - list: Salt probabilities used for each image.
   - list: Pepper probabilities used for each image.
   - list: Compression ratios used for each image.
   """
   
    brightness_iterations = [0, 1, 2, 3]
    salt_probs = [0, 0.1, 0.15, 0.2]
    pepper_probs = [0, 0.1, 0.15, 0.2]
    compression_ratios = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    combined_noisy_images = []
    brightness_levels_used = []
    salt_probs_used = []
    pepper_probs_used = []
    compression_ratios_used = []
    combined_labels = []

    for idx, original_image in enumerate(original_images):
        chosen_brightness_iter = random.choice(brightness_iterations)
        chosen_salt_prob = random.choice(salt_probs)
        chosen_pepper_prob = random.choice(pepper_probs)
        chosen_compression_ratio = random.choice(compression_ratios)

        brightened_image = original_image
        for _ in range(chosen_brightness_iter):
            brightened_image = generate_brightness_variation(brightened_image, variation_intensity)

        noisy_image, _ = generate_salt_and_pepper_noise(brightened_image, chosen_salt_prob, chosen_pepper_prob)

        compressed_images, _, _ = generate_compressed_images_jpeg2000([noisy_image], [labels[idx]], chosen_compression_ratio)
        decompressed_image = compressed_images[0]

        brightness_levels_used.append(chosen_brightness_iter)
        salt_probs_used.append(chosen_salt_prob)
        pepper_probs_used.append(chosen_pepper_prob)
        compression_ratios_used.append(chosen_compression_ratio)

        combined_noisy_images.append(decompressed_image)
        combined_labels.append(labels[idx])

    return combined_noisy_images, combined_labels, brightness_levels_used, salt_probs_used, pepper_probs_used, compression_ratios_used



def plot_random_samples(images, brightness_levels, salt_probs, pepper_probs, num_images=2, samples_per_image=5):
    """
    Plots random samples of images with annotations indicating brightness level, 
    salt, and pepper probabilities.

    Parameters:
    - images (list): A list of images to be displayed.
    - brightness_levels (list): Brightness levels used for each image.
    - salt_probs (list): Salt probabilities used for each image.
    - pepper_probs (list): Pepper probabilities used for each image.
    - num_images (int): The number of times to generate random samples.
    - samples_per_image (int): The number of images to display per generated sample.

    Returns:
    - None: Displays plots using matplotlib.
    """
    
    for _ in range(num_images):
        indices = random.sample(range(len(images)), samples_per_image)
        plt.figure(figsize=(15, 15))

        for idx, image_idx in enumerate(indices, 1):
            plt.subplot(1, samples_per_image, idx)
            plt.imshow(images[image_idx])
            annotation = f"Brightness: {brightness_levels[image_idx]}\nSalt: {salt_probs[image_idx]}\nPepper: {pepper_probs[image_idx]}"
            plt.title(annotation)
            plt.axis('off')
        plt.tight_layout()
        plt.show()


def plot_compressed_samples(images, brightness_levels, salt_probs, pepper_probs, compression_ratios, num_images=2, samples_per_image=5):
    """
   Plots random samples of compressed images with annotations indicating brightness level,
   salt and pepper probabilities, and compression ratio.

   Parameters:
   - images (list): A list of images to be displayed.
   - brightness_levels (list): Brightness levels used for each image.
   - salt_probs (list): Salt probabilities used for each image.
   - pepper_probs (list): Pepper probabilities used for each image.
   - compression_ratios (list): Compression ratios used for each image.
   - num_images (int): The number of times to generate random samples.
   - samples_per_image (int): The number of images to display per generated sample.

   Returns:
   - None: Displays plots using matplotlib.
   """
   
    for _ in range(num_images):
        indices = random.sample(range(len(images)), samples_per_image)
        plt.figure(figsize=(15, 15))

        for idx, image_idx in enumerate(indices, 1):
            plt.subplot(1, samples_per_image, idx)
            plt.imshow(images[image_idx])
            annotation = f"Brightness: {brightness_levels[image_idx]}\nSalt: {salt_probs[image_idx]}\nPepper: {pepper_probs[image_idx]}\nCompression Ratio: {compression_ratios[image_idx]}"
            plt.title(annotation)
            plt.axis('off')
        plt.tight_layout()
        plt.show()


def generate_gaussian_noisy_compressed_images(images, labels, sigma_values):
    """
   Generates images with Gaussian noise and then compresses them using JPEG2000.

   This function first introduces Gaussian noise to the images using the provided sigma values 
   and then compresses the resulting noisy images with JPEG2000.

   Parameters:
   - images (list): A list of images to be modified.
   - labels (list): Corresponding labels for each image.
   - sigma_values (list): Standard deviation values for the Gaussian noise.

   Returns:
   - list: A list of compressed noisy images.
   - list: Labels corresponding to the compressed noisy images.
   """
   
    noisy_images, _ = generate_gaussian_noisy_images(images, labels, sigma_values)
    decompressed_images, decompressed_labels = generate_multiple_compressed_images_jpeg2000(noisy_images, labels)

    return decompressed_images, decompressed_labels


def generate_blurred_compressed_images(images, labels, kernel_size = 10, angle = 45, max_iterations = 7):
    """
    Generates blurred images and then compresses them using JPEG2000.

    This function blurs the original images with a specified kernel size and angle 
    and then compresses the resulting blurred images using JPEG2000.

    Parameters:
    - images (list): A list of images to be modified.
    - labels (list): Corresponding labels for each image.
    - kernel_size (int): Size of the kernel used for blurring.
    - angle (int): Angle of motion for motion blur.
    - max_iterations (int): The maximum number of times blurring is applied.

    Returns:
    - list: A list of compressed blurred images.
    - list: Labels corresponding to the compressed blurred images.
    """
    blurred_images, _ = generate_blurred_images(images, labels, kernel_size, angle, max_iterations)
    decompressed_images, decompressed_labels = generate_multiple_compressed_images_jpeg2000(blurred_images, labels)

    return decompressed_images, decompressed_labels


def generate_salt_pepper_noisy_compressed_images(images, labels, salt_probs, pepper_probs):
    """
    Generates images with salt & pepper noise and then compresses them using JPEG2000.

    This function first introduces salt & pepper noise to the images with the provided 
    salt and pepper probabilities, and then compresses the resulting noisy images using JPEG2000.

    Parameters:
    - images (list): A list of images to be modified.
    - labels (list): Corresponding labels for each image.
    - salt_probs (list): Probabilities of salt noise for the images.
    - pepper_probs (list): Probabilities of pepper noise for the images.

    Returns:
    - list: A list of compressed noisy images.
    - list: Labels corresponding to the compressed noisy images.
    """
    noisy_images, _, _ = generate_salt_pepper_noisy_images(images, labels, salt_probs, pepper_probs)
    decompressed_images, decompressed_labels = generate_multiple_compressed_images_jpeg2000(noisy_images, labels)

    return decompressed_images, decompressed_labels


def generate_colour_variations_compressed_images(images, labels, variation_level = 9):
    """
    Generates images with color variations and then compresses them using JPEG2000.

    This function introduces color variations to the original images with the provided 
    variation level, and then compresses the resulting images using JPEG2000.

    Parameters:
    - images (list): A list of images to be modified.
    - labels (list): Corresponding labels for each image.
    - variation_level (int): The level of color variation applied to the images.

    Returns:
    - list: A list of compressed images with color variations.
    - list: Labels corresponding to the compressed images.
    """
    distorted_images, _ = generate_colour_variations_images(images, labels, variation_level)
    decompressed_images, decompressed_labels = generate_multiple_compressed_images_jpeg2000(distorted_images, labels)

    return decompressed_images, decompressed_labels


def generate_brightness_variation_compressed_images(images, labels, iteration_range, variation_intensity = 1.5):
    """
    Generates images with brightness variations and then compresses them using JPEG2000.
   
    This function applies brightness variations to the original images iteratively 
    using the provided iteration range and intensity, and then compresses the resulting images.
   
    Parameters:
    - images (list): A list of images to be modified.
    - labels (list): Corresponding labels for each image.
    - iteration_range (list): The range of iterations for brightness variation.
    - variation_intensity (float): The intensity of brightness variation.
   
    Returns:
    - list: A list of compressed images with brightness variations.
    - list: Labels corresponding to the compressed images.
    """
    brightness_variation_images, _ = generate_brightness_variation_images(images, labels, iteration_range, variation_intensity)
    decompressed_images, decompressed_labels = generate_multiple_compressed_images_jpeg2000(brightness_variation_images, labels)

    return decompressed_images, decompressed_labels



def model_construction(model_name, num_classes, trainable, num_layer, include_fc, num_neurons, lr):
    """
    Function to construct a chosen model, optionally add a fully connected layer, and add a logistic layer to it.
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

    # Conditionally add a fully-connected layer
    if include_fc:
        x = Dense(num_neurons, activation='relu')(x)

    # Add a logistic layer with the number of classes in the dataset
    predictions = Dense(num_classes, activation='softmax')(x)

    # Construct the full model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Conditionally freeze the base model layers
    if trainable:
        if model_name in ['VGG16', 'MobileNet', 'DenseNet121']:
            # For VGG16, MobileNet, and DenseNet121, make only the last 4 layers trainable
            for layer in base_model.layers[:-num_layer]:
                layer.trainable = False
        # Extend this logic for other models if needed
    else:
        for layer in base_model.layers:
            layer.trainable = False

    # Add Top 5 metrics
    top5_acc = TopKCategoricalAccuracy(k=5, name='top_5_accuracy')

    # Compile the model
    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy', top5_acc])

    return model



def train_plotting(model, train_images, train_labels, val_images, val_labels, batch_size):
    """
    Function to train the model and plot the accuracy and loss.
    """
    # Train the model
    history = model.fit(train_images, train_labels, batch_size=batch_size, epochs=20, validation_data=(val_images, val_labels))

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



def fine_tune_and_train(model_name, images, labels, trainable, num_layer, include_fc, num_neurons, lr, batch_size, val_split=0.1):
    """
   Fine-tunes and trains a specified model on the given images and labels.
   
   Args:
   - model_name (str): Name of the pre-trained model to use.
   - images (list): List of input images.
   - labels (list): List of labels corresponding to the images.
   - trainable (bool): If true, the model layers will be trainable.
   - num_layer (int): Number of layers to be included/unfrozen.
   - include_fc (bool): If true, includes a fully connected layer.
   - num_neurons (int): Number of neurons in the fully connected layer (if included).
   - lr (float): Learning rate for training.
   - batch_size (int): Number of samples per batch.
   - val_split (float, optional): Fraction of the dataset to be used as validation data.
   
   Returns:
   - model: The trained model.
   - history: Training history of the model.
   """
   
    # Convert string category labels into integers
    le = LabelEncoder()
    int_labels = le.fit_transform(labels)
    num_classes = len(le.classes_)

    # One-hot encode the integer labels
    one_hot_labels = to_categorical(int_labels, num_classes=num_classes)

    # Split the dataset into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(images, one_hot_labels, test_size=val_split, stratify=labels)

    # Construct and compile the model
    model = model_construction(model_name, num_classes=num_classes, trainable=trainable, num_layer = num_layer, include_fc=include_fc, num_neurons=num_neurons, lr = lr)

    # Train the model and plot the results
    history = train_plotting(model, np.array(train_images), train_labels, np.array(val_images), val_labels, batch_size = batch_size)
    highest_val_acc = np.max(history.history['val_accuracy'])
    highest_val_acc_epoch = np.argmax(history.history['val_accuracy']) + 1
    lowest_val_loss = np.min(history.history['val_loss'])
    lowest_val_loss_epoch = np.argmin(history.history['val_loss']) + 1
    print(f"For {model_name}, highest validation accuracy of {highest_val_acc} was achieved at epoch {highest_val_acc_epoch}")
    print(f"For {model_name}, lowest validation loss of {lowest_val_loss} was achieved at epoch {lowest_val_loss_epoch}")

    # Plot the top 5 accuracy for training and validation data
    highest_val_acc, highest_val_acc_epoch = plot_top5_accuracy(history)
    print(f"Highest validation top 5 accuracy of {highest_val_acc:.2f} achieved at epoch {highest_val_acc_epoch}.")

    return model, history



def generate_configurations():
    """
   Generates various configurations for model fine-tuning and training.
   
   Returns:
   - dict: A dictionary with configuration names as keys and respective configurations as values.
   """
   
    return {
        "benchmark_with_fc": {"trainable": False, "num_layer": 0, "include_fc": True, "num_neurons": 64},
        "benchmark_without_fc": {"trainable": False, "num_layer": 0, "include_fc": False, "num_neurons": 64},

        "VGG_unfreeze_1_with_fc": {"trainable": True, "num_layer": 1, "include_fc": True, "num_neurons": 64},
        "VGG_unfreeze_1_without_fc": {"trainable": True, "num_layer": 1, "include_fc": False, "num_neurons": 64},
        "VGG_unfreeze_2_without_fc": {"trainable": True, "num_layer": 2, "include_fc": False, "num_neurons": 64},
        "VGG_unfreeze_4_without_fc": {"trainable": True, "num_layer": 4, "include_fc": False, "num_neurons": 64},
        "VGG_unfreeze_5_without_fc": {"trainable": True, "num_layer": 5, "include_fc": False, "num_neurons": 64},

        "MobileNet_unfreeze_5_with_fc": {"trainable": True, "num_layer": 5, "include_fc": True, "num_neurons": 64},
        "MobileNet_unfreeze_5_without_fc": {"trainable": True, "num_layer": 5, "include_fc": False, "num_neurons": 64},
        "MobileNet_unfreeze_7_without_fc": {"trainable": True, "num_layer": 7, "include_fc": False, "num_neurons": 64},
        "MobileNet_unfreeze_8_without_fc": {"trainable": True, "num_layer": 8, "include_fc": False, "num_neurons": 64},
        "MobileNet_unfreeze_10_without_fc": {"trainable": True, "num_layer": 10, "include_fc": False, "num_neurons": 64},
        "MobileNet_unfreeze_12_without_fc": {"trainable": True, "num_layer": 12, "include_fc": False, "num_neurons": 64},

        "DenseNet_unfreeze_1_with_fc": {"trainable": True, "num_layer": 1, "include_fc": True, "num_neurons": 64},
        "DenseNet_unfreeze_1_without_fc": {"trainable": True, "num_layer": 1, "include_fc": False, "num_neurons": 64},
        "DenseNet_unfreeze_2_without_fc": {"trainable": True, "num_layer": 2, "include_fc": False, "num_neurons": 64},
        "DenseNet_unfreeze_4_without_fc": {"trainable": True, "num_layer": 4, "include_fc": False, "num_neurons": 64},
        "DenseNet_unfreeze_5_without_fc": {"trainable": True, "num_layer": 5, "include_fc": False, "num_neurons": 64},
    }



def fine_tune_models(images, labels, lr, batch_size, val_split):
    """
    Fine-tunes multiple models using various configurations.
    
    Args:
    - images (list): List of input images.
    - labels (list): List of labels corresponding to the images.
    - lr (float): Learning rate for training.
    - batch_size (int): Number of samples per batch.
    - val_split (float): Fraction of the dataset to be used as validation data.
    
    Returns:
    - dict: Dictionary containing training histories for each configuration.
    """
    models = ['VGG16', 'MobileNet', 'DenseNet121']
    configurations = generate_configurations()
    results = {}

    for model_name in models:
        for config_name, config in configurations.items():
            if config_name.startswith("benchmark") or \
               (model_name == 'VGG16' and config_name.startswith("VGG")) or \
               (model_name == 'DenseNet121' and config_name.startswith("DenseNet")) or \
               (model_name == 'MobileNet' and config_name.startswith("MobileNet")):
                print(f"Fine-tuning {model_name} with configuration: {config_name}")
                _, history = fine_tune_and_train(model_name, images, labels, lr=lr, batch_size=batch_size, val_split=val_split, **config)
                results[f"{model_name}_{config_name}"] = history

    return results



def separate_results_by_model(results):
    """
    Separates the training results by model type.
    
    Args:
    - results (dict): Dictionary of training results.
    
    Returns:
    - tuple: A tuple containing dictionaries of training results for VGG16, MobileNet, and DenseNet121 respectively.
    """
    
    vgg_results = {}
    mobilenet_results = {}
    densenet_results = {}

    for key, value in results.items():
        if key.startswith("VGG16"):
            vgg_results[key] = value
        elif key.startswith("MobileNet"):
            mobilenet_results[key] = value
        elif key.startswith("DenseNet121"):
            densenet_results[key] = value

    return vgg_results, mobilenet_results, densenet_results



def plot_results_for_model(model_results, model_name):
    """
   Plots the validation accuracy and loss for a specific model across various configurations.
   
   Args:
   - model_results (dict): Dictionary of training results for a specific model.
   - model_name (str): Name of the model for which results are to be plotted.
   """
   
    # Create a unique set of configuration base names (i.e., without the model name prefix)
    config_names = set(["_".join(key.split('_')[1:]) for key in model_results.keys()])

    # Create a unique color for each configuration to keep consistency across plots
    colors = plt.cm.jet(np.linspace(0, 1, len(config_names)))

    color_map = {config: color for config, color in zip(config_names, colors)}

    # Initialize the plots for each model
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for key, history in model_results.items():
        config_name = "_".join(key.split('_')[1:])
        color = color_map[config_name]
        epochs_range = range(1, len(history.history['val_accuracy']) + 1)
        axes[0].plot(epochs_range, history.history['val_accuracy'], '-o', label=f"{config_name}", color=color)
        axes[1].plot(epochs_range, history.history['val_loss'], '-o', label=f"{config_name}", color=color)

    # Set titles, labels, legends, and make them larger and bold
    font = {'weight': 'bold', 'size': 16}
    plt.rc('font', **font)

    axes[0].set_title(f"{model_name} Validation Accuracy", fontsize=18, fontweight='bold')
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(loc="best", prop={'size': 14})
    axes[0].grid(True)

    axes[1].set_title(f"{model_name} Validation Loss", fontsize=18, fontweight='bold')
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].legend(loc="best", prop={'size': 14})
    axes[1].grid(True)

    # Increase tick size
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Display the plot
    plt.tight_layout()
    plt.show()



def hyperparameter_tuning(images, labels, model_name, config_name, lr, val_split=0.1):
    """
  Fine-tunes a model over various batch sizes for hyperparameter tuning.
  
  Args:
  - images (list): List of input images.
  - labels (list): List of labels corresponding to the images.
  - model_name (str): Name of the pre-trained model to use.
  - config_name (str): Configuration name to use for fine-tuning.
  - lr (float): Learning rate for training.
  - val_split (float, optional): Fraction of the dataset to be used as validation data.
  
  Returns:
  - dict: Dictionary containing training histories for each batch size.
  """
  
    # Batch sizes to tune
    batch_sizes = [2, 4, 8, 16, 32, 64]
    results = {}

    configurations = generate_configurations()
    config = configurations.get(config_name)

    for batch in batch_sizes:
        print(f"Fine-tuning {model_name} with configuration: {config_name} and batch size: {batch}")
        _, history = fine_tune_and_train(
            model_name, images, labels,
            lr=lr, batch_size=batch, val_split=val_split,
            **config
        )
        results[f"{model_name}_{config_name}_batch_{batch}"] = history

    return results



def plot_tuning_results(results, metric='val_accuracy'):
    # Using ggplot style for a more professional look
    plt.style.use('ggplot')

    # Larger figure size for better visibility
    plt.figure(figsize=(14, 7))

    for key, history in results.items():
        epochs = range(1, len(history.history[metric]) + 1)  # This line ensures it starts from 1
        plt.plot(epochs, history.history[metric], '-o', label=f"{key}", linewidth=2, markersize=6)

    plt.title(f"Model Performance Based on {metric}", fontsize=18, fontweight='bold')
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel(metric, fontsize=15)
    plt.xticks(ticks=range(1, 21), labels=range(1, 21), fontsize=12)  # Set ticks from 1 to 20
    plt.yticks(fontsize=12)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.show()




def learning_rate_tuning(images, labels, model_name, config_name, batch_size, val_split=0.1):
    """
    Fine-tunes a model over various learning rates for hyperparameter tuning.
    
    Args:
    - images (list): List of input images.
    - labels (list): List of labels corresponding to the images.
    - model_name (str): Name of the pre-trained model to use.
    - config_name (str): Configuration name to use for fine-tuning.
    - batch_size (int): Number of samples per batch.
    - val_split (float, optional): Fraction of the dataset to be used as validation data.
    
    Returns:
    - dict: Dictionary containing training histories for each learning rate.
    """
    
    # Learning rates to tune
    learning_rates = [0.00001, 0.0001, 0.001, 0.01, 0.1]
    results = {}

    for lr in learning_rates:
        print(f"Fine-tuning {model_name} with configuration: {config_name} and learning rate: {lr}")
        _, history = fine_tune_and_train(
            model_name, images, labels,
            lr=lr, batch_size=batch_size, val_split=val_split,
            **generate_configurations()[config_name]  # Assuming you have a configuration function
        )
        results[f"{model_name}_{config_name}_lr_{lr}"] = history

    return results



def extract_metrics_from_results(results):
    """
   Extracts key metrics like validation accuracy, validation loss, and top-5 accuracy from the training results.
   
   Args:
   - results (dict): Dictionary containing training histories.
   
   Returns:
   - dict: Dictionary containing extracted metrics for each configuration.
   """
   
    metrics = {}

    for config, history in results.items():
        highest_val_acc = max(history.history['val_accuracy'])
        lowest_val_loss = min(history.history['val_loss'])

        # Assuming you have 'val_top_5_accuracy' in the history
        if 'val_top_5_accuracy' in history.history:
            highest_top_5_acc = max(history.history['val_top_5_accuracy'])
        else:
            highest_top_5_acc = 'N/A'  # or some default value

        metrics[config] = {
            'highest_val_acc': highest_val_acc,
            'lowest_val_loss': lowest_val_loss,
            'highest_top_5_acc': highest_top_5_acc
        }

    return metrics




def print_metrics_table(metrics):
    headers = ["Configuration", "Highest Val Accuracy", "Lowest Val Loss", "Highest Top 5 Accuracy"]
    table_data = []

    for config, values in metrics.items():
        table_data.append([config, values['highest_val_acc'], values['lowest_val_loss'], values['highest_top_5_acc']])

    # Printing the table
    print(tabulate(table_data, headers=headers))




def final_construct_train_and_test(model_name, combined_noisy_images, combined_labels, trainable, num_layer, include_fc, num_neurons, lr, batch_size, epochs, test_split=0.2):
    """
    Construct, train, and test the model using the provided training and test data.
    """
    # Convert string category labels into integers
    le = LabelEncoder()
    int_labels = le.fit_transform(combined_labels)
    num_classes = len(le.classes_)

    # One-hot encode the integer labels
    one_hot_labels = to_categorical(int_labels, num_classes=num_classes)

    # Split the combined datasets into training and test sets
    train_images, test_images, train_labels, test_labels = train_test_split(combined_noisy_images, one_hot_labels, test_size=test_split, stratify=combined_labels)

    # Construct and compile the model
    model = model_construction(model_name, num_classes=num_classes, trainable=trainable, num_layer=num_layer, include_fc=include_fc, num_neurons=num_neurons, lr=lr)

    # Train the model using only training data
    history = model.fit(np.array(train_images), train_labels, epochs=epochs, batch_size=batch_size)

    # Evaluate the model on the test set
    eval_results = model.evaluate(np.array(test_images), test_labels, verbose=0)
    test_loss, test_acc = eval_results[0], eval_results[1]
    print(f"\nFor {model_name}:")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    # Save the model
    model_path = f"{model_name}_final_model.h5"
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    return model, history



def train_and_test_on_other_noise(model_path, noisy_images, labels, batch_size, epochs, noise_name, test_split=0.2, val_split=0.1):
    """
    Load the model, set its layers to non-trainable, train it on noisy images, and then test.
    """
    # Load the saved model
    model = load_model(model_path)

    # Set layers of the model to be non-trainable
    for layer in model.layers:
        layer.trainable = False

    # Convert string category labels into integers
    le = LabelEncoder()
    int_labels = le.fit_transform(labels)
    num_classes = len(le.classes_)

    # One-hot encode the integer labels
    one_hot_labels = to_categorical(int_labels, num_classes=num_classes)

    # Split the noisy datasets into training, validation, and test sets
    train_images, temp_images, train_labels, temp_labels = train_test_split(noisy_images, one_hot_labels, test_size=(test_split + val_split), stratify=labels)
    val_images, test_images, val_labels, test_labels = train_test_split(temp_images, temp_labels, test_size=test_split/(test_split + val_split), stratify=temp_labels)

    # Train the model using only training data and validate
    history = model.fit(np.array(train_images), train_labels, validation_data=(np.array(val_images), val_labels), epochs=epochs, batch_size=batch_size)

    # Evaluate the model on the test set
    eval_results = model.evaluate(np.array(test_images), test_labels, verbose=0)
    test_loss, test_acc = eval_results[0], eval_results[1]
    print(f"\nFor {noise_name}:")
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")


    # Plot the training and validation accuracy
    plt.figure(figsize=(12, 6))

    plt.subplot(121)
    plt.plot(range(1, len(history.history['accuracy']) + 1), history.history['accuracy'], linewidth = 2)
    plt.plot(range(1, len(history.history['val_accuracy']) + 1), history.history['val_accuracy'], linewidth = 2)
    plt.title('Model accuracy', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.xticks(ticks=range(1, len(history.history['accuracy']) + 1))
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True)

    # Plot the training and validation loss
    plt.subplot(122)
    plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'])
    plt.plot(range(1, len(history.history['val_loss']) + 1), history.history['val_loss'])
    plt.title('Model loss', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.xticks(ticks=range(1, len(history.history['loss']) + 1))
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return model, history



def run_experiment_six(data_dir, model_path):
    
    """
    Function to run the sixth experimental setup for a dissertation project.

    This function executes a comprehensive experimental pipeline:
    1. Original and combined noisy images are generated.
    2. Multiple pre-trained models are fine-tuned with generated images.
    3. Hyperparameter tuning is conducted.
    4. Learning rates are tuned for optimal performance.
    5. The model is trained and tested on various noise settings, including:
       - Compressed images with brightness and salt & pepper noise.
       - Images with multiple JPEG2000 compression ratios.
       - Colour variation.
       - Gaussian noise.
       - Motion blur.
       - Salt & Pepper noise.
       - Brightness variation.
       - And their compressed variations.

    Parameters:
    - data_dir (str): Path to the directory containing the dataset.
    - model_path (str): Path to the model file to be used for experiments.

    Returns:
    - dict: A dictionary containing the results of various experimental stages.
    """
    

    categories = os.listdir(data_dir)
    categories = categories[:20]
    original_images, original_labels = generate_original_images(data_dir, categories)

    # 1. Fine-tune models on combined noisy images
    combined_noisy_images, combined_labels, brightness_levels, salt_probs, pepper_probs = generate_combined_noisy_images(original_images, original_labels)
    results = fine_tune_models(combined_noisy_images, combined_labels, lr=0.0001, batch_size=32, val_split=0.1)
    
    # 2. Hyperparameter tuning
    results_hyper = hyperparameter_tuning(combined_noisy_images, combined_labels, 'MobileNet', 'MobileNet_unfreeze_8_without_fc', lr=0.001)

    # 3. Learning rate tuning
    results_lr = learning_rate_tuning(combined_noisy_images, combined_labels, "MobileNet", "MobileNet_unfreeze_8_without_fc", batch_size=32)

    # 4. Train and test on other noise
    model, history = final_construct_train_and_test(model_path, combined_noisy_images, combined_labels, True, 8, False, None, 0.0001, 32, 10)

    # 5. Compressed combined noisy images
    compressed_combined_noisy_images, compressed_combined_labels, brightness_levels, salt_probs, pepper_probs, compression_ratios = generate_compressed_combined_noisy_images(original_images, original_labels)
    model, history = train_and_test_on_other_noise(model_path, compressed_combined_noisy_images, compressed_combined_labels, 32, 10, "Compressed_Brightness_SaltAndPepper")

    # 6. Train and test on decompressed images
    decompressed_images, decompressed_labels = generate_multiple_compressed_images_jpeg2000(original_images, original_labels)
    model, history = train_and_test_on_other_noise(model_path, decompressed_images, decompressed_labels, 32, 10, "Compressed_JPEG2000_MultipleRatios")

    # 7. Train and test on colour varied images
    colour_varied_images, colour_varied_labels = generate_colour_variations_images(original_images, original_labels, 9)
    model, history = train_and_test_on_other_noise(model_path, colour_varied_images, colour_varied_labels, 32, 10, "Colour_Variations")

    # 8. Train and test on Gaussian noisy images
    sigma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    noisy_images, noisy_labels = generate_gaussian_noisy_images(original_images, original_labels, sigma_values)
    model, history = train_and_test_on_other_noise(model_path, noisy_images, noisy_labels, 32, 10, "Gaussian_Noise")

    # 9. Train and test on motion blurred images
    blurred_images, blurred_labels = generate_blurred_images(original_images, original_labels, kernel_size = 5, angle = 45)
    model, history = train_and_test_on_other_noise(model_path, blurred_images, blurred_labels, 32, 10, "Motion_Blur")

    # 10. Train and test on salt & pepper noise
    salt_probs = [0, 0.1, 0.15, 0.2]
    pepper_probs = [0, 0.1, 0.15, 0.2]
    noisy_images, noisy_labels = generate_salt_pepper_noisy_images(original_images, original_labels, salt_probs, pepper_probs)
    model, history = train_and_test_on_other_noise(model_path, noisy_images, noisy_labels, 32, 10, "Salt_Pepper_Noise")

    # 11. Train and test on brightness variation
    iteration_range = [0, 1, 2, 3, 4, 5]
    brightness_variation_images, brightness_variation_labels = generate_brightness_variation_images(original_images, original_labels, iteration_range, variation_intensity = 1.5)
    model, history = train_and_test_on_other_noise(model_path, brightness_variation_images, brightness_variation_labels, 32, 10, "Brightness_Variation")

    # 12. Train and test on Gaussian noisy compressed images
    sigma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    noisy_images, noisy_labels = generate_gaussian_noisy_compressed_images(original_images, original_labels, sigma_values)
    model, history = train_and_test_on_other_noise(model_path, noisy_images, noisy_labels, 32, 10, "Gaussian_Noise Compressed")

    # 13. Train and test on blurred images
    blurred_images, blurred_labels = generate_blurred_compressed_images(original_images, original_labels)
    model, history = train_and_test_on_other_noise(model_path, blurred_images, blurred_labels, 32, 10, "Blurred_Images")

    # 14. Train and test on salt & pepper noisy compressed images
    noisy_images, noisy_labels = generate_salt_pepper_noisy_compressed_images(original_images, original_labels, salt_probs, pepper_probs)
    model, history = train_and_test_on_other_noise(model_path, noisy_images, noisy_labels, 32, 10, "Salt_Pepper_Noise")

    # 15. Train and test on color variation images
    color_varied_images, color_varied_labels = generate_colour_variations_compressed_images(original_images, original_labels)
    model, history = train_and_test_on_other_noise(model_path, color_varied_images, color_varied_labels, 32, 10, "Color_Variations")

    # 16. Train and test on colour variations noisy images
    colour_variations_noisy_images, colour_variations_noisy_labels = generate_colour_variations_compressed_images(original_images, original_labels)
    model, history = train_and_test_on_other_noise(model_path, colour_variations_noisy_images, colour_variations_noisy_labels, 32, 10, "Colour_Variations_Noise")

    # 17. Train and test on brightness variation compressed images
    brightness_variation_compressed_images, brightness_variation_compressed_labels = generate_brightness_variation_compressed_images(original_images, original_labels, iteration_range, variation_intensity = 1.5)
    model, history = train_and_test_on_other_noise(model_path, brightness_variation_compressed_images, brightness_variation_compressed_labels, 32, 10, "Brightness_Variation_Compressed")

    # Return results
    return {
        "fine_tune_results": results,
        "hyperparameter_tuning_results": results_hyper,
        "learning_rate_results": results_lr
    }


