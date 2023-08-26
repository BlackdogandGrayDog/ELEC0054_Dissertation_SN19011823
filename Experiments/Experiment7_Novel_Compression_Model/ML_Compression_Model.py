#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 01:47:10 2023

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

import os
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Dense, Reshape, Multiply, Flatten, Dropout, UpSampling2D, Concatenate
from tensorflow.image import psnr as tf_psnr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import random
from keras.metrics import top_k_categorical_accuracy
from tensorflow.keras import backend as K
import heapq
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
from scipy.signal import convolve
from scipy.ndimage import rotate
from tensorflow.keras.losses import categorical_crossentropy


# Corrected function to generate original images
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







def generate_gaussian_noise(image, sigma):
    # Generate Gaussian noise with mean 0 and standard deviation sigma
    gaussian_noise = np.random.normal(0, sigma, image.shape)

    # Normalize the noise for visualization purposes
    noise_to_show = (gaussian_noise - gaussian_noise.min()) / (gaussian_noise.max() - gaussian_noise.min())

    # Add the noise to the original image and clip the values to be in [0, 1]
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 1)

    return noise_to_show, noisy_image


def generate_gaussian_noisy_images(images, labels, sigma_values):
    noisy_images = []

    for image in images:
        # Randomly choose a sigma value from the provided sigma_values
        sigma = np.random.choice(sigma_values)

        _, noisy_image_array = generate_gaussian_noise(image, sigma)
        noisy_images.append(noisy_image_array)

    return noisy_images, labels








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


def generate_blurred_images(images, labels, kernel_size, angle, max_iterations=7):
    blurred_images = []

    for image in images:
        current_image = image.copy()
        chosen_iterations = random.randint(1, max_iterations)  # Randomly choose a number of iterations
        for _ in range(chosen_iterations):
            _, current_image = generate_motion_blur_noise(current_image, kernel_size, angle)
        blurred_images.append(current_image)

    return blurred_images, labels




def generate_brightness_variation(image, variation_intensity):
    # Convert the image to the HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Apply brightness variation to the V channel
    image_hsv[:,:,2] = image_hsv[:,:,2] * variation_intensity

    # Convert back to the RGB color space
    brightened_image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    brightened_image = np.clip(brightened_image, 0, 1)  # Clip values to be in the range [0, 1]
    return brightened_image


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



# Loading MobileNet without any modifications
def load_mobilenet_model(model_path):
    """
    Loads the MobileNet model from the given path.

    This function retrieves a pre-trained MobileNet model for further fine-tuning or inference tasks.

    Parameters:
    - model_path (str): Path to the saved MobileNet model.

    Returns:
    - Model: A TensorFlow Model object containing the MobileNet architecture.
    """
    
    mobilenet = load_model(model_path, compile=False)
    return mobilenet


# Model architecture functions
def noise_aware_attention(x, ch):
    """
    Implements the noise-aware attention mechanism.

    This function applies a noise-aware attention block to the given tensor to enhance significant features 
    and reduce the effect of noise.

    Parameters:
    - x (Tensor): Input tensor.
    - ch (int): Number of channels in the input tensor.

    Returns:
    - Tensor: Attention-weighted tensor.
    - Tensor: Global average of the attention weights.
    """
    
    gap = GlobalAveragePooling2D()(x)
    att = Dense(ch // 2, activation='relu')(gap)
    att = Dense(ch, activation='sigmoid')(att)
    att = Reshape((1, 1, ch))(att)
    x_att = Multiply()([x, att])
    global_avg_att = GlobalAveragePooling2D()(att)
    return x_att, global_avg_att


def adaptive_bottleneck(x, global_avg_att):
    """
   Implements the adaptive bottleneck mechanism.

   This function adjusts the tensor's scale based on its global average attention, 
   followed by a convolution to transform features.

   Parameters:
   - x (Tensor): Input tensor.
   - global_avg_att (Tensor): Global average attention value from the noise-aware attention mechanism.

   Returns:
   - Tensor: Transformed tensor after applying the adaptive bottleneck.
   """
   
    scaling_factor = 1 - global_avg_att
    x = Multiply()([x, scaling_factor])
    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    return x


def quantize(tensor, bits):
    """
    Quantizes the tensor to a fixed number of bits.

    This function performs quantization to reduce the bit-depth of the tensor, 
    mainly used for compression purposes.

    Parameters:
    - tensor (Tensor): Input tensor.
    - bits (int): Bit-depth for quantization.

    Returns:
    - Tensor: Quantized tensor.
    """
    min_val, max_val = tf.reduce_min(tensor), tf.reduce_max(tensor)
    scale = (max_val - min_val) / (2**bits - 1)
    quantized = tf.round(tensor / scale) * scale
    return quantized



def unet_encoder(input_shape, bits):
    """
    Encoder block with U-Net architecture.

    This function implements the encoder part of a U-Net with noise-aware attention 
    and adaptive bottleneck for noise-robust feature extraction.

    Parameters:
    - input_shape (tuple): Shape of the input tensor.
    - bits (int): Bit-depth for quantization.

    Returns:
    - Tensor: Input tensor.
    - Tensor: Quantized encoder output tensor.
    - List[Tensor]: List of skip connection tensors.
    - tuple: Shape of the quantized encoder output tensor.
    """
    inputs = Input(input_shape)

    c1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1, global_avg_att_1 = noise_aware_attention(c1, 64)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2, global_avg_att_2 = noise_aware_attention(c2, 128)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = adaptive_bottleneck(p2, global_avg_att_2)
    c3, _ = noise_aware_attention(c3, c3.shape[-1])
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    p3_quantized = quantize(p3, bits=bits)
    p3_shape = K.int_shape(p3_quantized)

    return inputs, p3_quantized, [c1, c2, c3], p3_shape



def srcnn_decoder(input_tensor, skip_connections):
    """
  Decoder block with SRCNN architecture.

  This function implements the decoder part of the architecture using Super-Resolution 
  Convolutional Neural Network (SRCNN) style layers with skip connections.

  Parameters:
  - input_tensor (Tensor): Input tensor from the encoder.
  - skip_connections (list): List of tensors for skip connections.

  Returns:
  - Tensor: Output tensor of the decoder.
  """
  
    x = Conv2D(256, (9, 9), activation='relu', kernel_initializer='he_normal', padding='same')(input_tensor)
    x, _ = noise_aware_attention(x, 256)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)

    x = Concatenate()([x, skip_connections[2]])
    x = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    x, _ = noise_aware_attention(x, 128)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)

    x = Concatenate()([x, skip_connections[1]])
    x = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    x, _ = noise_aware_attention(x, 64)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)

    outputs = Conv2D(3, (5, 5), activation='sigmoid', kernel_initializer='he_normal', padding='same')(x)

    return outputs


def generator_model(input_shape, bits):
    """
    Builds the complete generator model.

    Combines both the U-Net encoder and SRCNN decoder with quantization to create 
    the full model architecture.

    Parameters:
    - input_shape (tuple): Shape of the input tensor.
    - bits (int): Bit-depth for quantization.

    Returns:
    - Model: The full TensorFlow model.
    - tuple: Shape of the quantized encoder output tensor.
    """
    
    encoder_input, encoder_output, skip_connections, p3_shape = unet_encoder(input_shape, bits)
    decoder_output = srcnn_decoder(encoder_output, skip_connections)
    return Model(inputs=encoder_input, outputs=[decoder_output, encoder_output]), p3_shape


def kb_size_metric_for_p3(y_true, y_pred):
    """
    Metric to compute the kilobyte (KB) size of the bottleneck tensor.

    Given the true and predicted tensors, this function calculates the size 
    in kilobytes of the quantized tensor at the bottleneck.

    Parameters:
    - y_true (Tensor): True tensor (not used in the function but required for the Keras metric format).
    - y_pred (Tensor): Predicted (quantized) tensor.

    Returns:
    - float: Size of the quantized tensor in kilobytes (KB).
    """
    # y_pred here is the p3 tensor

    # Infer bit depth from the unique values of the tensor
    unique_values = tf.unique(tf.reshape(y_pred, [-1])).y
    bit_depth = tf.cast(tf.math.ceil(tf.math.log(tf.cast(tf.size(unique_values), tf.float32)) / tf.math.log(2.0)), tf.int32)

    num_elements_bottleneck = tf.size(y_pred)
    total_bits_bottleneck = tf.cast(num_elements_bottleneck, tf.float32) * tf.cast(bit_depth, tf.float32)
    total_bytes_bottleneck = total_bits_bottleneck / 8.0  # Convert bits to bytes
    total_kb_bottleneck = total_bytes_bottleneck / 1024.0  # Convert bytes to kilobytes

    return total_kb_bottleneck


def psnr(y_true, y_pred):
    """PSNR metric definition"""
    max_pixel = 1.0
    return tf.image.psnr(y_true, y_pred, max_val=max_pixel)


def train_compression_model(original_images, noisy_images, epochs, batch_size, bits):
    """
    Trains the compression model using given images.

    This function trains the full generator model on noisy images to compress and reconstruct 
    them. It also visualizes some results and provides training statistics.

    Parameters:
    - original_images (numpy array): Original, clean images.
    - noisy_images (numpy array): Noisy images to be compressed and reconstructed.
    - epochs (int): Number of training epochs.
    - batch_size (int): Training batch size.
    - bits (int): Bit-depth for quantization.

    Returns:
    - Model: Trained model.
    - History: Training history object containing loss and metric values.
    - float: Average size in kilobytes (KB) of the quantized tensor during training.
    """
    model, p3_shape = generator_model((224, 224, 3), bits)

    # Compile the model with Mean Squared Error loss and PSNR as a metric
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=[[psnr], [kb_size_metric_for_p3]])

    dummy_bottleneck = np.zeros((original_images.shape[0], p3_shape[1], p3_shape[2], p3_shape[3]))

    history = model.fit(noisy_images, [original_images, dummy_bottleneck], epochs=epochs, batch_size=batch_size, validation_split=0.2)

    # Dynamically retrieve the correct metric key for average file size in KB
    key_prefix = "tf.math.multiply"
    metric_key = next((key for key in history.history.keys() if key_prefix in key and "kb_size_metric_for_p3" in key), None)

    average_kb_size = None
    if metric_key:
        average_kb_size = sum(history.history[metric_key]) / len(history.history[metric_key])

    # Plot original vs decompressed images
    outputs = model.predict(noisy_images[50:55])  # Taking the first 5 images for demonstration
    decompressed_images = outputs[0]

    fig, axes = plt.subplots(5, 2, figsize=(10, 15))
    for idx, i in enumerate(range(5)):
        original_image = noisy_images[i + 50]
        decompressed_image = decompressed_images[i]

        # Calculate PSNR for the pair
        psnr_value = tf.image.psnr(original_image, decompressed_image, max_val=1.0).numpy()

        axes[idx, 0].imshow(original_image)
        axes[idx, 0].set_title("Original")
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(decompressed_image)
        axes[idx, 1].set_title(f"Decompressed\nPSNR: {psnr_value:.2f} dB")
        axes[idx, 1].axis('off')

    plt.tight_layout()
    plt.show()

    return model, history, average_kb_size



def denoise_images_with_labels(model, noisy_images, noisy_labels):
    """
   Uses the trained model to denoise a set of images.

   Given a trained model, noisy images, and their labels, this function predicts 
   the denoised images and returns them with their corresponding labels.

   Parameters:
   - model (Model): Trained TensorFlow model.
   - noisy_images (numpy array): Noisy images to be denoised.
   - noisy_labels (numpy array): Labels corresponding to the noisy images.

   Returns:
   - numpy array: Denoised images.
   - numpy array: Labels corresponding to the denoised images.
   """
    """Use the trained model to denoise images and return corresponding labels."""
    denoised_images = model.predict(np.array(noisy_images))
    return denoised_images[0], noisy_labels



def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)


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

    # Get highest validation accuracy and lowest validation loss
    highest_val_acc = max(history.history['val_accuracy'])
    lowest_val_loss = min(history.history['val_loss'])

    print(f"Highest validation accuracy: {highest_val_acc:.4f}")
    print(f"Lowest validation loss: {lowest_val_loss:.4f}")

    # Setting up the professional plotting style
    plt.style.use('ggplot')

    # Plotting
    epochs_range = range(1, epochs + 1)

    # Create a new figure with specified dimensions
    plt.figure(figsize=(18, 6))

    # Plot training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history.history['accuracy'], marker='o', linestyle='-', color="r", label='Training Accuracy')
    plt.plot(epochs_range, history.history['val_accuracy'], marker='o', linestyle='-', color="b", label='Validation Accuracy')
    plt.legend(fontsize=12)
    plt.title('Training and Validation Accuracy', fontweight='bold', fontsize=15)
    plt.xlabel('Epoch', fontweight='bold', fontsize=13)
    plt.ylabel('Accuracy', fontweight='bold', fontsize=13)
    plt.grid(True, which="both", ls="--", c='0.65')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history.history['loss'], marker='o', linestyle='-', color="b", label='Training Loss')
    plt.plot(epochs_range, history.history['val_loss'], marker='o', linestyle='-', color="r", label='Validation Loss')
    plt.legend(fontsize=12)
    plt.title('Training and Validation Loss', fontweight='bold', fontsize=15)
    plt.xlabel('Epoch', fontweight='bold', fontsize=13)
    plt.ylabel('Loss', fontweight='bold', fontsize=13)
    plt.grid(True, which="both", ls="--", c='0.65')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Tight layout often produces nicer results
    plt.tight_layout()
    plt.show()

    return history, highest_val_acc, lowest_val_loss, test_loss, test_acc


def train_and_test_for_bits(bits_value, original_images, noisy_images, noisy_labels, mobilenet_model_path):
    """
    Trains a compression model and subsequently tests it on a specific bit value.
    
    This function serves to:
    1. Train a compression model using provided bit value.
    2. Denoise the images using the trained model.
    3. Train and test the denoised images on various noise settings.

    Parameters:
    - bits_value (int): Bit value for compression.
    - original_images (list): Original images to be used for training.
    - noisy_images (list): Noisy versions of the images.
    - noisy_labels (list): Labels for noisy images.
    - mobilenet_model_path (str): Path to the pretrained MobileNet model.

    Returns:
    - tuple: A tuple containing various metrics like min_kb_size, max_val_acc, etc.
    """
    
    print(f"Examining bit value of: {bits_value}")

    trained_model, training_history, min_kb_size = train_compression_model(np.array(original_images), np.array(noisy_images), epochs=25, batch_size=2, bits=bits_value)

    denoised_images_array, denoised_labels = denoise_images_with_labels(trained_model, noisy_images, noisy_labels)

    history, max_val_acc, min_val_loss, test_loss, test_acc = train_and_test_on_other_noise(mobilenet_model_path, denoised_images_array, denoised_labels, batch_size=32, epochs=15, noise_name='Denoised')

    return min_kb_size, max_val_acc, min_val_loss, test_loss, test_acc


def analyze_compression_bits_effect(bits_series, original_images, noisy_images, noisy_labels, mobilenet_model_path):
    """
   Analyzes the effect of different compression bit values on model performance.
   
   This function:
   1. Trains and tests models for various bit values.
   2. Collects the metrics like min_kb_size, max_val_acc, etc. for each bit value.
   3. Plots the metrics against the bit values to visualize performance trends.

   Parameters:
   - bits_series (list): List of bit values to be examined.
   - original_images (list): Original images for training.
   - noisy_images (list): Noisy versions of the images.
   - noisy_labels (list): Labels for noisy images.
   - mobilenet_model_path (str): Path to the pretrained MobileNet model.
   """
   
    min_kb_sizes = []
    max_val_accs = []
    min_val_losses = []
    test_accs = []
    test_losses = []

    for bits_value in bits_series:
        min_kb_size, max_val_acc, min_val_loss, test_loss, test_acc = train_and_test_for_bits(bits_value, original_images, noisy_images, noisy_labels, mobilenet_model_path)

        min_kb_sizes.append(min_kb_size)
        max_val_accs.append(max_val_acc)
        min_val_losses.append(min_val_loss)
        test_accs.append(test_acc)
        test_losses.append(test_loss)

    plt.style.use('ggplot')

    # Plot for Validation Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(min_kb_sizes, max_val_accs, '-o', color="b")
    plt.xlabel('Minimum KB Size', fontweight='bold', fontsize=15)
    plt.ylabel('Validation Accuracy', fontweight='bold', fontsize=15)
    plt.title('Validation Accuracy vs Minimum KB Size', fontweight='bold', fontsize=17)
    plt.grid(True, which="both", ls="--", c='0.65')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

    # Plot for Validation Loss
    plt.figure(figsize=(12, 6))
    plt.plot(min_kb_sizes, min_val_losses, '-o', color="r")
    plt.xlabel('Minimum KB Size', fontweight='bold', fontsize=15)
    plt.ylabel('Validation Loss', fontweight='bold', fontsize=15)
    plt.title('Validation Loss vs Minimum KB Size', fontweight='bold', fontsize=17)
    plt.grid(True, which="both", ls="--", c='0.65')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

    # Plot for Test Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(min_kb_sizes, test_accs, '-o', color="b")
    plt.xlabel('Minimum KB Size', fontweight='bold', fontsize=15)
    plt.ylabel('Test Accuracy', fontweight='bold', fontsize=15)
    plt.title('Test Accuracy vs Minimum KB Size', fontweight='bold', fontsize=17)
    plt.grid(True, which="both", ls="--", c='0.65')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()

    # Plot for Test Loss
    plt.figure(figsize=(12, 6))
    plt.plot(min_kb_sizes, test_losses, '-o', color="r")
    plt.xlabel('Minimum KB Size', fontweight='bold', fontsize=15)
    plt.ylabel('Test Loss', fontweight='bold', fontsize=15)
    plt.title('Test Loss vs Minimum KB Size', fontweight='bold', fontsize=17)
    plt.grid(True, which="both", ls="--", c='0.65')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.show()



def modify_model_for_classification(generator_model, mobilenet_model_path, num_classes):
    """
    Modifies a generator model to incorporate a classification task using MobileNet.
    
    Parameters:
    - generator_model (Model): Pretrained generator model.
    - mobilenet_model_path (str): Path to the pretrained MobileNet model.
    - num_classes (int): Number of classes for the classification task.

    Returns:
    - Model: A modified model capable of reconstruction, encoding, and classification.
    """
    
    # Load the MobileNet model and make it untrainable
    mobilenet = load_mobilenet_model(mobilenet_model_path)
    for layer in mobilenet.layers:
        layer.trainable = False

    # Connect the generator's output to MobileNet
    reconstruction_output, encoder_output = generator_model.output
    classification_output = mobilenet(reconstruction_output)

    return Model(inputs=generator_model.input, outputs=[reconstruction_output, encoder_output, classification_output])


def combined_loss(y_true_list, y_pred_list):
    """
    Computes a combined loss incorporating reconstruction, bottleneck, and classification.
    
    Parameters:
    - y_true_list (list): List of true values including reconstruction and classification labels.
    - y_pred_list (list): List of predicted values from the model.

    Returns:
    - float: Combined loss value.
    """
    
    # Extract the true and predicted values
    reconstruction_true = y_true_list[0]
    classification_true = y_true_list[2]

    reconstruction_pred = y_pred_list[0]
    classification_pred = y_pred_list[2]
    bottleneck_pred = y_pred_list[1]

    # Calculate the losses
    rec_loss = tf.reduce_mean(tf.abs(reconstruction_true - reconstruction_pred))
    bottleneck_loss = tf.reduce_mean(tf.square(bottleneck_pred))
    class_loss = tf.keras.losses.categorical_crossentropy(classification_true, classification_pred)

    # Weights for each loss can be adjusted as required
    final_loss = rec_loss + 0.01 * bottleneck_loss + class_loss
    return final_loss



def train_and_test_combined_model(original_images, noisy_images, labels, epochs, batch_size, bits, mobilenet_model_path):
    """
    Trains and tests a combined model that handles compression and classification.
    
    This function:
    1. Prepares the data, converting labels and splitting the dataset.
    2. Constructs the model by combining a generator and a MobileNet.
    3. Trains the combined model.
    4. Tests the model and calculates various metrics.
    5. Visualizes original vs. decompressed images.

    Parameters:
    - original_images (list): Original images for training.
    - noisy_images (list): Noisy versions of the images.
    - labels (list): Class labels for the images.
    - epochs (int): Number of epochs for training.
    - batch_size (int): Batch size for training.
    - bits (int): Bit value for compression.
    - mobilenet_model_path (str): Path to the pretrained MobileNet model.

    Returns:
    - tuple: A tuple containing the trained model, training history, and various metrics.
    """
    
    # Convert string category labels into integers
    le = LabelEncoder()
    int_labels = le.fit_transform(labels)

    # One-hot encode the integer labels
    num_classes = len(le.classes_)
    one_hot_labels = to_categorical(int_labels, num_classes=num_classes)

    # Split dataset into training and test sets
    (train_images, test_images, train_noisy, test_noisy, train_labels, test_labels) = train_test_split(original_images, noisy_images, one_hot_labels, test_size=0.2)

    generator, p3_shape = generator_model((224, 224, 3), bits)
    model = modify_model_for_classification(generator, mobilenet_model_path, one_hot_labels.shape[1])

    # Compile the model with the new loss and metrics
    model.compile(optimizer=Adam(learning_rate=0.001), loss=[combined_loss], metrics=[[psnr, "accuracy"], [kb_size_metric_for_p3], []])

    dummy_bottleneck = np.zeros((train_images.shape[0], p3_shape[1], p3_shape[2], p3_shape[3]))
    history = model.fit(train_noisy, [train_images, dummy_bottleneck, train_labels], epochs=epochs, batch_size=batch_size, validation_split=0.2)

    # Test on test set
    dummy_test_bottleneck = np.zeros((test_images.shape[0], p3_shape[1], p3_shape[2], p3_shape[3]))
    test_results = model.evaluate(test_noisy, [test_images, dummy_test_bottleneck, test_labels], verbose=0)
    test_loss = test_results[0]
    test_psnr = test_results[2]
    test_accuracy = test_results[3]

    # Print the test results
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test PSNR: {test_psnr:.4f}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")

    # Dynamically retrieve the correct metric key for average file size in KB
    key_prefix = "tf.math.multiply"
    metric_key = next((key for key in history.history.keys() if key_prefix in key and "kb_size_metric_for_p3" in key), None)
    average_kb_size = None
    if metric_key:
        average_kb_size = sum(history.history[metric_key]) / len(history.history[metric_key])

    # Plot original vs decompressed images
    outputs = model.predict(train_noisy[50:55])  # Taking the first 5 images for demonstration from the training set
    decompressed_images = outputs[0]

    fig, axes = plt.subplots(5, 2, figsize=(10, 15))
    for idx, i in enumerate(range(5)):
        original_image = train_noisy[i + 50]
        decompressed_image = decompressed_images[i]

        # Calculate PSNR for the pair
        psnr_value = tf.image.psnr(original_image, decompressed_image, max_val=1.0).numpy()

        axes[idx, 0].imshow(original_image)
        axes[idx, 0].set_title("Original")
        axes[idx, 0].axis('off')

        axes[idx, 1].imshow(decompressed_image)
        axes[idx, 1].set_title(f"Decompressed\nPSNR: {psnr_value:.2f} dB")
        axes[idx, 1].axis('off')

    plt.tight_layout()
    plt.show()

    return model, history, average_kb_size, test_loss, test_psnr, test_accuracy


def test_on_bits_series_joint(original_images, noisy_images, labels, epochs, batch_size, bits_series, mobilenet_model_path):
    """
   Trains and tests a combined model on a series of bit values, collecting performance metrics.
   
   This function:
   1. Iterates over a series of bit values.
   2. For each bit value, trains and tests a combined model.
   3. Collects metrics for each bit value.
   4. Visualizes performance metrics against the bit values.

   Parameters:
   - original_images (list): Original images for training.
   - noisy_images (list): Noisy versions of the images.
   - labels (list): Class labels for the images.
   - epochs (int): Number of epochs for training.
   - batch_size (int): Batch size for training.
   - bits_series (list): List of bit values to be examined.
   - mobilenet_model_path (str): Path to the pretrained MobileNet model.

   Returns:
   - tuple: A tuple containing lists of metrics corresponding to each bit value.
   """
   
    # losses = []
    # accuracies = []
    kb_sizes = []
    test_losses = []
    test_accuracies = []
    test_psnrs = []
    min_val_losses = []
    max_val_accs = []

    for bits in bits_series:
        print(f"Training and testing on {bits} bits...")
        _, history, min_kb_size, test_loss, test_psnr, test_acc = train_and_test_combined_model(original_images, noisy_images, labels, epochs, batch_size, bits, mobilenet_model_path)

        min_val_loss = min(history.history['val_loss'])

        metric_key = next((key for key in history.history.keys() if "val_conv2d_" in key and "_accuracy" in key), None)
        max_val_acc = None
        if metric_key:
            max_val_acc = max(history.history[metric_key])

        print(f"Bits: {bits} | Test Loss: {test_loss:.2f} | Test Accuracy: {test_acc:.2f}% | Test PSNR: {test_psnr:.2f} | Min Validation Loss: {min_val_loss:.2f} | Max Validation Accuracy: {max_val_acc:.2f}")

        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        test_psnrs.append(test_psnr)
        min_val_losses.append(min_val_loss)
        max_val_accs.append(max_val_acc)
        kb_sizes.append(min_kb_size)

    # Plotting the results:
    plt.style.use('ggplot')

    # Plot for Test Accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(kb_sizes, test_accuracies, '-o', color="b")
    plt.xlabel('KB Size')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy vs KB Size')
    plt.show()

    # Plot for Test Loss
    plt.figure(figsize=(12, 6))
    plt.plot(kb_sizes, test_losses, '-o', color="r")
    plt.xlabel('KB Size')
    plt.ylabel('Test Loss')
    plt.title('Test Loss vs KB Size')
    plt.show()

    # Plot for Test PSNR
    plt.figure(figsize=(12, 6))
    plt.plot(kb_sizes, test_psnrs, '-o', color="g")
    plt.xlabel('KB Size')
    plt.ylabel('Test PSNR')
    plt.title('Test PSNR vs KB Size')
    plt.show()

    return min_val_losses, max_val_accs, test_losses, test_accuracies, test_psnrs, kb_sizes


def run_experiment_seven(data_dir, model_path):
    """
  This function is designed to perform a comprehensive experiment on image data by subjecting them to various types of noise, 
  analyzing the effect of different compression bit rates, and subsequently training the images using a joint approach.
  
  The entire experiment is divided into five main parts:
  
  1. Application of Salt & Pepper Noise:
     - This noise is introduced to the original images.
     - The impact of different compression bit rates on the noisy images is then analyzed.
     - The noisy images are used to train a joint model.
     
  2. Application of Colour Variations:
     - Variations in the color spectrum of the original images are introduced.
     - The effect of different compression bit rates on these colour varied images is studied.
     - The images are then used to train a joint model.
  
  3. Gaussian Noise Introduction:
     - This noise is applied to the original images.
     - The noisy images are then subjected to an analysis for different compression bit rates.
     - The images are utilized to train a joint model.
     
  4. Application of Motion Blur:
     - The original images are blurred to simulate motion.
     - An analysis based on different compression bit rates is then performed on these images.
     - Subsequently, the blurred images are used for joint model training.
     
  5. Introducing Brightness Variations:
     - The brightness of the original images is altered in varying intensities.
     - The effect of different compression bit rates on the altered images is then studied.
     - The images are then taken forward for joint model training.
  
  Parameters:
  - data_dir: Path to the directory containing image data.
  - model_path: Path to the pre-trained model (in this case, MobileNet).
  
  Returns:
  - Prints a statement indicating the completion of the experiment.
  """
  
    categories = os.listdir(data_dir)
    categories = categories[:20]
    bits_series = [1, 2, 4, 8, 16, 32]

    # 1. Salt & Pepper Noise
    salt_probs = [0, 0.1, 0.15, 0.2]
    pepper_probs = [0, 0.1, 0.15, 0.2]
    original_images, original_labels = generate_original_images(data_dir, categories)
    noisy_images, noisy_labels, _ = generate_salt_pepper_noisy_images(original_images, original_labels, salt_probs, pepper_probs)
    
    analyze_compression_bits_effect(bits_series, original_images, noisy_images, noisy_labels, model_path)
    test_on_bits_series_joint(np.array(original_images), np.array(noisy_images), original_labels, epochs=25, batch_size=32, bits_series=bits_series, mobilenet_model_path=model_path)

    # 2. Colour Variations
    colour_varied_images, colour_varied_labels = generate_colour_variations_images(original_images, original_labels, 9)
    
    analyze_compression_bits_effect(bits_series, original_images, colour_varied_images, colour_varied_labels, model_path)
    test_on_bits_series_joint(np.array(original_images), np.array(colour_varied_images), original_labels, epochs=25, batch_size=32, bits_series=bits_series, mobilenet_model_path=model_path)

    # 3. Gaussian Noise
    sigma_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    noisy_images, noisy_labels = generate_gaussian_noisy_images(original_images, original_labels, sigma_values)
    
    analyze_compression_bits_effect(bits_series, original_images, noisy_images, noisy_labels, model_path)
    test_on_bits_series_joint(np.array(original_images), np.array(noisy_images), original_labels, epochs=25, batch_size=32, bits_series=bits_series, mobilenet_model_path=model_path)

    # 4. Motion Blur
    blurred_images, blurred_labels = generate_blurred_images(original_images, original_labels, kernel_size=5, angle=45)
    
    analyze_compression_bits_effect(bits_series, original_images, blurred_images, blurred_labels, model_path)
    test_on_bits_series_joint(np.array(original_images), np.array(blurred_images), original_labels, epochs=25, batch_size=32, bits_series=bits_series, mobilenet_model_path=model_path)

    # 5. Brightness Variations
    iteration_range = [0, 1, 2, 3, 4, 5]
    brightness_variation_images, brightness_variation_labels = generate_brightness_variation_images(original_images, original_labels, iteration_range, variation_intensity=1.5)
    
    analyze_compression_bits_effect(bits_series, original_images, brightness_variation_images, brightness_variation_labels, model_path)
    test_on_bits_series_joint(np.array(original_images), np.array(brightness_variation_images), original_labels, epochs=25, batch_size=32, bits_series=bits_series, mobilenet_model_path=model_path)

    print("Experiment completed!")
