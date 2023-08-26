# Dissertation: Robustness Analysis of Neural Networks under Noisy and Compressed Conditions

This repository contains the code and datasets for a comprehensive investigation into the effects of real-world conditions on the performance of neural networks, focusing on noise and compression artefacts.

## Motivation

In practical applications, images often suffer from various types of noise due to less than ideal capture conditions. Furthermore, these images are frequently subjected to compression algorithms for efficient storage and transmission, which can introduce additional artefacts. This combination can significantly influence the performance of neural networks, making a thorough investigation essential.

## Contributions

- **Robustness Analysis under Noisy and Compressed Conditions:** Deep dive into neural networks' robustness against different codecs under noisy scenarios.
- **Analysis of Noise Interactions:** Explore the interactions between various noise types and their collective impact on neural network performance.
- **Investigation of Compression and Noise Interactions:** Understand the interplay between compression and different noise types.
- **Geometrical Analysis of Noise Impact:** Offer a geometrical perspective on the impact of different noise types.
- **Novel Deep Learning-based Compression Algorithm:** Propose a unique compression method tailored for noisy inputs.
- **Fine-tuning Based on Dominant Noise:** Introduce a fine-tuning approach based on dominant noise.

## Environment Setup

### Dependencies:

Ensure the following dependencies are installed:

```bash
dependencies:
  - python=3.9
  - pip
  - pip:
    - numpy
    - scikit-learn
    - tensorflow
    - keras
    - pillow
    - matplotlib
    - scikit-image
    - opencv-python
    - tabulate
    - imageio
    - scipy
    - glymur
    - plotly
    - seaborn
    - spyder-kernels


### Dataset:

Experiments utilize the Caltech101 dataset. Download from [here](https://data.caltech.edu/records/mzrjq-6wc02) and place the `101_ObjectCategories` folder in the `Dataset` directory.

## Experiments Plan

1. **Baseline Image Classification:** A foundational experiment to gauge DL model performance on the ImageNet dataset without noise or compression.
2. **Image Classification with Different Compression Techniques:** Evaluate the impact of JPEG and JPEG2000 compression techniques on DL model performance.
3. **Image Classification with Traditional Noise:** Assess DL model performance under traditional noise types like Gaussian Noise, Motion Blur, and Salt and Pepper Noise.
4. **Image Classification with New Trend Noise:** Analyze DL model robustness under emerging noise types, including Colour Variation and Brightness shifts.
5. **Image Classification with Compression on Noisy Images:** Simulate real-world scenarios where images are both affected by noise and subsequently compressed.
6. **Image Classification with Combined Noise:** Explore scenarios where images are influenced by multiple noise types simultaneously.
7. **Image Classification with Compressed Combined Noise:** Introduce compression to the noisy images from the previous experiment.
8. **Fine-tuning the NN with Traditional Codec:** Explore the benefits of fine-tuning the model based on the dominant noise identified from the previous experiments.
9. **Propose and Fine-tune the Novel Machine Learning Compression Algorithm:** Introduce a novel attention-based neural compression network tailored for noisy inputs.

## Getting Started

1. Clone this repository.
2. Download and set up the dataset.
3. Install the required dependencies.
4. Execute the main experiment script (main.py).

---

For further queries or issues, please refer to the attached documentation or contact the repository owner.
