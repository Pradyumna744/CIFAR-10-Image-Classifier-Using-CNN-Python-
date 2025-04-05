# CIFAR-10-Image-Classifier-Using-CNN-Python-

![CIFAR-10 Image Classifier](https://upload.wikimedia.org/wikipedia/en/thumb/2/2e/Cifar-10_1.png/220px-Cifar-10_1.png)

## Overview

This repository contains a Streamlit application for classifying images from the CIFAR-10 dataset using a Convolutional Neural Network (CNN) built with TensorFlow. The application provides a user-friendly interface to upload images, visualize model predictions, and explore model performance metrics.

## Features

- **Image Classification**: Upload an image and get predictions for one of the 10 CIFAR-10 classes.
- **Model Training**: Option to retrain the model with custom epochs.
- **Grad-CAM Visualization**: Visualize the model's attention areas for predictions.
- **Confusion Matrix**: Display the confusion matrix for model performance evaluation.
- **Training History**: Visualize the training and validation accuracy over epochs.
- **Educational Content**: Learn about CNNs and the CIFAR-10 dataset.

## Requirements

To run this application, you need the following dependencies:

- Python 3.7 or higher
- TensorFlow
- NumPy
- Streamlit
- Pillow
- OpenCV
- Pandas
- Matplotlib
- Seaborn
- Plotly
- streamlit-image-comparison

You can install the required packages using `pip`:

```bash
pip install -r requirements.txt
