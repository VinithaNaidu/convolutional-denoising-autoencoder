# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
In this experiment, we use an autoencoder to process handwritten digit images from the MNIST dataset. The autoencoder learns to encode and decode the images, reducing noise through layers like MaxPooling and convolutional. Then, we repurpose the encoded data to build a convolutional neural network for classifying digits into numerical values from 0 to 9. The goal is to create an accurate classifier for handwritten digits removing noise.
### Dataset
![329142347-ee3badfc-8765-4ed4-9262-85e3a201d262](https://github.com/Afsarjumail/convolutional-denoising-autoencoder/assets/118343395/b6c72c99-6edc-40ec-81c7-0851f57423f3)

## Convolution Autoencoder Network Model
![329142386-87fa1595-35ad-4967-8e26-0aa90582b4d3](https://github.com/Afsarjumail/convolutional-denoising-autoencoder/assets/118343395/af9cb643-f20a-4cb0-943b-acd0a1cf107a)


## DESIGN STEPS

### STEP 1:
Import TensorFlow, Keras, NumPy, Matplotlib, and Pandas.
### STEP 2:
Load MNIST dataset, normalize pixel values, add Gaussian noise, and clip values.
### STEP 3:
Plot a subset of noisy test images.
### STEP 4:
Define encoder and decoder architecture using convolutional and upsampling layers.
### STEP 5:
Compile the autoencoder model with optimizer and loss function.
### STEP 6:
Train the autoencoder model with specified parameters and validation data.
### STEP 7:
Plot training and validation loss curves.
### STEP 8:
Use the trained model to reconstruct images and visualize original, noisy, and reconstructed images.

## PROGRAM
```
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()
x_train.shape
```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot



### Original vs Noisy Vs Reconstructed Image




## RESULT
Thus, the convolutional autoencoder for image denoising application has been successfully developed.
