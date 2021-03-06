# Image classification from scratch using hot dog or not dataset
## Introduction
This code shows how to do image classification from scratch, starting from JPEG image files on disk, without leveraging pre-trained weights or a pre-made Keras Application model. The code is demonstrated on using the Kaggle Hot Dog-Not Hot Dog Dataset.
## Table of Contents
- Introduction
- Installation
- Loading Dataset
- Getting Started
- Methodology being used for image classification using Keras and Tensorflow library
- Input
- Parameters used for libraries
- Output
## Installation
**Create a conda virtual environment and activate it**
```
conda create --name tf24
conda activate tf24
```
**Install Dependencies**
```
conda install tensorflow==2.4.1=gpu_py38h8a7d6ce_0
conda install matplotlib
conda install numpy==1.19.5
pip install pydot
sudo apt install graphviz
```
## Loading Dataset
- Download the dataset from [Hot Dog-Not Hot Dog](https://drive.google.com/drive/folders/1wmsqkvEawtW18MHOhobG9QL5flbkjuKB?usp=sharing).
## Getting Started
**For Training**
```
python3 main.py
```
**For Testing**
```
python3 test.py
```
## Methodology being used for image classification using Keras and Tensorflow library
- tf.compat.v1.ConfigProto.gpu_options.allow_growth = True (prevents tensorflow from allocating the totality of a gpu memory)
- tf.compat.as_bytes() - Converts bytearray, bytes, or unicode python input types to bytes.
- tf.keras.preprocessing.image_dataset_from_directory() - Generates a tf.data.Dataset from image files in a directory.
  - Parameters used
  - directory = "images" (Directory where the data is located).
  - validation_split = 0.2 (Optional float between 0 and 1, fraction of data to reserve for validation).
  - subset="training" (One of "training" or "validation". Only used if validation_split is set).
  - seed=1337 (Optional random seed for shuffling and transformations).
  - image_size=image_size (Size to resize images to after they are read from disk).
  - batch_size=batch_size (Size of the batches of data).
- plt.figure() - Create a new figure.
- plt.subplot() - Add an Axes to the current figure.
- plt.imshow() - Display data as an image.
- plt.title() - Set a title for the axes.
- plt.axis() - Convenience method to get or set some axis properties.
- tf.keras.Sequential() - Sequential groups a linear stack of layers into a tf.keras.Model.
- tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal") - A preprocessing layer which randomly flips images horizontally during training.
- tf.keras.layers.experimental.preprocessing.RandomRotation() - A preprocessing layer which randomly rotates images during training.
  - parameters used
  - factor - a float represented as fraction of 2 Pi, or a tuple of size 2 representing lower and upper bound for rotating clockwise and counter-clockwise. A                  positive values means rotating counter clock-wise, while a negative value means clock-wise. When represented as a single float, this value is used                for both the upper and lower bound.
- tf.Keras.Input() - used to instantiate a Keras tensor.
- tf.keras.layers.Rescaling() - A preprocessing layer which rescales input values to a new range.
- tf.keras.layers.Conv2D() - 2D convolution layer (e.g. spatial convolution over images).
  - Parameters used
  - filters - Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
  - kernel_size - An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. 
  - strides - An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width.
  - padding - one of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding with zeros evenly to the left/right or up/down of the input. 
- tf.Keras.layers.BatchNormalization() - Layer that normalizes its inputs.
- tf.Keras.layers.Activation() - Applies an activation function to an output.
- tf.keras.layers.SeparableConv2D() - Depthwise separable 2D convolution.
- tf.keras.layers.MaxPool2D() - Max pooling operation for 2D spatial data.
- tf.keras.layers.Add() - Layer that adds a list of inputs.
- tf.keras.layers.GlobalAveragePooling2D() - Global average pooling operation for spatial data.
- tf.keras.layers.Dropout() - Applies Dropout to the input.
- tf.keras.layers.Dense() - regular densely-connected NN layer.
- tf.keras.activations.sigmoid - Sigmoid activation function, sigmoid(x) = 1 / (1 + exp(-x)).
- tf.keras.activations.softmax - Softmax converts a vector of values to a probability distribution.
- tf.keras.Model.compile() - Configures the model for training.
  - Parameters used
  - Optimizer - An optimizer is one of the two arguments required for compiling a Keras model.
  - Loss - The purpose of loss functions is to compute the quantity that a model should seek to minimize during training.
  - metrics - A metric is a function that is used to judge the performance of your model.
- tf.keras.metrics.Accuracy - Calculates how often predictions equal labels.
- tf.keras.losses.BinaryCrossentropy - Computes the cross-entropy loss between true labels and predicted labels.
- tf.keras.optimizers.Adam - Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order                                    moments.
- tf.keras.utils.plot_model() - Converts the Keras model to dot format and save to a file.
- tf.keras.callbacks.ModelCheckpoint() - Callback to save the Keras model or model weights at some frequency.
- tf.keras.preprocessing.load_img() - Loads an image into PIL format.
- tf.keras.preprocessing.image.img_to_array() - Converts a PIL Image instance to a Numpy array.
- tf.expand_dims() - Returns a tensor with a length 1 axis inserted at index axis.
## Input
Here we are using [Hot Dog-Not Hot Dog](https://drive.google.com/drive/folders/1wmsqkvEawtW18MHOhobG9QL5flbkjuKB?usp=sharing) from Kaggle as an input to the model which predicts a given image as either a hot dog or not a hot dog once trained.
## Parameters used
- image_size = (180, 180) - A standard dimension we fix for all images before feeding into the model.
- batch_size = 16 - batch size for training and testing.
- validation_split = 0.2 - fraction of data to reserve for validation.
- seed = 1337 - Optional random seed for shuffling and transformations.
- buffer_size = 32 - Creates a Dataset that prefetches elements from this dataset.
- filters - The dimensionality of the output space (i.e. the number of output filters in the convolution).
- kernel_size - An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window.
- strides - An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. 
- padding - one of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding with zeros evenly to the left/right or up/down of the input. When padding="same" and strides=1, the output has the same size as the input. 
- "relu" - Rectified Linear Unit Function which returns element-wise max(x, 0).
- "sigmoid" - Sigmoid activation function, sigmoid(x) = 1 / (1 + exp(-x)). Returns a value between 0 to 1.
- "softmax" - Softmax converts a vector of values to a probability distribution.The elements of the output vector are in range (0, 1) and sum to 1.
- rate = 0.5 - Float between 0 and 1. Fraction of the input units to drop. 
- num_classes = 2(hot_dog or not_hot_dog)
- Learning rate = 1e-3 for adam optimizer
## Ouput
After training the model for 50 epochs, it shows a training accuracy of 88 percent and a validation accuracy of 78 percent.
