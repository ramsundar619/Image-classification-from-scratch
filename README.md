# Image classification from scratch using hot dog or not dataset
## Introduction
This code shows how to do image classification from scratch, starting from JPEG image files on disk, without leveraging pre-trained weights or a pre-made Keras Application model. The code is demonstrated on using the Kaggle Hot Dog-Not Hot Dog Dataset.
## Table of Contents
- Introduction
- Installation
- Loading Dataset
- Getting Started
- Methodology being used for image classification using Keras and Tensorflow library
## Installation
**Create a conda virtual environment with dependencies and activate it**
```
conda create --name tf_gpu tensorflow-gpu 
conda activate tf_gpu
```
## Loading Dataset
- Download the dataset from [Hot Dog-Not Hot Dog](https://drive.google.com/drive/folders/1wmsqkvEawtW18MHOhobG9QL5flbkjuKB?usp=sharing).
## Getting Started
For running the code,
```
python3 main.py
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
- tf.keras.utils.plot_model() - Converts the Keras model to dot format and save to a file.
- tf.keras.callbacks.ModelCheckpoint() - Callback to save the Keras model or model weights at some frequency.
- tf.keras.preprocessing.load_img() - Loads an image into PIL format.
- tf.keras.preprocessing.image.img_to_array() - Converts a PIL Image instance to a Numpy array.
- tf.expand_dims() - Returns a tensor with a length 1 axis inserted at index axis.
