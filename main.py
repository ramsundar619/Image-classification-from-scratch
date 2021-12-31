#Importing tensorflow libraries
import tensorflow as tf                       
from tensorflow import keras
from tensorflow.keras import layers
import os

#Extending gpu limit for cudnn library
from tensorflow.compat.v1 import ConfigProto                  
from tensorflow.compat.v1 import InteractiveSession           
config = ConfigProto()                                        #A ProtocolMessage
config.gpu_options.allow_growth = True                        #Prevents tensorflow from allocating the totality of a gpu memory
session = InteractiveSession(config=config)                   #A TensorFlow Session for use in interactive contexts, such as a shell

#Filter out corrupted images
num_skipped = 0
for folder_name in ("hot_dog", "not_hot_dog"):                #accessing the folders
    folder_path = os.path.join("image", folder_name)          #makes the path image/folder_name
    for fname in os.listdir(folder_path):                     #accessing the images
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)  
        finally:
            fobj.close()

        if not is_jfif:                                       #badly-encoded images that do not feature the string "JFIF" in their header
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print("Deleted %d images" % num_skipped)

#Generate a Dataset
image_size = (180, 180)                                       #A standard dimension we fix for all images before feeding into the model
batch_size = 16                                               #batch size for training and testing

#Yields batches of images from the subdirectories hot_dog and not_hot_dog, together with their labels for training
train_ds = tf.keras.preprocessing.image_dataset_from_directory(      
    "image",                                                   #Directory where the data is located
    validation_split=0.2,                                      #fraction of data to reserve for validation
    subset="training",                                         #set for training
    seed=1337,                                                 #Optional random seed for shuffling and transformations
    image_size=image_size,                                     #Size to resize images to after they are read from disk
    batch_size=batch_size,                                     #Size of the batches of data
)

#Yields batches of images from the subdirectories hot_dog and not_hot_dog, together with their labels for testing
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "image",
    validation_split=0.2,                                       
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

#Visualizing the data
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(int(labels[i]))
        plt.axis("off")

#Data Augmentation to avoid overfitting        
data_augmentation = keras.Sequential(
    [
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal"),   #A preprocessing layer which randomly flips images horizontally during training                    
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),        #A preprocessing layer which randomly rotates images during training- The factor inside is a float represented as fraction of 2 Pi, or a tuple of size 2 representing lower and upper bound for rotating clockwise and counter-clockwise. A positive values means rotating counter clock-wise, while a negative value means clock-wise. When represented as a single float, this value is used for both the upper and lower bound.
    ]
)

#Visualization of Data Augmentation 
plt.figure(figsize=(10, 10))
for images, _ in train_ds.take(1):
    for i in range(9):
        augmented_images = data_augmentation(images)
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(augmented_images[0].numpy().astype("uint8"))
        plt.axis("off")
        
#Data Augmentation applied to the dataset        
augmented_train_ds = train_ds.map(                                   #A technique to increase the diversity of your training set by applying random (but realistic) transformations, such as image rotation.
  lambda x, y: (data_augmentation(x, training=True), y))             #To create a dataset that yields batches of augmented images. The function here defines a lambda expression that takes two arguments x(image) and y(label) and returns the augmented images along with their labels.

#Configuring the dataset for performance
train_ds = train_ds.prefetch(buffer_size=32)                         #Creates a Dataset that prefetches elements from this dataset.Most dataset input pipelines should end with a call to prefetch. This allows later elements to be prepared while the current element is being processed. This often improves latency and throughput, at the cost of using additional memory to store prefetched elements.
val_ds = val_ds.prefetch(buffer_size=32)                             #buffer_size - A tf.int64 scalar tf.Tensor, representing the maximum number of elements that will be buffered when prefetching. If the value tf.data.AUTOTUNE is used, then the buffer size is dynamically tuned. 

#Building a model
def make_model(input_shape, num_classes):                             #input_shape - shape of the input images, num_classes - number of categories of output
    inputs = keras.Input(shape=input_shape)                           #Feeding the input
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    #first compartment of layers
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)     #Rescaling layer - RGB channel values are in the [0, 255] range. This is not ideal for a neural network.So we will standardize values to be in the [0, 1] by using a Rescaling layer at the start of our model.
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)            #Convolution Layer for convolution operation. Parameters used here; filters=32 - Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution);kernel_size=3 - An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions;strides-An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions;padding-one of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding with zeros evenly to the left/right or up/down of the input. When padding="same" and strides=1, the output has the same size as the input. 
    x = layers.BatchNormalization()(x)                                #Batch Normalization to standardize the inputs to the next layer for each mini-batch
    x = layers.Activation("relu")(x)                                  #Activation Layer for non-linearizing - Here relu - Rectified Linear Unit Function which returns element-wise max(x, 0). 
    #second compartment of layers
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x                                     #Set aside residual

    for size in [128, 256, 512, 728]:                                 #running a for loop for this compartment of layers with filter size as 128,256,512 and 728
        #Third compartment of layers
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        #Fourth compartment of layers
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)       #Depthwise separable 2D convolution; parameters have same meaning as Conv2D
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)     #Max Pooling Layer for Max pooling operation for 2D spatial data; pool_size=3 - integer or tuple of 2 integers, window size over which to take the maximum. (2, 2) will take the max value over a 2x2 pooling window. If only one integer is specified, the same window length will be used for both dimensions. 

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])                                # Add back residual
        previous_block_activation = x                                # Set aside next residual
    #Fifth compartment of layers
    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)                           #Global Average Pooling Layer - Global average pooling operation for spatial data.
    if num_classes == 2:
        activation = "sigmoid"                                       #If binary use sigmoid activation(our case is binary) - Sigmoid activation function, sigmoid(x) = 1 / (1 + exp(-x)). Returns a value between 0 to 1.
        units = 1
    else:
        activation = "softmax"                                       #Else softmax activation(for multiclass classification) - Softmax converts a vector of values to a probability distribution.The elements of the output vector are in range (0, 1) and sum to 1.
        units = num_classes

    x = layers.Dropout(0.5)(x)                                       #Dropout Layer to avoid overfitting;  rate- Float between 0 and 1. Fraction of the input units to drop. 
    outputs = layers.Dense(units, activation=activation)(x)          #Output - regular densely-connected NN layer; units - Positive integer, dimensionality of the output space which is 2 here, activation - activation function to be used
    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (3,), num_classes=2)     #Calling the model with num_classes = 2(hot_dog or not_hot_dog),input shape = image size=(180,180)+ (3,) for RGB Channels
keras.utils.plot_model(model, show_shapes=True)                      #Converts the Keras model to dot format and save to a file


#Training the model
epochs = 50                                                          #No of Iterations for the training

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),           #Callback to save the Keras model or model weights at some frequency
]
#Configures the model for training
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),                           #An optimizer is one of the two arguments required for compiling a Keras model. Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.Parameter used - Learning rate = 1e-3
    loss="binary_crossentropy",                                      #The purpose of loss functions is to compute the quantity that a model should seek to minimize during training. Here "binary_crossentropy"- Computes the cross-entropy loss between true labels and predicted labels.
    metrics=["accuracy"],                                            #A metric is a function that is used to judge the performance of your model. "accuracy" - Calculates how often predictions equal labels.
)
#Trains the model for a fixed number of epochs
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,    #training dataset=train_ds, epochs - no of iterations, callbacks-for saving the weights of the model in a file after every iterations, validation data = val_ds
)
