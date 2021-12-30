#Importing tensorflow libraries
import tensorflow as tf                       
from tensorflow import keras
from tensorflow.keras import layers
import os

#Extending gpu limit for cudnn library
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True                        #prevents tensorflow from allocating the totality of a gpu memory
session = InteractiveSession(config=config)

#Filter out corrupted images
num_skipped = 0
for folder_name in ("hot_dog", "not_hot_dog"):                #accessing the folders
    folder_path = os.path.join("image", folder_name)
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
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),        #A preprocessing layer which randomly rotates images during training
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
augmented_train_ds = train_ds.map(
  lambda x, y: (data_augmentation(x, training=True), y))

#Configuring the dataset for performance
train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

#Building a model
def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)                           #Feeding the input
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)     #Rescaling layer - RGB channel values are in the [0, 255] range. This is not ideal for a neural network.So we will standardize values to be in the [0, 1] by using a Rescaling layer at the start of our model.
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)            #Convolution Layer for convolution operation
    x = layers.BatchNormalization()(x)                                #Batch Normalization to standardize the inputs to the next layer for each mini-batch
    x = layers.Activation("relu")(x)                                  #Activation Layer for non-linearizing

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x                                     #Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)       #Depthwise separable 2D convolution
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)     #Max Pooling Layer for Max pooling operation for 2D spatial data

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])                                # Add back residual
        previous_block_activation = x                                # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)                           #Global Average Pooling Layer
    if num_classes == 2:
        activation = "sigmoid"                                       #If binary use sigmoid activation(our case is binary)
        units = 1
    else:
        activation = "softmax"                                       #Else softmax
        units = num_classes

    x = layers.Dropout(0.5)(x)                                       #Dropout Layer
    outputs = layers.Dense(units, activation=activation)(x)          #Output
    return keras.Model(inputs, outputs)

model = make_model(input_shape=image_size + (3,), num_classes=2)     #Calling the model with num_classes = 2(hot_dog or not_hot_dog)
keras.utils.plot_model(model, show_shapes=True)                      #Converts the Keras model to dot format and save to a file


#Training the model
epochs = 50                                                          #No of Iterations

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),           #Callback to save the Keras model or model weights at some frequency
]
#Configures the model for training
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),                           #An optimizer is one of the two arguments required for compiling a Keras model
    loss="binary_crossentropy",                                      #The purpose of loss functions is to compute the quantity that a model should seek to minimize during training.
    metrics=["accuracy"],                                            #A metric is a function that is used to judge the performance of your model.
)
#Trains the model for a fixed number of epochs
model.fit(
    train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
)

#Run inference on new data
img = keras.preprocessing.image.load_img(
    "image/hot_dog/856178.jpg", target_size=image_size
)
img_array = keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)                             # Create batch axis

predictions = model.predict(img_array)                               #predicting the class of an unseen new image
score = predictions[0]   
print(
    "This image is %.2f percent hot_dog and %.2f percent not_hot_dog."
    % (100 * (1 - score), 100 * score)
)

