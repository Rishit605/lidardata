# Importing Libraries
import numpy as np
import cv2

import tensorflow as tf

import keras
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator as IDG
from keras.applications.vgg16 import preprocess_input


# Parameters
def params():
    IMG_SIZE = (256, 256)
    BATCH_SIZE = 32
    EPOCHS = 100
    return IMG_SIZE, BATCH_SIZE, EPOCHS


def keras_img_gen(dat_path) -> keras.preprocessing.image.ImageDataGenerator:
    train_datagen = IDG(
        rescale=1 / .225,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=20,
        height_shift_range=20,
        rotation_range=20,
        preprocessing_function=preprocess_input,
        validation_split=0.3,
        horizontal_flip=True,
    )

    train_gen = train_datagen.flow_from_directory(
        dat_path,
        target_size=params()[0],
        batch_size=params()[1],
        subset='training',
        class_mode='categorical',
        shuffle=True,
    )

    valid_gen = train_datagen.flow_from_directory(
        dat_path,
        target_size=params()[0],
        batch_size=params()[1],
        subset='validation',
        class_mode='categorical',
        shuffle=True,
    )

    return train_gen, valid_gen


# Function for preprocessing images
def inf_dat_prep(img_paths):
    """Preprocesses a list of images for model prediction.

    Args:
        img_paths (list): A list of image paths.

    Returns:
        numpy.ndarray: A 4D numpy array of preprocessed images.
    """

    images_list = []
    for img_path in img_paths:
        img = image.load_img(img_path, target_size=(224, 224))  # Adjust target size as needed
        img_array = image.img_to_array(img)
        images_list.append(img_array)

    images_array = np.array(images_list)  # Convert list to array
    images_array = tf.expand_dims(images_array, axis=4)  # Add batch dimension
    images_array = tf.keras.applications.vgg16.preprocess_input(images_array)  # Preprocess using VGG16 standards

    return images_array
