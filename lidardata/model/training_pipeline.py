# Importing the Libraries

import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.applications import VGG16
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, Flatten

from sklearn.preprocessing import LabelEncoder

from data_prep_pipeline import keras_img_gen, params

# Loading the Data
train_gen, valid_gen = keras_img_gen(data_path)

# Defining the classes
class_count = len(list(train_gen.class_indices.keys()))

# Defining the Model

new_input = Input(shape=(256, 256, 3))  # Changing the Input shape as Desired

base_mod = VGG16(weights='imagenet',
                 input_tensor=new_input,
                 include_top=False,
                 pooling='avg',
                 classes=class_count,
                 classifier_activation='softmax')

base_mod.trainable = False

# Defining the model architecture
x = base_mod.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = Dense(class_count, activation='softmax')(x)

model = Model(base_mod.input, x)

# Compile the model
opt = keras.optimizers.Adam(learning_rate=1e-3)

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

# Training the model

history = model.fit(train_gen, epochs=params()[2], validation_data=valid_gen,
                    validation_steps=valid_gen.samples // params()[1], steps_per_epoch=train_gen.samples // params()[1],
                    verbose=1)


def plot_loss_acc(history, save_path="plots"):
    """
    Plots the training and validation accuracy and loss curves and saves them as images.

    Args:
        history: The training history object.
        save_path: The path to save the plots (default: "plots").
    """

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.figure(figsize=(8, 6))  # Set figure size for better visualization

    # Accuracy plot
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)

    # Generate a unique filename with increment
    plot_filename = f"training_history{len(os.listdir(save_path)) + 1}.png"
    plot_path = os.path.join(save_path, plot_filename)

    plt.savefig(plot_path)
    plt.close()  # Close the figure to avoid overlapping plots

    # Loss plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, loss, 'bo', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title('Training and validation loss')
    plt.legend()

    plot_filename = f"training_history{len(os.listdir(save_path)) + 1}.png"
    plot_path = os.path.join(save_path, plot_filename)

    plt.savefig(plot_path)
    plt.close()


# Saving the Model
model.save('trained_model/best.h5')