import tensorflow as tf
from keras.preprocessing import image

from data_prep_pipeline import inf_dat_prep, keras_img_gen

# Load the pre-trained vehicle classification model
model = tf.keras.models.load_model('trained_model/model.h5')

# Defining the classes
class_count = len(list(keras_img_gen()[1].class_indices.keys()))


# Making predictions
def predict_vehicle(img_path):
    img_array = inf_dat_prep('/path/to/folder')
    predictions = model.predict(img_array)
    predicted_class = tf.argmax(predictions[0])  # Get the index of the most likely class
    class_labels = class_count  # Class labels
    predicted_label = class_labels[predicted_class]
    return predicted_label


# Showing the Output
if __name__ == "__main__":
    print("Predicted vehicle:", predict_vehicle(image_path))
