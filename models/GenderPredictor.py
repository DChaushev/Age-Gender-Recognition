import tensorflow as tf
import numpy as np


class GenderPredictor:
    model = 'models/gender_model.h5'
    weights = 'models/gender_weights.hdf5'

    def __init__(self):
        self.network = tf.keras.models.load_model(self.model)
        self.network.load_weights(self.weights)

    def predict(self, image):
        prediction = self.network.predict(image)
        return np.argmax(prediction[0])
