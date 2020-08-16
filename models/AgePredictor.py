import tensorflow as tf


class AgePredictor:

    model = 'models/age_model.h5'
    weights = 'models/age_weights.hdf5'

    def __init__(self):
        self.network = tf.keras.models.load_model(self.model)
        self.network.load_weights(self.weights)

    def predict(self, image):
        return self.network.predict(image)