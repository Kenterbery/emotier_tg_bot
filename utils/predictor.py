import keras.models
from settings.constants import SAVED_ESTIMATOR

class Predictor:
    def __init__(self):
        self.loaded_model = keras.models.load_model(SAVED_ESTIMATOR)

    def predict(self, feature_vector):
        return self.loaded_model.predict(feature_vector)
