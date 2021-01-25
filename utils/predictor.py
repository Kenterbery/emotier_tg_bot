from keras import models
from keras import layers
from settings.constants import SAVED_ESTIMATOR
from .metrics import f1_m
import numpy as np


class Predictor:
    def __init__(self, n_batch=1):
        self.dependencies = {
            'f1_m': f1_m
        }
        self.lm = models.load_model(SAVED_ESTIMATOR, custom_objects=self.dependencies)
        self.weights = self.lm.get_weights()
        self.model = models.Sequential()
        self.n_batch = n_batch
        self.model.add(layers.Conv1D(512, kernel_size=5, strides=1,
                                     padding="same", activation="relu",
                                     batch_input_shape=(self.n_batch, 2376, 1)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPool1D(pool_size=5, strides=2, padding="same"))

        self.model.add(layers.Conv1D(512, kernel_size=5, strides=1,
                                     padding="same", activation="relu"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPool1D(pool_size=5, strides=2, padding="same"))

        self.model.add(layers.Conv1D(256, kernel_size=5, strides=1,
                                     padding="same", activation="relu"))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPool1D(pool_size=5, strides=2, padding="same"))

        self.model.add(layers.Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling1D(pool_size=5, strides=2, padding='same'))

        self.model.add(layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.MaxPooling1D(pool_size=3, strides=2, padding='same'))

        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(512, activation='relu'))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.Dense(7, activation="softmax"))

        self.model.set_weights(self.weights)

    def predict(self, feature_vector):
        return dict(zip(('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'), np.squeeze(self.model.predict(feature_vector))))
