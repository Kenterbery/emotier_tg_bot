import numpy as np
import logging


class FeatureExtractor:
    def fit(self, feature_vector: np.ndarray):
        self.feature_vector = np.copy(feature_vector)

    def transform(self):
        # Fill nan
        self.feature_vector = np.nan_to_num(self.feature_vector, nan=0.)
        # Standardize
        mean = self.feature_vector.mean()
        std = self.feature_vector.std()
        self.feature_vector = (self.feature_vector - mean) / std
        return self.feature_vector

    def fit_transform(self, feature_vector: np.ndarray):
        self.feature_vector = np.copy(feature_vector)
        # Fill nan
        self.feature_vector = np.nan_to_num(self.feature_vector, nan=0.)
        # Standardize
        mean = self.feature_vector.mean()
        std = self.feature_vector.std()
        self.feature_vector = (self.feature_vector - mean) / std
        self.feature_vector = np.expand_dims(self.feature_vector, axis=1)
        self.feature_vector = np.expand_dims(self.feature_vector, axis=0)
        logging.info(f"Shape of the feature vector is: {self.feature_vector.shape}")
        return self.feature_vector