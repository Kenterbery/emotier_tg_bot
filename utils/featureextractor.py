import numpy as np


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
