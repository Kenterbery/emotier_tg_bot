import logging
import subprocess

import librosa
import numpy as np

from settings.constants import LENGTH


class AudioWorker:
    """
    Object class to work with Telegram audio data
    and prepare it for predictive model.
    """

    def __init__(self):
        self.path = None
        self.data = None
        self.sr = None

    def _set_path(self, new_path: str):
        self.path = new_path

    def _set_data(self, data: np.ndarray):
        self.data = data

    def _set_sr(self, sr: int):
        self.sr = sr

    def _convert_file(self, path):
        subprocess.run(["ftransc", "-f", "ogg", path])
        path_to_converted_file = path.partition(".oga")[0] + ".ogg"
        self._set_path(path_to_converted_file)

    @staticmethod
    def _zcr(data, frame_length=2048, hop_length=512):
        zcr = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
        return np.squeeze(zcr)

    @staticmethod
    def _rmse(data, frame_length=2048, hop_length=512):
        rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
        return np.squeeze(rmse)

    @staticmethod
    def _mfcc(data, sr, flatten: bool = True):
        mfcc_feature = librosa.feature.mfcc(y=data, sr=sr)
        return np.ravel(mfcc_feature.T) if flatten else np.squeeze(mfcc_feature.T)

    def _extract_features_from_data(self):
        result = np.array([])
        result = np.hstack((result,
                            self._zcr(self.data),
                            self._rmse(self.data),
                            self._mfcc(self.data, self.sr)))
        return result

    def fit(self, path: str) -> np.ndarray:
        """
        Fit object of AudioWorker class to feature vector using three main features: ''zero crossing rate,
        RMS and MFCC''.
        * Firstly, gotten audio converts to compatible format (in this case, .ogg).
        * Next, converts from audio to data and sample_rate, using librosa library.
        * Then apply functions for calculating
        needed features and contstruct the feature vector, which Predictor class can use.
        :param path: path to root voice file
        :return: feature vector of ndarray.
        """
        # Converts file to compatible format (like .ogg)
        self._convert_file(path)
        logging.info(f"Get voice from {path}")

        # Loading data from converted file for generating of feature vector
        data, sr = librosa.load(self.path, duration=2.5)

        # Bring data to trained shape
        diff = LENGTH - len(data)
        if diff > 0:
            data = np.append(data, np.zeros((diff,)))
        else:
            data = data[:LENGTH]
        print(len(data))
        assert len(data) == LENGTH

        self._set_data(data)
        self._set_sr(sr)
        logging.info(f"Data has been loaded from voice.")

        # Extracting features from audio and generating the feature vector
        feature_vector = self._extract_features_from_data()
        return feature_vector
