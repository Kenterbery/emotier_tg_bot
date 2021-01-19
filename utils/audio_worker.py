import numpy as np
import librosa
import subprocess
import logging

class AudioWorker():
    """ Object class to work with Telegram audio data
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
        process = subprocess.run(["ftransc", "-f", "ogg", path])
        path_to_converted_file = path.partition(".oga")[0] + ".ogg"
        self._set_path(path_to_converted_file)

    def _zcr(data, frame_length=2048, hop_length=512):
        zcr = librosa.feature.zero_crossing_rate(y=data, frame_length=frame_length, hop_length=hop_length)
        return np.squeeze(zcr)

    def _rmse(data, frame_length=2048, hop_length=512):
        rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
        return np.squeeze(rmse)

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

    def fit(self, path):
        self._convert_file(path)
        logging.info(f"Get voice from {path}")

        data, sr = librosa.load(self.path)
        self._set_data(data)
        self._set_sr(sr)
        logging.info(f"Data has been loaded from voice.")

        feature_vector = self._extract_features_from_data()
        return feature_vector





