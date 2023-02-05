from collections import defaultdict
import numpy as np

from data.room import Spatializer
from data.transform import Transform

import soundfile as sf

class Dataset:
    def __init__(
        self,
        fs: int,
        room: Spatializer,
        min_pan=-90,
        max_pan=90,
        pan_step=10,
        min_error=-90,
        max_error=90,
        error_step=10,
        signal_filter_kwargs=[],
        estim_filter_kwargs=[],
        **kwargs,
    ) -> None:
        self.room = room

        self.pans = np.linspace(min_pan, max_pan, (max_pan - min_pan) // pan_step + 1)

        self.errors = np.linspace(
            min_error, max_error, (max_error - min_error) // error_step + 1
        )

        self.n_pans = len(self.pans)
        self.n_errors = len(self.errors)

        self.length = self.n_files * self.n_pans * self.n_errors

        self.signal_cache = defaultdict(lambda: defaultdict(lambda: None))
        self.estim_cache = defaultdict(lambda: defaultdict(lambda: None))

        self.fs = fs

        self.signal_transform = Transform(signal_filter_kwargs, self.fs)
        self.estim_transform = Transform(estim_filter_kwargs, self.fs)

    
    def __len__(self):
        return self.length

    def __getitem__(self, index):

        file_idx = index // (self.n_pans * self.n_errors)
        angerr_idx = index % (self.n_pans * self.n_errors)

        path = self.files[file_idx]
        data, fs = sf.read(path, always_2d=True)
        data = data.T # chan by time

        data = 1.0 * data / np.ptp(data)

        pan_idx = angerr_idx // self.n_errors
        error_idx = angerr_idx % self.n_errors

        pan = self.pans[pan_idx]
        error = self.errors[error_idx]

        signal = self.signal_cache[file_idx][pan]

        if signal is None:
            signal = self.room.generate_one(data, pan)
            signal = self.signal_transform(signal)
            self.signal_cache[file_idx][pan] = signal

        estim = self.estim_cache[file_idx][pan + error]

        if estim is None:
            estim = self.room.generate_one(data, pan + error)
            estim = self.estim_transform(estim)
            self.estim_cache[file_idx][pan + error] = estim

        return {
            "xtrue": signal,
            "xest": estim,
            "true_pan": pan,
            "est_pan": pan + error,
            "est_deviation": error,
            "file": path,
        }
