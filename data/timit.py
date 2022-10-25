from collections import defaultdict
from typing import Optional
import pytorch_lightning as pl
from torch.utils import data
import os
import glob

import soundfile as sf

import numpy as np

from data.room import StereoMicInRoom


class TIMITDataset(data.Dataset):
    def __init__(
        self,
        mode,
        room: StereoMicInRoom,
        min_angle=-90,
        max_angle=90,
        angle_step=10,
        min_error=-90,
        max_error=90,
        error_step=10,
        root="/home/kwatchar3/data/timit/timit",
    ) -> None:
        super().__init__()

        subfolder = {"train": "train", "val": "test"}[mode]

        self.files = sorted(
            glob.glob(os.path.join(root, subfolder, "**", "*.wav"), recursive=True)
        )

        self.n_files = 4620 if mode == "train" else 1680

        assert len(self.files) == self.n_files

        self.room = room

        self.angles = np.linspace(
            min_angle, max_angle, (max_angle - min_angle) // angle_step + 1
        )

        self.errors = np.linspace(
            min_error, max_error, (max_error - min_error) // error_step + 1
        )
        
        print(self.angles, self.errors)

        self.n_angles = len(self.angles)
        self.n_errors = len(self.errors)

        self.length = self.n_files * self.n_angles * self.n_errors

        self.cache = defaultdict(lambda: defaultdict(lambda: None))
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):

        file_idx = index // (self.n_angles * self.n_errors)
        angerr_idx = index % (self.n_angles * self.n_errors)

        path = self.files[file_idx]
        data, fs = sf.read(path)

        angle_idx = angerr_idx // self.n_errors
        error_idx = angerr_idx % self.n_errors

        angle = self.angles[angle_idx]
        error = self.errors[error_idx]
        
        # print(angle_idx, error_idx)
        
        signal = self.cache[file_idx][angle]
        
        if signal is None:
            signal = self.room.generate_one(data, angle)
            self.cache[file_idx][angle] = signal
            
        estim = self.cache[file_idx][angle + error]
        
        if estim is None:
            estim = self.room.generate_one(data, angle + error)
            self.cache[file_idx][angle + error] = estim
            
            
        return {
            'xtrue': signal,
            'xest': estim,
            'true_angle': angle,
            'est_angle': angle+error,
            'est_deviation': error,
            'file': path
        }
