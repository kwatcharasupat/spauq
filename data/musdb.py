from collections import defaultdict
from typing import Optional
import pytorch_lightning as pl
import os
import glob

import soundfile as sf

import numpy as np
from data.dataset import Dataset

from data.room import Spatializer


class MusDB18Dataset(Dataset):
    def __init__(
        self,
        mode,
        instrument,
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
        root="/home/kwatchar3/data/musdb18hq",
        limit=None,
    ) -> None:

        assert instrument in ["vocals", "drums", "bass", "other", "mixture"]

        subfolder = {"train": "train", "val": "test", "all": "*"}[mode]

        self.files = sorted(
            glob.glob(
                os.path.join(root, subfolder, "*", f"{instrument}.wav"), recursive=True
            )
        )

        self.n_files = {
            "train": 100,
            "val": 50,
            "all": 150,
        }[mode]

        assert len(self.files) == self.n_files

        super().__init__(
            fs=fs,
            room=room,
            min_pan=min_pan,
            max_pan=max_pan,
            pan_step=pan_step,
            min_error=min_error,
            max_error=max_error,
            error_step=error_step,
            signal_filter_kwargs=signal_filter_kwargs,
            estim_filter_kwargs=estim_filter_kwargs,
            limit=limit,
        )

        print(mode, self.pans, self.errors)


class MusDB18HQDataset(MusDB18Dataset):
    def __init__(
        self,
        mode,
        instrument,
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
        root="/home/kwatchar3/data/musdb18hq",
        **kwargs,
    ) -> None:
        super().__init__(
            mode=mode,
            instrument=instrument,
            fs=44100,
            room=room,
            limit=44100 * 60,
            min_pan=min_pan,
            max_pan=max_pan,
            pan_step=pan_step,
            min_error=min_error,
            max_error=max_error,
            error_step=error_step,
            signal_filter_kwargs=signal_filter_kwargs,
            estim_filter_kwargs=estim_filter_kwargs,
            root=root,
        )
