
import os
import glob

from data.dataset import Dataset

from data.room import Spatializer


class TIMITDataset(Dataset):
    def __init__(
        self,
        mode,
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
        root="/home/kwatchar3/data/timit/timit",
    ) -> None:

        subfolder = {"train": "train", "val": "test"}[mode]

        self.files = sorted(
            glob.glob(os.path.join(root, subfolder, "**", "*.wav"), recursive=True)
        )

        self.n_files = 4620 if mode == "train" else 1680

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
        )

        print(mode, self.pans, self.errors)
