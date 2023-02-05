import pyroomacoustics as room

import pytorch_lightning as pl

from torch.utils import data
from data.musdb import MusDB18HQDataset
from data.room import MonoToStereoPanLaw, StereoToStereoPanLawWithMix

from data.timit import TIMITDataset

import pyroomacoustics as pra

DATASETS = ["TIMIT", "MUSDB18-HQ"]


class StereoDatamodule:
    def __init__(
        self,
        dataset,
        min_pan=-90,
        max_pan=90,
        pan_step=30,
        min_error=-90,
        max_error=90,
        error_step=15,
        signal_filter_kwargs=[],
        estim_filter_kwargs=[],
        **kwargs
    ) -> None:
        super().__init__()

        if dataset == "TIMIT":
            DatasetConstructor = TIMITDataset
            self.room = MonoToStereoPanLaw()
            self.fs = 16000
        elif dataset == "MUSDB18-HQ":
            DatasetConstructor = MusDB18HQDataset
            self.room = StereoToStereoPanLawWithMix()
            self.fs = 44100
        else:
            raise NameError

        self.train_dataset = DatasetConstructor(
            mode="train",
            fs=self.fs,
            room=self.room,
            min_pan=min_pan,
            max_pan=max_pan,
            pan_step=pan_step,
            min_error=min_error,
            max_error=max_error,
            error_step=error_step,
            signal_filter_kwargs=signal_filter_kwargs,
            estim_filter_kwargs=estim_filter_kwargs,
            **kwargs
        )
        self.val_dataset = DatasetConstructor(
            mode="val",
            fs=self.fs,
            room=self.room,
            min_pan=min_pan,
            max_pan=max_pan,
            pan_step=pan_step,
            min_error=min_error,
            max_error=max_error,
            error_step=error_step,
            signal_filter_kwargs=signal_filter_kwargs,
            estim_filter_kwargs=estim_filter_kwargs,
            **kwargs
        )

    def train_dataloader(self):
        return data.DataLoader(dataset=self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return data.DataLoader(dataset=self.val_dataset, batch_size=1)


if __name__ == "__main__":

    timit_dm = StereoDatamodule(
        dataset="TIMIT",
        room_dim=(3, 3),
        snr=None,
        rt60=0.150,
        src_dist=1.0,
        mic_spacing=0.60,
        directivity="CARDIOID",
    )

    timit_dl = timit_dm.train_dataloader()

    for item in timit_dl:
        print(item)
