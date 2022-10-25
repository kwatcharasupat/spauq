import pyroomacoustics as room

import pytorch_lightning as pl

from torch.utils import data
from data.musdb import MusDB18HQDataset
from data.room import StereoMicInRoom

from data.timit import TIMITDataset

import pyroomacoustics as pra

DATASETS = ["TIMIT", "MUSDB18-HQ"]


class StereoDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        room_dim,
        snr,
        rt60,
        src_dist=1.0,
        mic_spacing=0.60,
        directivity="CARDIOID",
        min_angle=-90,
        max_angle=90,
        angle_step=30,
        min_error=-90,
        max_error=90,
        error_step=15,
    ) -> None:
        super().__init__()


        if dataset == "TIMIT":
            DatasetConstructor = TIMITDataset
            self.fs = 16000
        elif dataset == "MUSDB18-HQ":
            DatasetConstructor = MusDB18HQDataset
            self.fs = 44100
        else:
            raise NameError

        self.room = StereoMicInRoom(
            dimensions=room_dim,
            rt60=rt60,
            fs=self.fs,
            mic_spacing=mic_spacing,
            directivity=pra.directivities.DirectivityPattern.__dict__[directivity],
            src_dist=src_dist,
            snr=snr,
        )

        self.train_dataset = DatasetConstructor(mode="train", room=self.room, min_angle=min_angle, max_angle=max_angle, angle_step=angle_step, min_error=min_error, max_error=max_error, error_step=error_step)
        self.val_dataset = DatasetConstructor(mode="val", room=self.room, min_angle=min_angle, max_angle=max_angle, angle_step=angle_step, min_error=min_error, max_error=max_error, error_step=error_step)

    def train_dataloader(self):
        return data.DataLoader(
            dataset=self.train_dataset,
            batch_size=1
        )

    def val_dataloader(self):
        return data.DataLoader(
            dataset=self.val_dataset,
            batch_size=1
        )


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
