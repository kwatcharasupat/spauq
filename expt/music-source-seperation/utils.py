import os
import torchaudio as ta
from tqdm import tqdm
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch
import numpy as np


class LightningWrapper(pl.LightningModule):
    def __init__(
        self, model, output_path, source_order, chunk_size=60, overlap=0.25
    ) -> None:
        super().__init__()
        self.model = model
        self.output_path = output_path
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.n_src = 4
        self.source_order = source_order

    def forward(self, x):
        return self.model(x)

    def chunk_audio(self, audio, fs, length):
        n_chunk = int(fs * self.chunk_size)
        n_hop = int(fs * (1 - self.overlap) * self.chunk_size)

        n_slices = int((length - n_chunk) / n_hop) + 1
        padded_length = n_hop * n_slices + n_chunk
        n_pad = padded_length - length

        audio = torch.nn.functional.pad(audio, (0, n_pad))

        audio = audio.unfold(-1, n_chunk, n_hop)
        audio = torch.permute(audio, (1, 0, 2)).contiguous()

        return audio, n_pad

    def unchunk_audio(self, audio, fs, length, n_pad):
        n_slices, _, n_chan, n_chunk = audio.shape

        n_hop = int(fs * (1 - self.overlap) * self.chunk_size)
        n_overlap = n_chunk - n_hop

        fade = ta.transforms.Fade(
            fade_in_len=0, fade_out_len=n_overlap, fade_shape="linear"
        )

        output = torch.zeros((self.n_src, n_chan, length))

        for i in range(n_slices):
            if i == n_slices - 1:
                fade.fade_out_len = 0

            faded = fade(audio[i])

            if i == n_slices - 1:
                faded = faded[:, :, :-n_pad]

            output[:, :, i * n_hop : min(length, i * n_hop + n_chunk)] += faded

            if i == 0:
                fade.fade_in_len = n_overlap

        return output

    def predict_step(self, batch, batch_idx):
        mix, fs, length, name = batch

        if self.chunk_size < np.inf:
            mix, n_pad = self.chunk_audio(mix[0, 0], fs, length)

            n_slices, n_chan, n_samples = mix.shape

            output = torch.zeros((n_slices, self.n_src, n_chan, n_samples))

            with torch.inference_mode():
                for i in tqdm(range(n_slices)):
                    output[i] = self.model(mix[i].unsqueeze(0))[0]

            output = self.unchunk_audio(output, fs, length, n_pad)
        else:
            with torch.inference_mode():
                output = self.model(mix[:, 0, :, :])[0]

        for s, source in zip(self.source_order, output):
            os.makedirs(os.path.join(self.output_path, name[0]), exist_ok=True)
            ta.save(
                os.path.join(self.output_path, name[0], f"{s}.wav"),
                source.cpu().squeeze(),
                sample_rate=fs,
            )


class LightningDataWrapper(pl.LightningDataModule):
    def __init__(self, ds, num_workers=16) -> None:
        super().__init__()
        self.ds = ds
        self.num_workers = num_workers

    def predict_dataloader(self):
        return DataLoader(
            self.ds,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )
