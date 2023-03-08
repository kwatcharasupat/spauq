import os
import numpy as np
import glob


_DefaultOutputPath = "/home/kwatchar3/spauq-home/data/musdb-hq/{model}-{variant}"


def inference_spleeter(variant, audio_path, output_path):
    assert variant in ["4stems"]

    from spleeter.__main__ import separate
    from spleeter.audio import Codec, STFTBackend

    mixtures = glob.glob(os.path.join(audio_path, "musdb18hq/test", "**", "mixture.wav"), recursive=True)
    os.makedirs(output_path, exist_ok=True)

    separate(
        deprecated_files=None,
        files=mixtures,
        adapter="spleeter.audio.ffmpeg.FFMPEGProcessAudioAdapter",
        bitrate="128k",
        codec=Codec.WAV,
        duration=600.0,
        offset=0,
        output_path=output_path,
        stft_backend=STFTBackend.AUTO,
        filename_format="{foldername}/{instrument}.{codec}",
        params_filename=f"spleeter:{variant}",
        mwf=False,
        verbose=True,
    )


def inference_torch(
    model,
    variant,
    audio_path,
    output_path,
):
    import torch
    import torchaudio as ta
    import pytorch_lightning as pl
    from utils import LightningWrapper, LightningDataWrapper

    torch.set_float32_matmul_precision("high")

    ds = LightningDataWrapper(
        ds=ta.datasets.MUSDB_HQ(
            root=audio_path,
            subset="test",
            sources=["mixture"],
            download=True,
        )
    )

    if model == "HDemucs":
        assert variant in ["MUSDB", "MUSDB_PLUS"]
        if variant == "MUSDB":
            model = ta.pipelines.HDEMUCS_HIGH_MUSDB.get_model()
            source_order = model.sources
        elif variant == "MUSDB_PLUS":
            model = ta.pipelines.HDEMUCS_HIGH_MUSDB_PLUS.get_model()
            source_order = model.sources
        else:
            raise NameError(f"Variant {variant} not found")
        chunk_size = 60.0
    elif model == "OpenUnmix":
        assert variant in ["umxhq"]
        model = torch.hub.load("sigsep/open-unmix-pytorch", variant)
        source_order = model.target_models.keys()
        chunk_size = np.inf
    else:
        raise NameError(f"Model {model} not found")

    model = LightningWrapper(model, output_path, source_order, chunk_size=chunk_size)

    trainer = pl.Trainer(accelerator="gpu")
    trainer.predict(model, ds)


def inference(
    model,
    variant,
    audio_path="/home/kwatchar3/spauq-home/data/musdb-hq/raw",
    output_path=None,
):
    if output_path is None:
        output_path = _DefaultOutputPath.format(model=model, variant=variant)

    if model == "Spleeter":
        inference_spleeter(variant, audio_path, output_path)
    else:
        inference_torch(model, variant, audio_path, output_path)


if __name__ == "__main__":
    import fire

    fire.Fire(inference)
