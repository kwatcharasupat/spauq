import os
import numpy as np
import glob
from tqdm import tqdm

_DefaultOutputPath = "/home/kwatchar3/spauq-home/data/musdb-hq/{model}-{variant}"

from demucs.pretrained import tasnet


def inference_spleeter(variant, audio_path, output_path):
    assert variant in ["4stems"]

    # from spleeter.__main__ import separate
    from spleeter.audio import Codec, STFTBackend
    from spleeter.audio.adapter import AudioAdapter
    from spleeter.separator import Separator

    mixtures = glob.glob(
        os.path.join(audio_path, "musdb18hq/test", "**", "mixture.wav"), recursive=True
    )
    os.makedirs(output_path, exist_ok=True)

    files = mixtures
    adapter = "spleeter.audio.ffmpeg.FFMPEGProcessAudioAdapter"
    bitrate = "128k"
    codec = Codec.WAV
    duration = 600.0
    offset = 0
    output_path = output_path
    stft_backend = STFTBackend.AUTO
    filename_format = "{foldername}/{instrument}.{codec}"
    params_filename = f"spleeter:{variant}"
    mwf = False
    verbose = True

    audio_adapter = AudioAdapter.get(adapter)

    for filename in tqdm(files):
        # FIXME: this a workaround to avoid spleeter saving files to wrong path
        separator = Separator(
            params_filename, MWF=mwf, stft_backend=stft_backend, multiprocess=False
        )
        separator.separate_to_file(
            str(filename),
            str(output_path),
            audio_adapter=audio_adapter,
            offset=offset,
            duration=duration,
            codec=codec,
            bitrate=bitrate,
            filename_format=filename_format,
            synchronous=True,
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
        assert variant in ["musdb", "extra"]
        if variant == "musdb":
            model = ta.pipelines.HDEMUCS_HIGH_MUSDB.get_model()
            source_order = model.sources
        elif variant == "extra":
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
    elif model == "ConvTasNet":
        assert variant in ["musdb", "extra"]
        model = tasnet(pretrained=True, extra=variant == "extra")
        source_order = model.sources
        chunk_size = 60.0
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
