from collections import defaultdict
from pprint import pprint
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import os

import glob

import sys

sys.path.append("/home/kwatchar3/spauq-home/spauq/src")
from spauq.core.metrics import _spauq_eval

from itertools import product

import soundfile as sf

from spatial import MonoToStereoPanLaw

from scipy.signal import remez, filtfilt, lfilter


def degrade(x, mode, settings):

    if mode == "pan":
        (pan,) = settings
        x = stereoify(x, pan=pan)
    elif mode == "panboth":
        _, pan = settings
        x = stereoify(x, pan=pan)
    elif mode == "delay":
        x = degrade(x, mode="delaypan", settings=settings + (0,))
    elif mode == "delaypan":
        delay, pan = settings
        xr = np.roll(x, delay, axis=-1)

        if delay > 0:
            xr[:, :delay] = 0
        elif delay < 0:
            xr[:, delay:] = 0

        x = stereoify(np.concatenate([x, xr], axis=0), pan=pan)
    elif mode in ["lpfpan", "lpfpan2"]:

        b, pan = settings
        if mode == "lpfpan2":
            x = lfilter(b, [1], x, axis=-1)
        else:
            x = filtfilt(b, [1], x, axis=-1)

        x = stereoify(x, pan=pan)
    elif mode == "noisepan":
        snr, pan = settings

        x = stereoify(x, pan=pan)

        noise = np.random.randn(*x.shape)
        rms = np.sqrt(np.mean(np.square(x)))
        noise_rms = np.sqrt(np.mean(np.square(noise)))
        noise_gain = np.power(10, -snr / 20) * rms / noise_rms

        x = x + noise * noise_gain
    else:
        raise NotImplementedError

    return x


def stereoify(x, pan=0):
    pl = MonoToStereoPanLaw()
    x = pl.generate_one(x, pan=pan)
    return x


def evaluate_one(inputs):
    r, mode, setting = inputs
    filename = r.split("/")[-2]
    mono, fs0 = sf.read(r, always_2d=True)
    mono = mono.T

    mono = mono / (np.max(np.abs(mono)) * 1.1)

    r = stereoify(mono, pan=setting[0] if mode == "panboth" else 0)
    e = degrade(mono, mode=mode, settings=setting)

    metrics = _spauq_eval(
        reference=r,
        estimate=e,
        fs=16000,
        return_framewise=False,
        return_cost=True,
        return_shift=True,
        return_scale=True,
        verbose=False,
    )
    return filename, metrics


_DELAYS = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def evaluate_timit(
    mode,
    reference_path="/home/kwatchar3/data/timit/timit/test",
    output_path="/home/kwatchar3/spauq-home/spauq/expt/degradation/timit/results-2s",
    sources="sa1.wav",
    max_workers=8,
):

    if mode == "pan":
        settings = [(pan,) for pan in np.linspace(-1, 1, 11)]
    elif mode == "panboth":
        settings = []

        for pan in np.linspace(-1, 1, 5):
            for err in np.concatenate(
                [np.geomspace(-2, -1e-4, 10), np.geomspace(1e-4, 2, 10)]
            ):
                if -1 <= pan + err <= 1:
                    settings.append((pan, pan + err))

    elif mode == "delay":
        settings = [(delay,) for delay in _DELAYS]
        settings_ = settings
    elif mode == "delaypan":
        settings = list(product(_DELAYS, np.concatenate([np.linspace(-1, 1, 11)])))
        settings_ = settings
    elif mode == "lpfpan":
        settings_ = list(
            product(125 * np.power(2, np.linspace(0, 6, 13)), np.linspace(-1, 1, 11))
        )
        settings = []
        for cutoff, pan in settings_:
            if cutoff >= 8000:
                cutoff = None
            if cutoff is not None:
                b = remez(
                    128,
                    [0, cutoff, cutoff * np.power(2, 1 / 3), 8000],
                    [1, 0],
                    fs=16000,
                )

            settings.append((b, pan))
    elif mode == "noisepan":
        settings = list(
            product([-24, -12, -6, -3, 0, 3, 6, 12, 24], np.linspace(-1, 1, 11))
        )
        settings_ = settings
    else:
        raise ValueError("Bad mode!")

    data = defaultdict(dict)

    for s, s_ in tqdm(list(zip(settings, settings_))):
        ref = sorted(
            glob.glob(os.path.join(reference_path, "**", sources), recursive=True)
        )
        assert len(ref) == 168, len(ref)

        fnmetrics = process_map(
            evaluate_one,
            zip(ref, [mode] * len(ref), [s] * len(ref)),
            max_workers=max_workers,
            total=len(ref),
        )

        filenames = [fn for fn, _ in fnmetrics]
        metrics = [m for _, m in fnmetrics]

        for filename, metric in zip(filenames, metrics):
            data[(filename, *s_)] = metric

            print(filename, s_)
            pprint(metric)

    os.makedirs(os.path.join(output_path, mode), exist_ok=True)

    df = pd.DataFrame.from_dict(data, orient="index").sort_index()
    df["shift"] = df["shift"].apply(lambda x: x.tolist())
    df["scale"] = df["scale"].apply(lambda x: x.tolist())
    df.to_csv(os.path.join(output_path, mode, f"{mode}.csv"))
    print(df[["SSR", "SRR"]].describe())


if __name__ == "__main__":
    import fire

    fire.Fire()
