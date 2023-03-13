from collections import defaultdict
from pprint import pprint
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import os

import glob

import sys

sys.path.append("/home/kwatchar3/spauq-home/spauq/src")
from spauq.core.metrics import spauq_eval

import soundfile as sf

_EstimatePathFormat = "/home/kwatchar3/spauq-home/data/musdb-hq/{codec}/{setting}/wav"
MUSDB_FS = 44100


def evaluate_one(inputs):
    r, e = inputs
    filename = r.split("/")[-2]
    r, fs0 = sf.read(r)
    e, fs = sf.read(e)

    if e.shape[0] > r.shape[0]:
        e = e[: r.shape[0]]

    if r.shape == e.shape:
        metrics = spauq_eval(
            reference=r.T,
            estimate=e.T,
            fs=MUSDB_FS,
            return_framewise=False,
            return_cost=True,
            return_shift=True,
            return_scale=True,
            verbose=False,
        )
        return filename, metrics
    else:
        print("Bad!")
        return None

def evaluate_musdb(
    codec, setting,
    estimate_path=None,
    reference_path="/home/kwatchar3/spauq-home/data/musdb-hq/raw/musdb18hq/test",
    output_path="/home/kwatchar3/spauq-home/spauq/expt/codec/musdb/results-2s",
    sources=["mixture"],
    fs=MUSDB_FS,
):

    if estimate_path is None:
        estimate_path = _EstimatePathFormat.format(codec=codec, setting=setting)

    data = defaultdict(dict)

    for s in sources:
        print("Evaluating", s)
        ref = sorted(glob.glob(os.path.join(reference_path, "**", f"{s}.wav"), recursive=True))
        est = [r.replace(reference_path, estimate_path) for r in ref]

        fnmetrics = process_map(
            evaluate_one,
            zip(ref, est),
            max_workers=4,
            total=len(ref),
        )
        # fnmetrics = [evaluate_one(x) for x in zip(ref, est)]

        filenames = [fn for fn, _ in fnmetrics]
        metrics = [m for _, m in fnmetrics]

        for filename, metric in zip(filenames, metrics):
            data[(filename, s)] = metric

            print(filename)
            pprint(metrics)

    os.makedirs(output_path, exist_ok=True)

    df = pd.DataFrame.from_dict(data, orient="index").sort_index()
    df["shift"] = df["shift"].apply(lambda x: x.tolist())
    df["scale"] = df["scale"].apply(lambda x: x.tolist())
    df.to_csv(os.path.join(output_path, f"{codec}-{setting}.csv"))
    print(df[["SSR", "SRR"]].describe())


def evaluate_all_mp3():
    codec = "mp3"
    settings = glob.glob("/home/kwatchar3/spauq-home/data/musdb-hq/mp3/*", recursive=False)

    for setting in tqdm(settings):
        try:
            evaluate_musdb(codec, setting)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    import fire

    fire.Fire()
