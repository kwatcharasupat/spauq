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

_EstimatePathFormat = "/home/kwatchar3/spauq-home/data/musdb-hq/{model_name}"
MUSDB_FS = 44100


def evaluate_one(inputs):
    r, e, filename = inputs
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


def evaluate(
    model_name,
    estimate_path=None,
    reference_path="/home/kwatchar3/spauq-home/data/musdb-hq/raw/musdb18hq/test",
    output_path="/home/kwatchar3/spauq-home/spauq/expt/music-source-seperation/results",
    sources=["vocals", "drums", "bass", "other"],
    fs=MUSDB_FS,
):

    if estimate_path is None:
        estimate_path = _EstimatePathFormat.format(model_name=model_name)

    data = defaultdict(dict)

    for s in sources:
        print("Evaluating", s)
        ref = glob.glob(os.path.join(reference_path, "**", f"{s}.wav"), recursive=True)
        est = [r.replace(reference_path, estimate_path) for r in ref]

        ref_signal = [sf.read(r)[0] for r in tqdm(ref)]
        est_signal = [sf.read(e)[0] for e in tqdm(est)]
        filenames = [r.split("/")[-2] for r in ref]

        fnmetrics = process_map(
            evaluate_one,
            zip(ref_signal, est_signal, filenames),
            max_workers=2,
            total=len(ref),
        )

        filenames = [fn for fn, _ in fnmetrics]
        metrics = [m for _, m in fnmetrics]

        for filename, metric in zip(filenames, metrics):
            data[(filename, s)] = metric

            print(filename)
            pprint(metrics)

    df = pd.DataFrame.from_dict(data, orient="index").sort_index()
    df["shift"] = df["shift"].apply(lambda x: x.tolist())
    df["scale"] = df["scale"].apply(lambda x: x.tolist())
    df.to_csv(os.path.join(output_path, f"{model_name}.csv"))
    print(df[["SSR", "SRR"]].describe())


if __name__ == "__main__":
    import fire

    fire.Fire(evaluate)
