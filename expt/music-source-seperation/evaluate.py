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
    r, e = inputs
    filename = r.split("/")[-2]
    r = sf.read(r)[0]
    e = sf.read(e)[0]

    # print(r.shape, e.shape)
    assert r.shape == e.shape

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
    output_path="/home/kwatchar3/spauq-home/spauq/expt/music-source-seperation/results-2s",
    sources=["vocals", "drums", "bass", "other"],
    fs=MUSDB_FS,
):

    if estimate_path is None:
        estimate_path = _EstimatePathFormat.format(model_name=model_name)

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
    df.to_csv(os.path.join(output_path, f"{model_name}.csv"))
    print(df[["SSR", "SRR"]].describe())


# def evaluate_all():
#     models = ["HDemucs-HQ", "HDemucs-HQPLUS", "OpenUnmix-umxhq", "Spleeter-4stems"]

#     for model in models:
#         try:
#             evaluate(model)
#         except Exception as e:
#             print(e)


if __name__ == "__main__":
    import fire

    fire.Fire()
