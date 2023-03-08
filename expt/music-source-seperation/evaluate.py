from collections import defaultdict
from pprint import pprint
import pandas as pd
import tqdm
import os

import glob

import sys

sys.path.append("/home/kwatchar3/spauq-home/spauq/src")
from spauq.core.metrics import spauq_eval

import soundfile as sf

_EstimatePathFormat = "/home/kwatchar3/spauq-home/data/musdb-hq/{model_name}"

def evaluate(
    model_name,
    estimate_path=None,
    reference_path="/home/kwatchar3/spauq-home/data/musdb-hq/raw/musdb18hq/test",
    output_path="/home/kwatchar3/spauq-home/spauq/expt/music-source-seperation/",
    sources=["vocals", "drums", "bass", "other"],
):

    if estimate_path is None:
        estimate_path = _EstimatePathFormat.format(model_name=model_name)

    data = defaultdict(dict)

    for s in sources:
        ref = glob.glob(os.path.join(reference_path, "**", f"{s}.wav"), recursive=True)
        est = [r.replace(reference_path, estimate_path) for r in ref]
        for r, e in tqdm.tqdm(zip(ref, est), total=len(ref)):
            filename = r.split("/")[-2]
            r, fs = sf.read(r)
            e, _ = sf.read(e)
            metrics = spauq_eval(
                reference=r.T,
                estimate=e.T,
                fs=fs,
                return_framewise=False,
                return_cost=True,
                return_shift=True,
                return_scale=True,
            )

            print(filename, s)
            pprint(metrics)

            data[(filename, s)] = metrics

    df = pd.DataFrame.from_dict(data, orient="index").sort_index()
    df.to_csv(os.path.join(output_path, f"{model_name}.csv"))
    print(df[["SSR", "SRR"]].describe())


if __name__ == "__main__":
    import fire
    fire.Fire(evaluate)
