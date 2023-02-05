import os
from pprint import pprint
from data.dataloader import StereoDatamodule
import pandas as pd
from tqdm import tqdm

from museval import evaluate

from datetime import datetime

from eval.decomposition import distortion_ratio, project_spatial_magphase

import numpy as np

from tqdm.contrib.concurrent import process_map

from scipy.signal import correlate

import numpy as np


def run_one(item):
    np.random.seed(42)
    x = item["xtrue"][0, :]

    acf = correlate(x, x, mode="full")

    return acf


def run_eval(
    dataset="TIMIT",
    n_files=100,
    subfolder=None,
    multiprocessing=True,
):


    timit_dm = StereoDatamodule(dataset=dataset, min_pan=0, max_pan=0, max_error=0, min_error=0)

    results = []

    is_incomplete = False

    try:
        # if True:
        n_samples = (
            n_files * timit_dm.train_dataset.n_pans * timit_dm.train_dataset.n_errors
        )
        cs = 128
        if multiprocessing:
            for i in tqdm(range(int(np.ceil(n_samples / cs)))):

                results += process_map(
                    run_one,
                    [
                        timit_dm.train_dataset[j]
                        for j in range(i * cs, min(n_samples, (i + 1) * cs))
                    ],
                    chunksize=1,
                    max_workers=16,
                )
        else:
            for i in tqdm(range(n_samples)):
                results.append(run_one(timit_dm.train_dataset[i]))

    except KeyboardInterrupt:
        is_incomplete = True
    finally:

        folder = f"./results/{subfolder +'/' if subfolder is not None else ''}{datetime.now().strftime('%Y%m%d%H%M%S')}"

        if not os.path.exists(folder):
            os.makedirs(folder)

        results = np.array(results, dtype=object)
        np.savez_compressed(f"{folder}/acf.npz", results)



if __name__ == "__main__":
    import fire

    fire.Fire()
