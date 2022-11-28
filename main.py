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

from scipy.signal import iirfilter

import numpy as np


def run_one(item):
    np.random.seed(42)
    spr_num, sfr_num, cost = distortion_ratio(
        item["xest"],
        item["xtrue"],
        use_numerical=True,
        iterative=False,
        use_exact_grad=True,
    )
    # spr_cal, sfr_cal = distortion_ratio(item["xest"], item["xtrue"], use_numerical=False)

    out_dict = {
        "spr/num": spr_num,
        # "spr/cal": np.round(spr_cal, 3),
        "sfr/num": sfr_num,
        # "sfr/cal": np.round(sfr_cal, 3),
        "cost": cost,
        **{k: item[k] for k in ["true_angle", "est_angle", "est_deviation", "file"]},
    }

    del item

    # pprint(out_dict)

    return out_dict


filter_dict = dict(
    N=20,
    Wn=[6000],
    btype="lowpass",
    analog=False,
    ftype="butter",
    fs=16000,
    output="sos",
)

ideal_filt_dict = lambda f, bt: dict(
    Wn=f,
    btype=bt,
    ftype="ideal",
    fs=16000,
)

right_delay_dict = lambda rd: dict(ldelay=0, rdelay=rd, ftype="delay")


def run_eval(
    dataset="TIMIT",
    room_dim=(3, 3),
    snr=None,
    rt60=0.150,
    src_dist=1.0,
    mic_spacing=0.60,
    directivity="CARDIOID",
    min_angle=0,
    max_angle=0,
    angle_step=45,
    min_error=-45,
    max_error=45,
    error_step=15,
    signal_filter_kwargs=right_delay_dict(1024),
    estim_filter_kwargs=right_delay_dict(1024),  # filter_dict,
    use_phase=True,
    use_exact_grad=False,
    subfolder=None,
):

    kwargs = locals()

    if signal_filter_kwargs is not None:
        if signal_filter_kwargs["ftype"] in ["ideal", "delay"]:
            signal_filter = signal_filter_kwargs
        else:
            signal_filter = iirfilter(**signal_filter_kwargs)
        kwargs["signal_filter_kwargs"] = signal_filter

    if estim_filter_kwargs is not None:
        if estim_filter_kwargs["ftype"] in ["ideal", "delay"]:
            estim_filter = estim_filter_kwargs
        else:
            estim_filter = iirfilter(**estim_filter_kwargs)
        kwargs["estim_filter_kwargs"] = estim_filter

    timit_dm = StereoDatamodule(**kwargs)

    # timit_dl = timit_dm.train_dataloader()

    results = []

    is_incomplete = False

    try:
        # if True:
        n_samples = (
            64 * timit_dm.train_dataset.n_angles * timit_dm.train_dataset.n_errors
        )
        cs = 256

        for i in tqdm(range(int(np.ceil(n_samples / cs)))):
            results += process_map(
                run_one,
                [
                    timit_dm.train_dataset[j]
                    for j in range(i * cs, min(n_samples, (i + 1) * cs))
                ],
                chunksize=4,
                max_workers=16,
            )

    except KeyboardInterrupt:
        is_incomplete = True
    finally:

        folder = f"./results/{subfolder +'/' if subfolder is not None else ''}{datetime.now().strftime('%Y%m%d%H%M%S')}"

        import json

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        df = pd.DataFrame(results)

        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "args.json"), "w") as f:
            json.dump(dict(kwargs), f, indent=4, cls=NumpyEncoder)
        df.to_csv(
            os.path.join(
                folder, f"results-{'incomplete' if is_incomplete else 'ok'}.csv"
            )
        )


def run_multiple(
    subfolder,
    signal_filter_kwargs_list=None,
    estim_filter_kwargs_list=[None]
    + [
        ideal_filt_dict([f/np.power(2, 1/6), f * np.power(2, 1/6)], "bandstop") for f in [
            125, 250, 500, 1000, 2000, 4000
        ]
    ]  # [125, 250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000]],
    # estim_filter_kwargs_list=[None] + [ideal_filt_dict([f], 'low') for f in [125, 250, 500, 1000, 2000, 4000, 6000, 7000]],
):

    if signal_filter_kwargs_list is None:
        signal_filter_kwargs_list = [None for _ in range(len(estim_filter_kwargs_list))]

    for sfk, efk in tqdm(zip(signal_filter_kwargs_list, estim_filter_kwargs_list)):
        run_eval(signal_filter_kwargs=sfk, estim_filter_kwargs=efk, subfolder=subfolder)


if __name__ == "__main__":
    import fire

    fire.Fire()
