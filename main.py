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
        **{k: item[k] for k in ["true_pan", "est_pan", "est_deviation", "file"]},
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

leftright_delay_dict = lambda rd: dict(ldelay=rd, rdelay=rd, ftype="delay")


def run_eval(
    dataset,
    min_pan=0,
    max_pan=0,
    pan_step=45,
    min_error=-45,
    max_error=45,
    error_step=15,
    signal_filter_kwargs=right_delay_dict(1024),
    estim_filter_kwargs=right_delay_dict(1024),  # filter_dict,
    use_phase=True,
    use_exact_grad=False,
    n_files=128,
    subfolder=None,
    multiprocessing=True,
    **dskwargs,
):

    print(dskwargs)

    kwargs = locals()

    # if signal_filter_kwargs is not None:
    #     if signal_filter_kwargs["ftype"] in ["ideal"]:
    #         signal_filter = signal_filter_kwargs
    #     else:
    #         signal_filter = iirfilter(**signal_filter_kwargs)
    #     kwargs["signal_filter_kwargs"] = signal_filter

    # if estim_filter_kwargs is not None:
    #     if estim_filter_kwargs["ftype"] in ["ideal"]:
    #         estim_filter = estim_filter_kwargs
    #     else:
    #         estim_filter = iirfilter(**estim_filter_kwargs)
    #     kwargs["estim_filter_kwargs"] = estim_filter

    timit_dm = StereoDatamodule(**kwargs, **dskwargs)

    # timit_dl = timit_dm.train_dataloader()

    results = []

    is_incomplete = False

    try:
        # if True:
        n_samples = (
            n_files * timit_dm.train_dataset.n_pans * timit_dm.train_dataset.n_errors
        )
        cs = 1024
        if multiprocessing:
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
        else:
            for i in tqdm(range(n_samples)):
                results.append(run_one(timit_dm.train_dataset[i]))

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
    dataset="TIMIT",
    multiprocessing=True,
    signal_filter_kwargs_list=None,
    estim_filter_kwargs_list=[[]],
    **kwargs,
):

    if signal_filter_kwargs_list is None:
        signal_filter_kwargs_list = [[] for _ in range(len(estim_filter_kwargs_list))]

    for sfk, efk in tqdm(zip(signal_filter_kwargs_list, estim_filter_kwargs_list)):
        print(efk)
        run_eval(
            dataset=dataset,
            signal_filter_kwargs=sfk,
            estim_filter_kwargs=efk,
            subfolder=subfolder,
            multiprocessing=multiprocessing,
            **kwargs,
        )


def run_pow2_delays(
    subfolder,
    dataset,
    multiprocessing=True,
    max_delay=4096,
    sampling_rate=None,
    **kwargs,
):  

    if sampling_rate is None:
        if dataset == "TIMIT":
            sampling_rate = 16000
        elif dataset == "MUSDB18-HQ":
            sampling_rate = 44100
        else:
            raise ValueError(f"Unknown dataset {dataset}")

    estim_filter_kwargs_list = [[]]

    p2 = int(np.log2(max_delay))

    for ldp in range(-1, p2+1):
        ld = np.power(2.0, ldp)/sampling_rate if ldp >= 0 else 0
        for rdp in range(-1, p2+1):
            rd = np.power(2.0, rdp)/sampling_rate if rdp >= 0 else 0

            estim_filter_kwargs_list.append(
                [
                    {
                    "backend": "custom",
                    "name": "delay",
                    "kwargs": {
                        "positions": [ld, rd],
                    }
                }
                ]
            )
    
    pprint(estim_filter_kwargs_list)

    run_multiple(
        subfolder=subfolder,
        dataset=dataset,
        multiprocessing=multiprocessing,
        estim_filter_kwargs_list=estim_filter_kwargs_list,
        signal_filter_kwargs_list=None,
        **kwargs,
    )



def run_lin_delays(
    subfolder,
    dataset,
    multiprocessing=True,
    max_delay=64,
    sampling_rate=None,
    **kwargs,
):  

    if sampling_rate is None:
        if dataset == "TIMIT":
            sampling_rate = 16000
        elif dataset == "MUSDB18-HQ":
            sampling_rate = 44100
        else:
            raise ValueError(f"Unknown dataset {dataset}")

    estim_filter_kwargs_list = [[]]

    for ldp in range(4, max_delay, 4):
        ld = ldp/sampling_rate
        for rdp in range(0, max_delay):
            rd = rdp/sampling_rate
            estim_filter_kwargs_list.append(
                [
                    {
                    "backend": "custom",
                    "name": "delay",
                    "kwargs": {
                        "positions": [ld, rd],
                    }
                }
                ]
            )
    
    pprint(estim_filter_kwargs_list)

    run_multiple(
        subfolder=subfolder,
        dataset=dataset,
        multiprocessing=multiprocessing,
        estim_filter_kwargs_list=estim_filter_kwargs_list,
        signal_filter_kwargs_list=None,
        **kwargs,
    )



if __name__ == "__main__":
    import fire

    fire.Fire()
