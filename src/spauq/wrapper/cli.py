import glob
import os.path

import fire
import numpy as np
from tqdm import tqdm

from ..core.metrics import spauq_eval
import soundfile as sf

_ALLOWED_SAVE_FORMATS = ["npz", "npz_compressed"]

def spauq_eval_file(
    reference_path: str,
    estimate_path: str,
    result_dir: str = None,
    result_name: str = None,
    save_format: str = "npz",
    *,
    return_framewise: bool = False,
    return_cost: bool = False,
    return_shift: bool = False,
    return_scale: bool = False,
    verbose: bool = True,
    **kwargs
):

    assert save_format in _ALLOWED_SAVE_FORMATS

    reference, fsr = sf.read(reference_path, always_2d=True)
    estimate, fse = sf.read(estimate_path, always_2d=True)

    reference = reference.T
    estimate = estimate.T

    assert fsr == fse
    out = spauq_eval(
        reference,
        estimate,
        fsr,
        return_framewise=return_framewise,
        return_cost=return_cost,
        return_shift=return_shift,
        return_scale=return_scale,
        verbose=verbose,
        **kwargs
    )

    if result_dir is None:
        result_dir = os.path.dirname(estimate_path)

    os.makedirs(result_dir, exist_ok=True)

    if result_name is None:
        result_name = os.path.splitext(os.path.basename(estimate_path))[0]

    if save_format in ["npz", "npz_compressed"]:
        extension = ".npz"
    else:
        raise ValueError(f"`save_format` must be one of {_ALLOWED_SAVE_FORMATS}")

    result_path = os.path.join(result_dir, result_name + extension)

    if save_format == "npz":
        np.savez(result_path, **out)
    elif save_format == "npz_compressed":
        np.savez_compressed(result_path, **out)
    else:
        raise ValueError(f"`save_format` must be one of {_ALLOWED_SAVE_FORMATS}")


def spauq_eval_dir(
    reference_path: str,
    estimate_dir: str,
    estimate_ext: str = ".wav",
    result_dir: str = None,
    result_name_format: str = None,
    save_format: str = "npz",
    *,
    return_framewise: bool = False,
    return_cost: bool = False,
    return_shift: bool = False,
    return_scale: bool = False,
    verbose: bool = True,
    **kwargs
):

    estimate_paths = glob.glob(
        os.path.join(estimate_dir, "*" + estimate_ext)
    )

    estimate_paths = [f for f in estimate_paths if not os.path.samefile(f, reference_path)]

    for estimate_path in tqdm(estimate_paths):
        tqdm.write(f"Evaluating {os.path.basename(estimate_path)}")
        if result_name_format is not None:
            result_name = result_name_format.format(
                estimate_path=os.path.splitext(os.path.basename(estimate_path))[0]
            )
        else:
            result_name = None

        spauq_eval_file(
            reference_path,
            estimate_path,
            result_dir=result_dir,
            result_name=result_name,
            save_format=save_format,
            return_framewise=return_framewise,
            return_cost=return_cost,
            return_shift=return_shift,
            return_scale=return_scale,
            verbose=verbose,
            **kwargs
        )


if __name__ == "__main__":
    fire.Fire()
