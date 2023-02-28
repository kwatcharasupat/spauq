import typing
import numpy as np
from typing import Literal, Optional, Tuple
from .decomposition import compute_projection
from .preprocessing import (
    _ForgiveType,
    _AlignType,
    _ScaleType,
)
import fast_bss_eval as fbe

_BssEvalBackendType = Literal["fast_bss_eval", "museval"]
_BssEvalBackendDefault = "museval"


def _snr(signal: np.ndarray, noise: np.ndarray):

    signal_energy = np.sum(np.square(signal), axis=-1)
    noise_energy = np.sum(np.square(noise), axis=-1)

    snr = 10 * np.log10(signal_energy / noise_energy)

    snr[noise_energy == 0] = np.inf

    return snr


def _source_image_to_duplex_distortion_ratio(
    reference: np.ndarray,
    estimate: np.ndarray,
    fs: int,
    *,
    return_framewise: bool = False,
    forgive_mode: Optional[_ForgiveType] = None,
    align_mode: Optional[_AlignType] = None,
    align_use_diag_only: bool = True,
    scale_mode: Optional[_ScaleType] = None,
    window_length: Optional[int] = None,
    hop_length: Optional[int] = None,
    tikhonov_lambda: float = 1e-6,
):

    return _spauq_eval(
        reference,
        estimate,
        fs,
        return_framewise=return_framewise,
        forgive_mode=forgive_mode,
        align_mode=align_mode,
        align_use_diag_only=align_use_diag_only,
        scale_mode=scale_mode,
        window_length=window_length,
        hop_length=hop_length,
        tikhonov_lambda=tikhonov_lambda,
    )["IDR"]


def source_image_to_duplex_distortion_ratio(
    reference: np.ndarray,
    estimate: np.ndarray,
    fs: int,
    *,
    return_framewise: bool = False,
):

    return _source_image_to_duplex_distortion_ratio(
        reference,
        estimate,
        fs,
        return_framewise=return_framewise,
    )


def _signal_to_residual_distortion_ratio(
    reference: np.ndarray,
    estimate: np.ndarray,
    fs: int,
    *,
    return_framewise: bool = False,
    forgive_mode: Optional[_ForgiveType] = None,
    align_mode: Optional[_AlignType] = None,
    align_use_diag_only: bool = True,
    scale_mode: Optional[_ScaleType] = None,
    window_length: Optional[int] = None,
    hop_length: Optional[int] = None,
    tikhonov_lambda: float = 1e-6,
):
    return _spauq_eval(
        reference,
        estimate,
        fs,
        return_framewise=return_framewise,
        forgive_mode=forgive_mode,
        align_mode=align_mode,
        align_use_diag_only=align_use_diag_only,
        scale_mode=scale_mode,
        window_length=window_length,
        hop_length=hop_length,
        tikhonov_lambda=tikhonov_lambda,
    )["SRR"]


def signal_to_residual_distortion_ratio(
    reference: np.ndarray,
    estimate: np.ndarray,
    fs: int,
    *,
    return_framewise: bool = False,
):
    return _signal_to_residual_distortion_ratio(
        reference,
        estimate,
        fs,
        return_framewise=return_framewise,
    )


def _spauq_eval(
    reference: np.ndarray,
    estimate: np.ndarray,
    fs: int,
    *,
    return_framewise: bool = False,
    forgive_mode: Optional[_ForgiveType] = None,
    align_mode: Optional[_AlignType] = None,
    align_use_diag_only: bool = True,
    scale_mode: Optional[_ScaleType] = None,
    window_length: Optional[int] = None,
    hop_length: Optional[int] = None,
    tikhonov_lambda: float = 1e-6,
):
    refs, ref_projs, est_projs, cost, shift, scale = compute_projection(
        reference,
        estimate,
        fs,
        forgive_mode=forgive_mode,
        align_mode=align_mode,
        align_use_diag_only=align_use_diag_only,
        scale_mode=scale_mode,
        window_length=window_length,
        hop_length=hop_length,
        tikhonov_lambda=tikhonov_lambda,
    )

    errs_spat = [ref_proj - ref for ref_proj, ref in zip(ref_projs, refs)]

    errs_resid = [
        est_proj - ref_proj for ref_proj, est_proj in zip(ref_projs, est_projs)
    ]

    duplex_snr = [_snr(ref, err_spat) for ref, err_spat in zip(refs, errs_spat)]

    resid_snr = [_snr(ref, err_resid) for ref, err_resid in zip(refs, errs_resid)]

    duplex_snr = np.stack(duplex_snr, axis=-1)
    resid_snr = np.stack(resid_snr, axis=-1)

    if return_framewise:
        return {
            "IDR": duplex_snr,
            "SRR": resid_snr,
        }

    return {
        "IDR": np.mean(duplex_snr, axis=-1),
        "SRR": np.mean(resid_snr, axis=-1),
    }


def spauq_eval(
    reference: np.ndarray,
    estimate: np.ndarray,
    fs: int,
    *,
    return_framewise: bool = False,
):
    return _spauq_eval(
        reference,
        estimate,
        fs,
        return_framewise=return_framewise,
    )

# TODO
def _spauq_bss_eval(
    reference: np.ndarray,
    estimate: np.ndarray,
    fs: int,
    *,
    return_framewise: bool = False,
    forgive_mode: Optional[_ForgiveType] = None,
    align_mode: Optional[_AlignType] = None,
    align_use_diag_only: bool = True,
    scale_mode: Optional[_ScaleType] = None,
    window_length: Optional[int] = None,
    hop_length: Optional[int] = None,
    tikhonov_lambda: float = 1e-6,
    backend: _BssEvalBackendType = _BssEvalBackendDefault,
):
    
    raise NotImplementedError("spauq_bss_eval is not implemented yet")

    assert backend in typing.get_args(
        _BssEvalBackendType
    ), f"Invalid backend: {backend}"

    refs, ref_projs, est_projs, cost, shift, scale = compute_projection(
        reference,
        estimate,
        fs,
        forgive_mode=forgive_mode,
        align_mode=align_mode,
        align_use_diag_only=align_use_diag_only,
        scale_mode=scale_mode,
        window_length=window_length,
        hop_length=hop_length,
        tikhonov_lambda=tikhonov_lambda,
    )

    if backend == "fast_bss_eval":
        raise NotImplementedError("fast_bss_eval is not implemented yet")
    elif backend == "museval":
        raise NotImplementedError("museval is not implemented yet")
    else:
        raise ValueError(f"Invalid backend: {backend}")
