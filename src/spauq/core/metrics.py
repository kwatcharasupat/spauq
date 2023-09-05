import typing
import numpy as np
from typing import Literal, Optional, Tuple
from .decomposition import compute_projection
from .preprocessing import (
    _ForgiveType,
    _AlignType,
    _ScaleType,
)

_BssEvalBackendType = Literal["fast_bss_eval", "museval"]
_BssEvalBackendDefault = "museval"


def _snr(signal: np.ndarray, noise: np.ndarray, eps: float = 1e-8):

    signal_energy = np.mean(np.square(signal))
    noise_energy = np.mean(np.square(noise))

    if noise_energy == 0 and signal_energy == 0:
        ratio = 1.0
    elif noise_energy == 0:
        ratio = np.inf
    elif signal_energy == 0:
        ratio = -np.inf
    else:
        ratio = signal_energy / noise_energy

    ratio = np.clip(ratio, a_min=eps, a_max=1.0 / eps)

    snr = 10 * np.log10(ratio)

    return snr


def _signal_to_spatial_distortion_ratio(
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
    )["SSR"]


def signal_to_spatial_distortion_ratio(
    reference: np.ndarray,
    estimate: np.ndarray,
    fs: int,
    *,
    return_framewise: bool = False,
):

    return _signal_to_spatial_distortion_ratio(
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
    return_cost: bool = False,
    return_shift: bool = False,
    return_scale: bool = False,
    forgive_mode: Optional[_ForgiveType] = None,
    align_mode: Optional[_AlignType] = None,
    align_use_diag_only: bool = True,
    max_global_shift_seconds: Optional[float] = None,
    max_segment_shift_seconds: Optional[float] = None,
    scale_mode: Optional[_ScaleType] = None,
    window_length: Optional[int] = None,
    hop_length: Optional[int] = None,
    tikhonov_lambda: float = 1e-6,
    verbose: bool = True,
):
    refs, ref_projs, est_projs, cost, shift, scale = compute_projection(
        reference,
        estimate,
        fs,
        forgive_mode=forgive_mode,
        align_mode=align_mode,
        align_use_diag_only=align_use_diag_only,
        max_global_shift_seconds=max_global_shift_seconds,
        max_segment_shift_seconds=max_segment_shift_seconds,
        scale_mode=scale_mode,
        window_length=window_length,
        hop_length=hop_length,
        tikhonov_lambda=tikhonov_lambda,
        verbose=verbose,
    )

    errs_spat = [ref_proj - ref for ref_proj, ref in zip(ref_projs, refs)]

    errs_resid = [
        est_proj - ref_proj for ref_proj, est_proj in zip(ref_projs, est_projs)
    ]

    duplex_snr = [_snr(ref, err_spat) for ref, err_spat in zip(refs, errs_spat)]

    resid_snr = [
        _snr(ref_proj, err_resid) for ref_proj, err_resid in zip(ref_projs, errs_resid)
    ]

    duplex_snr = np.stack(duplex_snr, axis=-1)
    resid_snr = np.stack(resid_snr, axis=-1)

    out = {
        "SSR": duplex_snr,
        "SRR": resid_snr,
    }

    if return_cost:
        out["cost"] = cost

    if return_shift:
        out["shift"] = shift

    if return_scale:
        out["scale"] = scale

    if return_framewise:
        return out
    else:
        return {k: np.median(v, axis=-1) for k, v in out.items()}


def spauq_eval(
    reference: np.ndarray,
    estimate: np.ndarray,
    fs: int,
    *,
    return_framewise: bool = False,
    return_cost: bool = False,
    return_shift: bool = False,
    return_scale: bool = False,
    verbose: bool = True,
    **kwargs,
):
    return _spauq_eval(
        reference,
        estimate,
        fs,
        return_framewise=return_framewise,
        return_cost=return_cost,
        return_shift=return_shift,
        return_scale=return_scale,
        verbose=verbose,
        **kwargs,
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
