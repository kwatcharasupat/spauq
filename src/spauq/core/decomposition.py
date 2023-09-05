import warnings
import numpy as np
import numpy.typing as npt
from typing import Any, List, Optional, Tuple

from numpy import ndarray
from scipy import signal as sps

# from numba import jit

from .preprocessing import (
    _ForgiveType,
    _AlignType,
    _ScaleType,
    _validate_inputs,
    _apply_global_forgive,
)

__all__ = ["compute_projection"]

_DefaultWindowLengthSeconds = 1.0
_DefaultHopLengthSeconds = 0.5
_DefaultMaximumGlobalShiftSeconds = np.inf
_DefaultMaximumSegmentShiftSeconds = 0.1
_DefaultSilenceThreshold = 1e-8


def _check_interchannel_leakage(
    reference: np.ndarray,
    estimate: np.ndarray,
):
    pass


def _project_shift(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    max_shift_samples: Optional[int] = None,
    silence_threshold: float = _DefaultSilenceThreshold,
):

    assert reference.shape == estimate.shape
    assert reference.ndim == 2
    assert max_shift_samples is not None

    n_chan, n_sampl = reference.shape[-2:]

    full_lags = sps.correlation_lags(n_sampl, n_sampl, mode="full")
    if np.isfinite(max_shift_samples):
        max_shift_filter = np.abs(full_lags) <= max_shift_samples
        lags = full_lags[max_shift_filter]
    else:
        max_shift_filter = None
        lags = full_lags

    optim_lags = np.zeros((n_chan, n_chan), dtype=int)

    for i in range(n_chan):

        if np.std(reference[i]) < silence_threshold:
            # print("reference is silent at channel", i, np.std(reference[i]))
            continue

        for j in range(n_chan):

            if np.std(estimate[j]) < silence_threshold:
                # print("estimate is silent at channel", j, np.std(estimate[j]))
                # print(estimate[j])
                continue

            corr = sps.correlate(reference[i], estimate[j], mode="full", method="fft")
            if np.isfinite(max_shift_samples):
                corr = corr[max_shift_filter]
            optim_lags[i, j] = lags[np.argmax(np.abs(corr))]

    # print(optim_lags)

    min_lags = np.min(optim_lags)
    max_lags = np.max(optim_lags)

    shifted_ref = np.zeros((n_chan, n_chan, n_sampl), dtype=reference.dtype)
    masked_ref = np.tile(reference, (n_chan, 1, 1))
    # np.zeros((n_chan, n_chan, n_sampl), dtype=reference.dtype)

    for i in range(n_chan):
        for j in range(n_chan):
            shifted_ref[i, j] = np.roll(reference[i], -optim_lags[i, j])

    # print(np.round(shifted_ref[0, :, :16], 3))

    # if min_lags < 0 and max_lags < 0: then abs(min_lags) > abs(max_lags), only deal with min_lags
    # if min_lags > 0 and max_lags > 0: then abs(max_lags) > abs(min_lags), only deal with max_lags
    # if min_lags < 0 and max_lags > 0: then max_lags > min_lags, deal with both
    # if min_lags > 0 and max_lags < 0: cannot happen

    # if estimate is behind, lag is negative --> reference is ahead and gets pulled backward
    # usable region of estimate: -lags to end
    # unusable region of estimate: 0 to -lags

    # if estimate is ahead, lag is positive --> reference is behind and gets pulled forward
    # usable region of estimate: 0 to -lags
    # unusable region of estimate: -lags to end

    mask = np.ones((n_sampl,), dtype=bool)
    if min_lags < 0:
        mask[:-min_lags] = False
        if max_lags > 0:
            mask[-max_lags:] = False
    elif min_lags > 0 and max_lags > 0:
        mask[-max_lags:] = False

    shifted_ref = shifted_ref[:, :, mask]
    estimate = estimate[:, mask]
    masked_ref = masked_ref[:, :, mask]

    return masked_ref, shifted_ref, estimate, optim_lags


def _compute_corrs(
    shifted_reference: np.ndarray,
    shifted_estimate: np.ndarray,
):
    n_chan, _, _ = shifted_reference.shape
    # chan of estimate, chan of reference, sample

    xcf = np.sum(shifted_reference * shifted_estimate[:, None, :], axis=-1)
    # (chan of estimate, chan of reference)

    acf = np.sum(
            shifted_reference[:, :, None, :] * shifted_reference[:, None, :, :],
            axis=-1,
        )

    return acf, xcf


def _compute_projections(
    shifted_reference: np.ndarray,
    scale: np.ndarray,
):
    _, n_chan, n_sampl = shifted_reference.shape
    ref_proj = np.zeros((n_chan, n_sampl), dtype=shifted_reference.dtype)

    for cest in range(n_chan):
        ref_proj[cest, :] = scale[cest, :, :] @ shifted_reference[cest, :, :]

    return ref_proj


def _project_scale(
    shifted_reference: np.ndarray,
    shifted_estimate: np.ndarray,
    tikhonov_lambda: float = 1e-6,
    silence_threshold: float = _DefaultSilenceThreshold,
):
    assert shifted_reference.ndim == 3

    n_chan, n_chan, _ = shifted_reference.shape

    # print(shifted_reference.shape)
    # print(np.round(shifted_reference[:, 0, :16], 3))
    # chan of reference, chan of estimate, sample

    acf, xcf = _compute_corrs(shifted_reference, shifted_estimate)

    shifted_reference_t = np.ascontiguousarray(shifted_reference.transpose((1, 0, 2)))
    # (chan of estimate, chan of reference, sample)

    # acf.shape = (n_chan_e, n_chan_r, n_chan_r)
    # xcf.shape = (n_chan_e, n_chan_r)

    # TODO: detect singular matrix
    try:
        scaleT, _, _, _ = np.linalg.lstsq(acf.T, xcf.T, rcond=None)
        scale = scaleT.T
    except: # np.linalg.LinAlgError:
        warnings.warn(
            "Singular matrix in projection, using Tikhonov regularization",
            RuntimeWarning,
        )
        # scale @ acf = xcf
        # acf.T @ scale.T = xcf.T
        # acf @ acf.T @ scale.T = acf @ xcf.T
        # (acf @ acf.T + lambd * I) @ scale.T = acf @ xcf.T
        # scale.T = (acf @ acf.T + lambd * I)^-1 @ acf @ xcf.T


    estimate_std = np.std(shifted_estimate, axis=-1)
    estimate_nonsilent = estimate_std > silence_threshold # (n_chan,)

    reference_std = np.std(shifted_reference_t, axis=-1)
    reference_nonsilent = reference_std > silence_threshold # (n_chan_e, n_chan_r)

    scale = np.zeros((n_chan, n_chan), dtype=acf.dtype)

    for cest in range(n_chan):
        if not estimate_nonsilent[cest]:
            continue

        active_ref = reference_nonsilent[cest, :]
        n_active_ref = int(np.sum(active_ref))

        if n_active_ref == 0:
            continue

        acf_cest = acf[cest][active_ref, :][:, active_ref] # (n_chan_r, n_chan_r)
        xcf_cest = xcf[cest][active_ref][None, :] # (1, n_chan_r)

        # TODO: detect singular matrix
        if np.linalg.matrix_rank(acf_cest) == n_active_ref:
            # scale @ acf = xcf
            # acf.T @ scale.T = xcf.T
            scaleTcest, _, _, _ = np.linalg.lstsq(acf_cest.T, xcf_cest.T, rcond=None)
        else:
            # scale @ acf = xcf
            # acf.T @ scale.T = xcf.T
            # acf @ acf.T @ scale.T = acf @ xcf.T
            # (acf @ acf.T + lambd * I) @ scale.T = acf @ xcf.T
            # scale.T = (acf @ acf.T + lambd * I)^-1 @ acf @ xcf.T

            acf2 = acf_cest @ acf_cest.T
            axcf = acf_cest @ xcf_cest.T
            reg = tikhonov_lambda * np.eye(n_active_ref)
            scaleTcest, _, _, _ = np.linalg.lstsq(acf2 + reg, axcf, rcond=None)

        scale[cest][active_ref] = scaleTcest[:, 0]

    ref_proj = _compute_projections(
        shifted_reference_t,
        np.ascontiguousarray(scale[:, None, :]),
    )

    return ref_proj, scale


# @jit(nopython=True)
def _compute_cost(reference: np.ndarray, estimate: np.ndarray):
    return np.linalg.norm(reference - estimate)


def _compute_framewise_projection(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    tikhonov_lambda: float = 1e-6,
    max_shift_samples: Optional[int] = None,
):

    ref_shifted, ref_proj, est_proj, shift = _project_shift(
        reference, estimate, max_shift_samples=max_shift_samples
    )

    ref_proj, scale = _project_scale(
        ref_shifted, est_proj, tikhonov_lambda=tikhonov_lambda
    )

    cost = _compute_cost(ref_proj, est_proj)

    return ref_shifted, ref_proj, est_proj, cost, shift, scale


def compute_projection(
    reference: npt.ArrayLike,
    estimate: npt.ArrayLike,
    fs: int,
    *,
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
) -> Tuple[List[Any], List[ndarray], List[Any], ndarray, ndarray, ndarray]:

    reference, estimate = _validate_inputs(
        reference, estimate, forgive_mode=forgive_mode
    )

    if max_segment_shift_seconds is None:
        max_segment_shift_seconds = _DefaultMaximumSegmentShiftSeconds

    if max_global_shift_seconds is None:
        max_global_shift_seconds = _DefaultMaximumGlobalShiftSeconds

    reference, estimate = _apply_global_forgive(
        reference=reference,
        estimate=estimate,
        forgive_mode=forgive_mode,
        align_mode=align_mode,
        max_shift_samples=max_global_shift_seconds * fs,
        align_use_diag_only=align_use_diag_only,
        scale_mode=scale_mode,
        verbose=verbose,
    )

    n_chan = reference.shape[-2]

    # compute projection

    if window_length is None:
        window_length = (
            int(_DefaultWindowLengthSeconds * fs)
            if np.isfinite(_DefaultWindowLengthSeconds)
            else _DefaultWindowLengthSeconds
        )
    if hop_length is None and np.isfinite(window_length):
        hop_length = int(_DefaultHopLengthSeconds * fs)

    n_sampl = reference.shape[-1]
    if np.isposinf(window_length):
        n_frames = 1
    else:

        if n_sampl < window_length:
            raise ValueError(
                "The input signal is too short to be decomposed into frames."
            )

        n_frames = int(np.ceil((n_sampl - window_length) / hop_length) + 1)

    refs = []
    refprojs = []
    estprojs = []
    costs = np.full((n_frames,), np.nan)
    shifts = np.full((n_chan, n_chan, n_frames), np.nan)
    scales = np.full((n_chan, n_chan, n_frames), np.nan)

    if n_frames == 1:
        if verbose:
            warnings.warn(
                "The input signal is too short to be decomposed into frames. "
                "The entire signal is used as a single frame."
            )
        starts = [0]
        ends = [n_sampl]
    else:
        starts = np.arange(0, n_frames) * hop_length
        ends = starts + window_length

    for i in range(n_frames):
        ref, refproj, estproj, cost, shift, scale = _compute_framewise_projection(
            reference=reference[..., starts[i] : ends[i]],
            estimate=estimate[..., starts[i] : ends[i]],
            tikhonov_lambda=tikhonov_lambda,
            max_shift_samples=int(np.round(max_segment_shift_seconds * fs)),
        )

        refs.append(ref)
        refprojs.append(refproj)
        estprojs.append(estproj)

        costs[i] = cost
        shifts[..., i] = shift
        scales[..., i] = scale

    return refs, refprojs, estprojs, costs, shifts, scales
