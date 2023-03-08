import warnings
import numpy as np
import numpy.typing as npt
from typing import Optional, Tuple
from scipy import signal as sps

from .preprocessing import (
    _ForgiveType,
    _AlignType,
    _ScaleType,
    _validate_inputs,
    _apply_global_forgive,
)

__all__ = ["compute_projection"]

_DefaultWindowLengthSeconds = 30  # follows bsseval
_DefaultHopLengthSeconds = 15  # follows bsseval
_DefaultMaximumGlobalShiftSeconds = np.inf
_DefaultMaximumSegmentShiftSeconds = 1.0


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
        lags = full_lags

    optim_lags = np.zeros((n_chan, n_chan), dtype=int)

    for i in range(n_chan):
        for j in range(n_chan):
            corr = sps.correlate(reference[i], estimate[j], mode="full")
            if np.isfinite(max_shift_samples):
                corr = corr[max_shift_filter]
            optim_lags[i, j] = lags[np.argmax(np.abs(corr))]

    min_lags = np.min(optim_lags)
    max_lags = np.max(optim_lags)

    # if estimate is behind, lag is negative --> reference is ahead
    # usable region of estimate: abs(lags) to end

    # if estimate is ahead, lag is positive --> reference is behind
    # usable region of estimate: 0 to end - abs(lags)

    shifted_ref = np.zeros((n_chan, n_chan, n_sampl), dtype=reference.dtype)
    masked_ref = np.zeros((n_chan, n_chan, n_sampl), dtype=reference.dtype)

    for i in range(n_chan):
        for j in range(n_chan):
            shifted_ref[i, j] = np.roll(reference[i], -optim_lags[i, j])
            masked_ref[i, j] = reference[i]

    mask = np.ones((n_sampl,), dtype=bool)

    if min_lags < 0:
        mask[min_lags:] = False
        if max_lags < 0:
            pass
        elif max_lags > 0:
            mask[:max_lags] = False
    elif min_lags > 0:
        mask[:min_lags] = False
        if max_lags < 0:
            pass
        elif max_lags > 0:
            mask[:max_lags] = False

    # print("n_sampl", n_sampl, "min_lags", min_lags, "max_lags", max_lags)

    shifted_ref = shifted_ref[:, :, mask]
    estimate = estimate[:, mask]
    masked_ref = masked_ref[:, :, mask]

    return masked_ref, shifted_ref, estimate, optim_lags


def _project_scale(
    shifted_reference: np.ndarray,
    shifted_estimate: np.ndarray,
    tikhonov_lambda: float = 1e-6,
):
    assert shifted_reference.ndim == 3

    n_chan, n_chan, n_sampl = shifted_reference.shape
    # chan of reference, chan of estimate, sample

    xcf = np.zeros((n_chan, n_chan), dtype=shifted_reference.dtype)
    acf = np.zeros((n_chan, n_chan), dtype=shifted_reference.dtype)

    # TODO: vectorize
    for cest in range(n_chan):
        for cref in range(n_chan):
            xcf[cest, cref] = np.sum(
                shifted_reference[cref, cest, :] * shifted_estimate[cest, :], axis=-1
            )
            acf[cest, cref] = np.sum(
                shifted_reference[cref, cest, :] * shifted_reference[cest, cest, :],
                axis=-1,
            )

    # scale @ acf = xcf
    # acf.T @ scale.T = xcf.T

    # TODO: detect singular matrix
    try:
        scaleT, _, _, _ = np.linalg.lstsq(acf.T, xcf.T, rcond=None)
        scale = scaleT.T
    except np.linalg.LinAlgError:
        # scale @ acf = xcf
        # acf.T @ scale.T = xcf.T
        # acf @ acf.T @ scale.T = acf @ xcf.T
        # (acf @ acf.T + lambd * I) @ scale.T = acf @ xcf.T
        # scale.T = (acf @ acf.T + lambd * I)^-1 @ acf @ xcf.T

        scaleT = np.linalg.pinv(acf @ acf.T + tikhonov_lambda * np.eye(n_chan)) @ (
            acf @ xcf.T
        )
        scale = scaleT.T

    ref_proj = np.zeros_like(shifted_estimate)

    for cest in range(n_chan):
        ref_proj[cest, :] = scale[[cest], :] @ shifted_reference[:, cest, :]

    return ref_proj, scale


def _compute_cost(reference: np.ndarray, estimate: np.ndarray):

    return np.linalg.norm(reference - estimate, ord=2)


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
        ref_proj, est_proj, tikhonov_lambda=tikhonov_lambda
    )
    # print(ref_proj.shape, est_proj.shape)
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
    warn_short_signal: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:

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
    )

    n_chan = reference.shape[-2]

    # compute projection

    if window_length is None:
        window_length = int(_DefaultWindowLengthSeconds * fs)
    if hop_length is None:
        hop_length = int(_DefaultHopLengthSeconds * fs)

    n_sampl = reference.shape[-1]
    n_frames = int(np.ceil((n_sampl - window_length) / hop_length) + 1)

    refs = []
    refprojs = []
    estprojs = []
    costs = np.full((n_frames,), np.nan)
    shifts = np.full((n_chan, n_chan, n_frames), np.nan)
    scales = np.full((n_chan, n_chan, n_frames), np.nan)

    if n_frames == 1 and warn_short_signal:
        warnings.warn(
            "The input signal is too short to be decomposed into frames. "
            "The entire signal is used as a single frame."
        )

    starts = np.arange(0, n_frames) * hop_length
    ends = starts + window_length

    # print(n_frames)

    for i in range(n_frames):
        # print(starts[i], min(n_sampl, ends[i]), n_sampl, min(n_sampl, ends[i]) - starts[i])
        ref, refproj, estproj, cost, shift, scale = _compute_framewise_projection(
            reference=reference[..., starts[i] : ends[i]],
            estimate=estimate[..., starts[i] : ends[i]],
            tikhonov_lambda=tikhonov_lambda,
            max_shift_samples=max_segment_shift_seconds * fs,
        )

        refs.append(ref)
        refprojs.append(refproj)
        estprojs.append(estproj)

        costs[i] = cost
        shifts[..., i] = shift
        scales[..., i] = scale

    return refs, refprojs, estprojs, costs, shifts, scales
