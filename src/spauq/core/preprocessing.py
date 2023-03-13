import typing
import warnings
import numpy as np
import numpy.typing as npt
from typing import Literal, Optional, Tuple
from scipy import signal as sps

from .utils import root_mean_square

_ForgiveType = Literal["none", "scale", "shift", "both"]
_ForgiveDefault = "none"

# _AlignType = Literal["zero_pad", "overlap", "trim_estimate", "trim_reference"]
_AlignType = Literal["overlap"]
_AlignDefault = "overlap"

_ScaleType = Literal["least_square", "equal_rms"]
_ScaleDefault = "equal_rms"


def _validate_input(signal: np.ndarray) -> None:
    assert signal.ndim >= 2, "Signal must be at least 2-dimensional"

    n_chan, _ = signal.shape[-2:]

    assert n_chan > 1, "Number of channels must be at least 2"


def _validate_inputs(
    reference: npt.ArrayLike,
    estimate: npt.ArrayLike,
    *,
    forgive_mode: Optional[_ForgiveType] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate input signal.

    Checks that
        - the input signal is at least 2-dimensional
        - the last axis is the time axis
        - the second last axis is the channel axis
    """
    reference = np.asarray(reference)
    estimate = np.asarray(estimate)

    _validate_input(reference)
    _validate_input(estimate)

    shape_r = reference.shape
    shape_e = estimate.shape

    # print(shape_r, shape_e)

    n_chan_r, n_sampl_r = shape_r[-2:]
    n_chan_e, n_sampl_e = shape_e[-2:]

    ndim_r = reference.ndim
    ndim_e = estimate.ndim

    assert ndim_r == ndim_e, "Number of dimensions must be equal"
    assert n_chan_r == n_chan_e, "Number of channels must be equal"
    assert shape_r[:-2] == shape_e[:-2], "Shape of non-time axes must be equal"

    if n_sampl_r != n_sampl_e:
        raise NotImplementedError(
            "Shifting for unequal sample lengths is not implemented yet."
        )

        if forgive_mode is None:
            forgive_mode = _ForgiveDefault

        if forgive_mode in ["shift", "both"]:
            warnings.warn(
                "Numbers of samples are not equal. Shifting estimate to match reference.",
                UserWarning,
            )
        else:
            raise ValueError(
                "Number of samples is not equal. Set `forgive_mode` to `shift` or `both` to allow shifting."
            )

    return reference, estimate


def _compute_optimal_shift(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    use_diag_only: bool = True,
    max_shift_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    n_chan, n_sampl_r = reference.shape[-2:]
    _, n_sampl_e = estimate.shape[-2:]

    xclags = sps.correlation_lags(n_sampl_r, n_sampl_e, mode="full")
    n_lags = (
        int(max_shift_samples) * 2 + 1
        if np.isfinite(max_shift_samples)
        else len(xclags)
    )

    if reference.ndim == 2:
        xcvals = np.zeros((n_chan, n_chan, n_lags))
        for i in range(n_chan):
            for j in range(n_chan):
                if use_diag_only and i != j:
                    continue
                xcf = sps.correlate(
                    reference[i], estimate[j], mode="full", method="fft"
                )

                if np.isfinite(max_shift_samples):
                    xcf = xcf[np.abs(xclags) <= max_shift_samples]
                xcvals[i, j] = xcf
        xcvals = np.abs(
            np.mean(
                xcvals,
                axis=(
                    -3,
                    -2,
                ),
            )
        )
        # use mean for stability
        xclags = xclags[np.abs(xclags) <= max_shift_samples]
        best_lag = xclags[np.argmax(xcvals, axis=-1)]
    else:
        raise NotImplementedError

    return best_lag


def _apply_shift(
    reference: np.ndarray,
    estimate: np.ndarray,
    best_lag: np.ndarray,
    *,
    align_mode: _AlignType,
) -> Tuple[np.ndarray, np.ndarray]:

    shape_r = reference.shape
    shape_l = best_lag.shape

    assert shape_r[:-2] == shape_l, "Shape of non-time axes must be equal"

    n_chan, n_sampl_r = shape_r[-2:]
    _, n_sampl_e = estimate.shape[-2:]

    if len(shape_l) > 0:
        raise NotImplementedError("Batched shifting is not implemented yet.")

    if best_lag == 0:
        return reference, estimate

    estimate_padded = np.concatenate(
        [estimate, np.zeros((n_chan, abs(best_lag)))], axis=-1
    )  # (n_chan, n_sampl_e + abs(best_lag
    estimate_rolled = np.roll(
        estimate_padded, best_lag, axis=-1
    )  # (n_chan, n_sampl_e + abs(best_lag))

    if align_mode == "overlap":
        if best_lag < 0:
            # if estimate is behind, lag is negative
            n_overlap = n_sampl_e + best_lag

            reference = reference[..., :n_overlap]
            estimate = estimate_rolled[..., :n_overlap]
        else:
            # if estimate is ahead, lag is positive
            n_overlap = n_sampl_r - best_lag

            reference = reference[..., best_lag:]
            estimate = estimate_rolled[..., best_lag:]
    else:
        raise NotImplementedError("Other alignment modes are not implemented yet.")

    return reference, estimate


def _apply_global_shift_forgive(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    align_mode: Optional[_AlignType] = None,
    use_diag_only: bool = True,
    max_shift_samples: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:

    if align_mode is None:
        if verbose:
            warnings.warn(
                f"No align_mode specified, defaulting to `{_AlignDefault}`", UserWarning
            )
        align_mode = _AlignDefault

    assert align_mode in typing.get_args(_AlignType), "Invalid align_mode"

    best_lag = _compute_optimal_shift(
        reference,
        estimate,
        use_diag_only=use_diag_only,
        max_shift_samples=max_shift_samples,
    )

    reference, estimate = _apply_shift(
        reference, estimate, best_lag, align_mode=align_mode
    )

    print("best_lag", best_lag)

    return reference, estimate


def _apply_global_scale_forgive(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    scale_mode: Optional[_ScaleType] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    if scale_mode is None:
        if verbose:
            warnings.warn(
                f"No align_mode specified, defaulting to `{_ScaleDefault}`", UserWarning
            )
        scale_mode = _ScaleDefault

    assert scale_mode in typing.get_args(_ScaleType), "Invalid scale_mode"

    if scale_mode == "least_square":
        # for compatibility with SI-SDR
        ref_pow = np.sum(np.square(reference), axis=(-2, -1), keepdims=True)
        cross_pow = np.sum(
            np.square(reference * estimate), axis=(-2, -1), keepdims=True
        )
        scale = cross_pow / ref_pow
    elif scale_mode == "equal_rms":
        ref_rms = root_mean_square(reference, axis=(-2, -1), keepdims=True)
        est_rms = root_mean_square(estimate, axis=(-2, -1), keepdims=True)
        scale = est_rms / ref_rms
    else:
        # this should never be reached
        raise ValueError("Invalid scale_mode")

    reference = reference * scale

    print("scale", scale)

    return reference, estimate


def _apply_global_forgive(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    forgive_mode: Optional[_ForgiveType] = None,
    align_mode: Optional[_AlignType] = None,
    align_use_diag_only: bool = True,
    max_shift_samples: Optional[int] = None,
    scale_mode: Optional[_ScaleType] = None,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:

    if forgive_mode is None:
        if verbose:
            warnings.warn(
                f"No forgive_mode specified, defaulting to `{_ForgiveDefault}`",
                UserWarning,
            )
        forgive_mode = _ForgiveDefault

    assert forgive_mode in typing.get_args(_ForgiveType), "Invalid forgive_mode"

    if forgive_mode == "none":
        return reference, estimate

    # apply shifting first so that the scaling is only computed on the relevant part
    if forgive_mode in ["shift", "both"]:
        reference, estimate = _apply_global_shift_forgive(
            reference,
            estimate,
            align_mode=align_mode,
            use_diag_only=align_use_diag_only,
            max_shift_samples=max_shift_samples,
            verbose=verbose,
        )

    if forgive_mode in ["scale", "both"]:
        reference, estimate = _apply_global_scale_forgive(
            reference, estimate, scale_mode=scale_mode, verbose=verbose
        )

    return reference, estimate
