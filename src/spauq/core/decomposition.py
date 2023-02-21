import typing
import warnings
import numpy as np
import numpy.typing as npt
from typing import Literal, Optional, Tuple
from scipy import signal as sps

from .preprocessing import (
    _ForgiveType,
    _AlignType,
    _ScaleType,
    _validate_inputs,
    _apply_global_forgive,
)

__all__ = ["compute_projection"]

_DefaultWindowLengthSeconds = 2 # follows bsseval
_DefaultHopLengthSeconds = 1.5 # follows bsseval


def _check_interchannel_leakage(
    reference: np.ndarray,
    estimate: np.ndarray,
):
    pass

def _project_shift(
    reference: np.ndarray,
    estimate: np.ndarray,
):
    pass


def _project_scale(
    reference: np.ndarray,
    estimate: np.ndarray,
):
    pass

def _compute_cost(reference: np.ndarray, estimate: np.ndarray):
    return np.linalg.norm(reference - estimate, ord=2, axis=-1)

def _compute_framewise_projection(
    reference: np.ndarray,
    estimate: np.ndarray,
):
    
    ref_proj, est_proj, shift = _project_shift(reference, estimate)
    ref_proj, scale = _project_scale(ref_proj, est_proj)
    cost = _compute_cost(ref_proj, est_proj)

    pass



def compute_projection(
    reference: npt.ArrayLike,
    estimate: npt.ArrayLike,
    fs: int,
    *,
    forgive_mode: Optional[_ForgiveType] = None,
    align_mode: Optional[_AlignType] = None,
    align_use_diag_only: bool = True,
    scale_mode: Optional[_ScaleType] = None,
    window_length: Optional[int] = None,
    hop_length: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    reference, estimate = _validate_inputs(
        reference, estimate, forgive_mode=forgive_mode
    )

    reference, estimate = _apply_global_forgive(
        reference=reference,
        estimate=estimate,
        forgive_mode=forgive_mode,
        align_mode=align_mode,
        align_use_diag_only=align_use_diag_only,
        scale_mode=scale_mode,
    )

    # compute projection

    if window_length is None:
        window_length = int(_DefaultWindowLengthSeconds * fs)
    if hop_length is None:
        hop_length = int(_DefaultHopLengthSeconds * fs)

    n_frames = int(np.ceil(reference.shape[-1] / hop_length)) + 1

    refprojs = []
    estprojs = []

    if n_frames == 1:
        warnings.warn(
            "The input signal is too short to be decomposed into frames. "
            "The entire signal is used as a single frame."
        )
        refproj, estproj = _compute_framewise_projection(reference, estimate)

        refprojs.append(refproj)
        estprojs.append(estproj)

    else:
        starts = np.arange(0, n_frames * hop_length, hop_length)
        ends = starts + window_length
        
        for i in range(n_frames):
            refproj, estproj = _compute_framewise_projection(
                reference=reference[..., starts[i]:ends[i]],
                estimate=estimate[..., starts[i]:ends[i]],
            )

            refprojs.append(refproj)
            estprojs.append(estproj)

    return refprojs, estprojs