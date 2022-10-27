import numpy as np

import scipy as sp
from scipy import fft


def project_spatial(est, ref):

    assert len(est.shape) == 2
    assert len(ref.shape) == 2

    n_chan_e, n_sampl_e = est.shape
    n_chan, n_sampl = ref.shape

    assert n_chan_e == n_chan
    assert n_sampl_e == n_sampl

    n_fft = np.power(2.0, np.ceil(np.log2(n_sampl)))

    s_est = fft.fft(est, n=n_fft, axis=-1).T  # n_freq, n_chan
    s_ref = fft.fft(ref, n=n_fft, axis=-1).T  # n_freq, n_chan

    n_freq, n_chan_s = s_est.shape

    assert n_chan_s == n_chan

    # vec(J.T) = (
    #   sum(s_ref[f].T (x) s_ref[f].T, f)
    # )^(-1)
    # @ vec(
    #   sum(s_est[f].T @ s[f], f)
    # )

    # (n_freq, 1, n_chan) @ (n_freq, n_chan, 1)
    #   --> (n_freq, 1, 1)
    #   --sum--> (1, 1)
    prod_est_ref = np.sum(s_est[:, None, :] @ s_ref[:, :, None], axis=0, keepdims=False)

    # cheat using broadcasting
    # (n_freq, 1, n_chan) + (n_freq, n_chan, 1)
    #   --> (n_freq, n_chan, n_chan)
    #   --reshape--> (n_freq, n_chan*n_chan)
    #   --sum--> (1, n_chan*n_chan)
    kron_ref_ref = np.sum(
        np.reshape(s_ref[:, None, :] + s_ref[:, :, None], (n_freq, n_chan * n_chan)),
        axis=0,
        keepdims=True,
    )

    # kron_ref_ref @ vec(J.T) = prod_est_ref
    vecJt = np.linalg.solve(kron_ref_ref, prod_est_ref)

    J = np.reshape(vecJt, (n_chan, n_chan))

    s_proj = J @ s_ref

    proj = fft.ifft(s_proj, n=n_sampl)
    
    return proj
