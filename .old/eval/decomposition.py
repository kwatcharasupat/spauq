import numpy as np

import scipy as sp
from scipy import fft

from scipy import optimize
from scipy import signal as ss
from scipy import ndimage as ndi


def transform(est, ref, drop_conj_sym=False):
    est = np.squeeze(est)
    ref = np.squeeze(ref)
    assert len(est.shape) == 2
    assert len(ref.shape) == 2

    n_chan_e, n_sampl_e = est.shape
    n_chan, n_sampl = ref.shape

    assert n_chan_e == n_chan
    # assert n_sampl_e == n_sampl

    n_fft = np.power(2.0, np.ceil(np.log2(n_sampl))).astype(int)

    fft_op = fft.rfft if drop_conj_sym else fft.fft

    s_est = fft_op(est, n=n_fft, axis=-1).T  # n_freq, n_chan
    s_ref = fft_op(ref, n=n_fft, axis=-1).T  # n_freq, n_chan

    farr = fft.fftfreq(n=n_fft)
    if drop_conj_sym:
        farr = np.abs(farr[: n_fft // 2 + 1])

    _, n_chan_s = s_est.shape

    assert n_chan_s == n_chan

    return s_est, s_ref, n_sampl, n_fft, farr


def project_spatial_const_phase(est, ref):

    s_est, s_ref, n_sampl, _, _ = transform(est, ref)
    n_freq, n_chan = s_est.shape

    # vec(J.T) = (
    #   sum(s_ref[f].T (x) s_ref[f].T, f)
    # )^(-1)
    # @ vec(
    #   sum(s_est[f].T @ s[f], f)
    # )

    # (n_freq, 1, n_chan) @ (n_freq, n_chan, 1)
    #   --> (n_freq, 1, 1)
    #   --sum--> (1, 1)
    prod_est_ref = np.sum(
        np.conj(s_est[:, None, :]) @ s_ref[:, :, None], axis=0, keepdims=False
    )

    # cheat using broadcasting
    # (n_freq, 1, n_chan) + (n_freq, n_chan, 1)
    #   --> (n_freq, n_chan, n_chan)
    #   --reshape--> (n_freq, n_chan*n_chan)
    #   --sum--> (1, n_chan*n_chan)
    kron_ref_ref = np.sum(
        np.reshape(
            s_ref[:, :, None] + np.conj(s_ref[:, None, :]), (n_freq, n_chan * n_chan)
        ),
        axis=0,
        keepdims=True,
    )

    # kron_ref_ref @ vec(J.T) = prod_est_ref
    vecJt = np.linalg.solve(kron_ref_ref, prod_est_ref)

    J = np.reshape(vecJt, (n_chan, n_chan))

    s_proj = J @ s_ref

    proj = fft.ifft(s_proj, n=n_sampl)

    return proj


def compute_freq_dependent_proj_matrix(s_est, s_ref):
    # ref_cov = s_ref[:, :, None] * np.conj(s_ref)[:, None, :]  # (n_freq, n_chan, n_chan)
    # er_xcov = s_est[:, :, None] * np.conj(s_ref)[:, None, :]  # (n_freq, n_chan, n_chan)

    # fd_proj_matrix = er_xcov * np.inv(ref_cov) # unstable

    # x = np.linalg.solve(a, b) s.t. ax = b
    # X = UV^-1
    # XV = U
    # V^T X^T = U^T

    _, n_chan = s_est.shape

    ref_cov_T = (
        np.conj(s_ref)[:, :, None] * s_ref[:, None, :]
    )  # (n_freq, n_chan, n_chan)
    er_xcov_T = (
        np.conj(s_ref)[:, :, None] * s_est[:, None, :]
    )  # (n_freq, n_chan, n_chan)

    fd_proj_matrix = np.transpose(
        np.linalg.solve(ref_cov_T + 1e-8 * np.eye(n_chan), er_xcov_T),
        (0, 2, 1),
    )

    return fd_proj_matrix


def compute_freq_independent_proj_matrices(fd_proj_matrix, farr):
    mag_proj = np.mean(
        np.abs(fd_proj_matrix), axis=0, keepdims=True
    )  # (n_chan, n_chan)

    fd_phase = np.angle(fd_proj_matrix)

    fi_phase = np.mean(
        fd_phase[1:, :, :] / farr[1:, None, None], axis=0, keepdims=True
    )  # drop DC

    phase_proj = np.exp(1j * farr[:, None, None] * fi_phase)

    fi_proj = mag_proj * phase_proj

    return fi_proj, mag_proj, fi_phase


def compute_numerical_proj_matrix(
    s_est, s_ref, farr, use_phase=True, use_exact_grad=False
):

    _, n_chan = s_est.shape

    A0 = np.eye(n_chan)
    T0 = np.zeros((n_chan, n_chan))

    AT0 = np.concatenate([A0, T0]).flatten("C")

    def optim_func(x, s_ref, s_est, farr, n_chan):

        A = np.reshape(x[: n_chan * n_chan], (n_chan, n_chan))
        T = np.reshape(x[n_chan * n_chan :], (n_chan, n_chan))

        if use_phase:
            phase = np.exp(-2 * np.pi * 1j * farr[:, None, None] * T[None, ...])
        else:
            phase = 1.0

        proj = A[None, ...] * phase

        s_proj = proj @ s_ref[..., None]  # [..., 0]
        # print(farr)
        s_proj = s_proj[..., 0]

        qhase = np.exp(2 * np.pi * 1j * farr[:, None, None] * T[None, ...])
        diff = s_proj - s_est

        DxSH = diff[:, :, None] * np.conj(s_ref)[:, None, :]

        DxSHq = DxSH * qhase

        dA = 2 * np.mean(np.real(DxSHq), axis=0)

        dT = 2 * np.mean(
            np.real(2 * np.pi * 1j * farr[:, None, None] * A * DxSHq), axis=0
        )

        err = np.mean(np.square(np.abs(diff)))
        grad = np.concatenate([dA, dT]).flatten("C")
        return err, grad

    if use_exact_grad:
        # print('exact')
        AT = optimize.minimize(
            optim_func,
            jac=True,
            x0=AT0,
            method="BFGS",
            args=(s_ref, s_est, farr, n_chan),
            options=dict(disp=False, gtol=1e-8),
        )
    else:
        AT = optimize.minimize(
            optim_func, x0=AT0, method="BFGS", args=(s_ref, s_est, farr, n_chan)
        )

    cost = AT.fun

    # if cost > 1e-5:
    #     print("non zero cost:", cost)

    AT = AT.x

    A = np.reshape(AT[: n_chan * n_chan], (n_chan, n_chan))
    T = np.reshape(AT[n_chan * n_chan :], (n_chan, n_chan))

    phase = np.exp(-2 * np.pi * 1j * farr[:, None, None] * T[None, ...])

    mag = A

    proj = mag[None, ...] * phase

    print("\nT", T[1][1])

    return proj, mag, T, cost


def compute_numerical_proj_matrix_iterative(s_est, s_ref, farr, max_iter=10):

    _, n_chan = s_est.shape

    # A0 = np.eye(n_chan)
    T0 = np.zeros((n_chan, n_chan))

    A0 = np.ones((n_chan, n_chan)) / np.sqrt(2)

    A0 = A0.flatten("C")
    T0 = T0.flatten("C")

    costAprev = np.inf
    costTprev = np.inf
    costA = np.inf
    costT = np.inf

    # AT0 = np.concatenate([A0, T0]).flatten("C")

    def optim_func_mag(A, s_ref, s_est, n_chan):

        A = np.reshape(A, (n_chan, n_chan))

        s_proj = A @ s_ref[..., None]  # [..., 0]

        s_proj = s_proj[..., 0]

        err = np.mean(np.square(np.abs(s_proj - s_est)))

        return err

    def optim_func_phase(T, s_ref, s_est, farr, n_chan):
        T = np.reshape(T, (n_chan, n_chan))

        phase = np.exp(-2 * np.pi * 1j * farr[:, None, None] * T[None, ...])

        proj = phase

        s_proj = proj @ s_ref[..., None]  # [..., 0]

        s_proj = s_proj[..., 0]

        err = np.mean(np.square(np.abs(s_proj - s_est)))

        return err

    for _ in range(max_iter):

        Aopt = optimize.minimize(
            optim_func_mag, x0=A0, method="BFGS", args=(s_ref, s_est, n_chan)
        )
        costAprev = costA
        costA = Aopt.fun
        A0 = Aopt.x

        Topt = optimize.minimize(
            optim_func_phase, x0=T0, method="BFGS", args=(s_ref, s_est, farr, n_chan)
        )
        costTprev = costT
        costT = Topt.fun
        T0 = Topt.x

        if np.abs(costAprev - costA) < 1e-3 and np.abs(costTprev - costT) < 1e-3:
            break

    T0 = np.reshape(T0, (n_chan, n_chan))
    A0 = np.reshape(A0, (n_chan, n_chan))

    print("A", A0, "\nT", T0)

    phase = np.exp(-2 * np.pi * 1j * farr[:, None, None] * T0[None, ...])

    mag = A0

    proj = mag[None, ...] * phase

    s_proj = proj @ s_ref[..., None]  # [..., 0]

    s_proj = s_proj[..., 0]

    return proj, mag, T0, np.mean(np.square(np.abs(s_proj - s_est)))


def compute_corr_proj_matrix(s_est, s_ref, use_aligned_refref=True, lambd=1e-3):
    n_chan, n_sampl = s_est.shape
    # print("n_chan", n_chan, "n_sampl", n_sampl)

    lags = ss.correlation_lags(n_sampl, n_sampl, mode="full")

    optim_lags = np.zeros((n_chan, n_chan))

    # print("aligning signals")
    for c in range(n_chan):
        for d in range(n_chan):
            # print("aligning channels:", c, d)
            optim_lags[c, d] = lags[
                np.argmax(ss.correlate(s_est[c, :], s_ref[d, :], method="fft"))
            ]

    # print("optim lags", optim_lags)

    optim_lags = np.round(optim_lags).astype(int)
    corr_refref = np.zeros((n_chan, n_chan))
    corr_estref = np.zeros((n_chan, n_chan))

    s_ref_aligned = np.zeros((n_chan, n_chan, n_sampl))

    # print("computing correlations")
    for c in range(n_chan):
        for d in range(n_chan):
            s_est_aligned = s_est[c, :]
            s_ref_aligned[c, d, :] = np.roll(s_ref[d, :], optim_lags[c, d])
            if optim_lags[c, d] > 0:
                s_ref_aligned[c, d, : optim_lags[c, d]] = 0
                s_est_aligned[: optim_lags[c, d]] = 0
            elif optim_lags[c, d] < 0:
                s_ref_aligned[c, d, optim_lags[c, d] :] = 0
                s_est_aligned[optim_lags[c, d] :] = 0

            if use_aligned_refref:
                corr_refref[c, d] = np.mean(
                    s_ref_aligned[c, d, :] * s_ref_aligned[c, d, :]
                )
            else:
                corr_refref[c, d] = np.mean(s_ref[d, :] * s_ref[d, :])
                
            corr_estref[c, d] = np.mean(s_est_aligned * s_ref_aligned[c, d, :])

    # A * R = Rhat
    # R^T * A^T = Rhat^T
    # R * R^T * A^T = R * Rhat^T
    # A^T = (R * R^T + lambda * I)^-1 * R * Rhat^T

    try:
        A = np.linalg.lstsq(corr_refref.T, corr_estref.T, rcond=-1)[0].T
    except np.linalg.LinAlgError:
        A = np.linalg.pinv(
            corr_refref @ corr_refref.T + lambd * np.eye(n_chan)
        ) @ corr_refref @ corr_estref.T


    s_proj = np.zeros((n_chan, n_sampl))

    for c in range(n_chan):
        s_proj[c] = np.squeeze(A[[c], :] @ s_ref_aligned[c, :, :])

    # print("sproj", s_proj.shape)

    return s_proj


def adjust_est_global_mag(s_est, s_ref):

    # n_chan, n_sampl = s_est.shape

    est_rms = np.linalg.norm(s_est)
    ref_rms = np.linalg.norm(s_ref)
    est_adjusted = s_est * ref_rms / est_rms

    ref_adjusted = s_ref

    return est_adjusted, ref_adjusted

def adjust_est_global_phase(s_est, s_ref):

    n_chan, n_sampl = s_est.shape

    corrs = np.zeros((n_chan, 2 * n_sampl - 1))
    corr_lags = ss.correlation_lags(n_sampl, n_sampl, mode="full")

    for c in range(n_chan):
        corrs[c] = ss.correlate(s_est[c, :], s_ref[c, :], mode="full", method="fft")

    corrs = np.sum(corrs, axis=0)
    max_corr_idx = np.argmax(corrs)
    max_corr_lag = corr_lags[max_corr_idx]

    if max_corr_lag > 0:
        s_est = s_est[:, max_corr_lag:]
        s_ref = s_ref[:, :-max_corr_lag]
    elif max_corr_lag < 0:
        s_est = s_est[:, :max_corr_lag]
        s_ref = s_ref[:, -max_corr_lag:]
    else:
        pass

    return s_est, s_ref


def project_spatial_magphase(
    est,
    ref,
    use_bsseval_proj=False,
    drop_conj_sym=True,
    allow_global_scale=True,
    use_numerical=True,
    use_phase=True,
    iterative=True,
    use_exact_grad=False,
    use_time_domain_corr=True,
    use_aligned_refref=False,
    forgive_global_mag=True,
    forgive_global_shift=True,
):
    if forgive_global_shift:
        est, ref = adjust_est_global_phase(est, ref)
    if forgive_global_mag:
        est, ref = adjust_est_global_mag(est, ref)

    if use_time_domain_corr:
        # print("Using time domain correlation")
        ref_trans = compute_corr_proj_matrix(
            est, ref, use_aligned_refref=use_aligned_refref
        )
        cost = np.mean(np.square(est - ref_trans))
        return ref, est, ref_trans, cost

    if use_bsseval_proj:
        raise NotImplementedError
    else:
        s_est, s_ref, n_sampl, n_fft, farr = transform(
            est, ref, drop_conj_sym=drop_conj_sym
        )

        if use_numerical:

            if not drop_conj_sym:
                n_freq = s_ref.shape[0]
                s_ref = s_ref[: n_freq // 2 + 1, :]
                s_est = s_est[: n_freq // 2 + 1, :]
                farr = farr[: n_freq // 2 + 1]

            if iterative:
                assert use_phase
                (
                    fi_proj_matrix,
                    mag_proj,
                    phase_proj,
                    cost,
                ) = compute_numerical_proj_matrix_iterative(s_est, s_ref, farr)
            else:
                (
                    fi_proj_matrix,
                    mag_proj,
                    phase_proj,
                    cost,
                ) = compute_numerical_proj_matrix(
                    s_est,
                    s_ref,
                    farr,
                    use_phase=use_phase,
                    use_exact_grad=use_exact_grad,
                )
            # print("MAG:", mag_proj)
            # print("DELAY:", phase_proj)
        else:
            fd_proj_matrix = compute_freq_dependent_proj_matrix(s_est, s_ref)
            fi_proj_matrix, mag_proj, _ = compute_freq_independent_proj_matrices(
                fd_proj_matrix, farr
            )

            cost = None
            # print("MAG:", mag_proj)

        s_ref_trans = (fi_proj_matrix @ s_ref[:, :, None])[..., 0]

        if drop_conj_sym:
            ifft_op = fft.irfft
        else:
            ifft_op = fft.ifft
            raise NotImplementedError  # TODO: fix reconstruction

        ref_trans = np.real(ifft_op(s_ref_trans, n=n_fft, axis=0))[:n_sampl, ...]

        ref_trans = ref_trans.T[None, ...]

        return ref_trans, cost


def root_mean_square(x):
    return np.sqrt(np.mean(np.square(x)))


def distortion_ratio(
    est,
    ref,
    use_numerical=True,
    use_phase=True,
    iterative=False,
    use_exact_grad=False,
    forgive_global_mag=True,
    forgive_global_shift=True,
):

    ref, est, ref_trans, cost = project_spatial_magphase(
        est,
        ref,
        use_time_domain_corr=True,
        use_numerical=use_numerical,
        use_phase=use_phase,
        iterative=iterative,
        use_exact_grad=use_exact_grad,
        forgive_global_mag=forgive_global_mag,
        forgive_global_shift=forgive_global_shift,
    )

    # n_sampl = min(ref.shape[-1], est.shape[-1])

    # est = est[..., :n_sampl]
    # ref = ref[..., :n_sampl]
    # ref_trans = ref_trans[..., :n_sampl]

    e_spat = ref_trans - ref
    e_filt = est - ref_trans

    ref_pow = np.mean(np.square(ref))

    ref_trans_pow = np.mean(np.square(ref_trans))

    spat_pow = np.mean(np.square(e_spat))
    filt_pow = np.mean(np.square(e_filt))

    spat_ratio = ref_pow / spat_pow
    filt_ratio = ref_trans_pow / filt_pow

    return 10 * np.log10(spat_ratio), 10 * np.log10(filt_ratio), cost
