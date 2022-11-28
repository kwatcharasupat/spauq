import numpy as np

import scipy as sp
from scipy import fft

from scipy import optimize


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

    s_est, s_ref, n_sampl = _transform(est, ref)
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

        dA = 2 * np.real(np.mean(DxSHq, axis=0))
        

        dT = 2 * np.real(
            np.mean(2 * np.pi * 1j * farr[:, None, None] * A * DxSHq, axis=0)
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

    # print("A", A, "\nT", T)

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
):

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
    est, ref, use_numerical=True, use_phase=True, iterative=False, use_exact_grad=False
):

    # ref = root_mean_square(est)/root_mean_square(ref) * ref
    # print(ref)

    ref_trans, cost = project_spatial_magphase(
        est,
        ref,
        use_numerical=use_numerical,
        use_phase=use_phase,
        iterative=iterative,
        use_exact_grad=use_exact_grad,
    )

    n_sampl = min(ref.shape[-1], est.shape[-1])

    est = est[..., :n_sampl]
    ref = ref[..., :n_sampl]
    ref_trans = ref_trans[..., :n_sampl]

    e_spat = ref_trans - ref
    e_filt = est - ref_trans  # est - (ref + e_spat)

    ref_pow = np.mean(np.square(ref))
    est_pow = np.mean(np.square(est))

    ref_trans_pow = np.mean(np.square(ref_trans))

    spat_pow = np.mean(np.square(e_spat))
    filt_pow = np.mean(np.square(e_filt))

    # print("SIGNAL POWERS")
    # print("REF", ref_pow)
    # print("EST", est_pow)
    # print("PROJ", ref_trans_pow)
    # print(filt_pow, ref_trans_pow, spat_pow)

    spat_ratio = ref_pow / spat_pow
    filt_ratio = ref_trans_pow / filt_pow

    return 10 * np.log10(spat_ratio), 10 * np.log10(filt_ratio), cost
