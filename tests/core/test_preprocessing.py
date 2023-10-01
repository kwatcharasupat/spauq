import numpy as np
import pytest

from spauq.core import preprocessing as _pp


class TestValidateInput:
    def test_lt_1dim(self):
        x = np.ones((10,))

        with pytest.raises(AssertionError):
            _pp._validate_input(x)

    def test_lt_2chan(self):
        x = np.ones((1, 10))

        with pytest.raises(AssertionError):
            _pp._validate_input(x)


class TestValidateInputs:
    def test_unequal_dim(self):

        ref = np.ones((2, 10))
        est = np.ones((2, 10, 1))

        with pytest.raises(AssertionError):
            _pp._validate_inputs(ref, est)

    def test_unequal_chan(self):

        ref = np.ones((2, 10))
        est = np.ones((3, 10))

        with pytest.raises(AssertionError):
            _pp._validate_inputs(ref, est)

    def test_unequal_batch(self):

        ref = np.ones((10, 2, 10))
        est = np.ones((11, 2, 10))

        with pytest.raises(AssertionError):
            _pp._validate_inputs(ref, est)

    # def test_unequal_time(self):
    #
    #     ref = np.ones((2, 10))
    #     est = np.ones((2, 11))
    #
    #     with pytest.raises(NotImplementedError):
    #         _pp._validate_inputs(ref, est)

    def test_np_return(self):

        ref = np.ones((2, 10)).tolist()
        est = np.ones((2, 10))

        ref, est = _pp._validate_inputs(ref, est)

        assert isinstance(ref, np.ndarray)
        assert isinstance(est, np.ndarray)

        ref = np.ones((2, 10))
        est = np.ones((2, 10)).tolist()

        ref, est = _pp._validate_inputs(ref, est)

        assert isinstance(ref, np.ndarray)
        assert isinstance(est, np.ndarray)

        ref = np.ones((2, 10)).tolist()
        est = np.ones((2, 10)).tolist()

        ref, est = _pp._validate_inputs(ref, est)

        assert isinstance(ref, np.ndarray)
        assert isinstance(est, np.ndarray)


class TestComputeOptimalShift:
    def test_gt_2dim(self):

        ref = np.ones((2, 3, 10))
        est = np.ones((2, 3, 10))

        with pytest.raises(NotImplementedError):
            _pp._compute_optimal_shift(ref, est)

    def test_infinite_max_shift(self):

        ref = np.stack(
            [
                np.sin(2 * np.pi * 40 * np.linspace(0, 1, 44100))
                + np.linspace(-1, 1, 44100),
                np.sin(2 * np.pi * 41 * np.linspace(0, 1, 44100))
                + np.linspace(-1, 1, 44100),
            ],
            axis=0,
        )
        est = np.roll(ref, 1000, axis=1)

        best_lag = _pp._compute_optimal_shift(ref, est, max_shift_samples=np.inf)

        assert best_lag == -1000

    def test_finite_max_shift(self):
        ref = np.stack(
            [
                np.sin(2 * np.pi * 400 * np.linspace(0, 1, 44100))
                + np.linspace(0, 1, 44100),
                np.sin(2 * np.pi * 450 * np.linspace(0, 1, 44100))
                + np.linspace(0, 1, 44100),
            ],
            axis=0,
        )
        est = np.roll(ref, 1000, axis=1)

        best_lag = _pp._compute_optimal_shift(ref, est, max_shift_samples=500)

        assert abs(best_lag) < 500


class TestApplyShift:
    def test_batch_shape(self):
        ref = np.ones((2, 10))
        est = np.ones((2, 10))

        best_lag = np.array([0])

        with pytest.raises(AssertionError):
            _pp._apply_shift(ref, est, best_lag=best_lag)

    def test_best_lag_shape(self):
        ref = np.ones((10, 2, 10))
        est = np.ones((10, 2, 10))

        best_lag = np.zeros((10,))

        with pytest.raises(NotImplementedError):
            _pp._apply_shift(ref, est, best_lag=best_lag)

    def test_zero_best_lag(self):
        ref = np.ones((2, 10))
        est = np.ones((2, 10))

        best_lag = 0

        ref_shifted, est_shifted = _pp._apply_shift(ref, est, best_lag=best_lag)

        np.testing.assert_equal(ref, ref_shifted)
        np.testing.assert_equal(est, est_shifted)

    def test_unsupported_align_mode(self):
        ref = np.ones((2, 10))
        est = np.ones((2, 10))

        best_lag = 1

        with pytest.raises(NotImplementedError):
            _pp._apply_shift(ref, est, best_lag=best_lag, align_mode="unsupported")


    def test_positive_shift(self):
        ref = np.ones((2, 10)) + 0.1 * np.arange(-2, 8)[None, :]
        est = np.stack(
            [np.arange(0, 10), np.arange(2, 12)], axis=0
        )

        best_lag = 2

        ref_shifted, est_shifted = _pp._apply_shift(ref, est, best_lag=best_lag)

        assert ref_shifted.shape == (2, 8)
        assert est_shifted.shape == (2, 8)

        np.testing.assert_equal(ref_shifted, ref[:, 2:])
        np.testing.assert_equal(est_shifted, est[:, :-2])

    def test_negative_shift(self):
        ref = np.ones((2, 10)) + 0.1 * np.arange(2, 12)[None, :]
        est = np.stack(
            [np.arange(0, 10), np.arange(2, 12)], axis=0
        )

        best_lag = -2

        ref_shifted, est_shifted = _pp._apply_shift(ref, est, best_lag=best_lag)

        assert ref_shifted.shape == (2, 8)
        assert est_shifted.shape == (2, 8)

        np.testing.assert_equal(ref_shifted, ref[:, :-2])
        np.testing.assert_equal(est_shifted, est[:, 2:])

class TestApplyGlobalShiftForgive:
    def test_unsupported_align_mode(self):
        ref = np.ones((2, 10))
        est = np.ones((2, 10))

        with pytest.raises(AssertionError):
            _pp._apply_global_shift_forgive(ref, est, align_mode="unsupported")