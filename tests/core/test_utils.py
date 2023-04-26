import numpy as np

from spauq.core import utils as _utils


class TestRootMeanSquare:
    def test_rms_zeros(self):
        x = np.zeros((10, 2))
        np.testing.assert_allclose(_utils.root_mean_square(x), 0.0)

    def test_rms_ones(self):
        x = np.ones((10, 2))
        np.testing.assert_allclose(_utils.root_mean_square(x), 1.0)

    def test_rms_values(self):
        np.testing.assert_allclose(_utils.root_mean_square([1, 2, 3]), 2.16024689947)
        np.testing.assert_allclose(_utils.root_mean_square([-1, 1]), 1.0)
        np.testing.assert_allclose(
            _utils.root_mean_square([1, 2, 3, 4, 5]), 3.31662479036
        )
