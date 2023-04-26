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

    def test_unequal_time(self):

        ref = np.ones((2, 10))
        est = np.ones((2, 11))

        with pytest.raises(NotImplementedError):
            _pp._validate_inputs(ref, est)

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

        