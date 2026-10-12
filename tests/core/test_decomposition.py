import numpy as np
import pytest

from spauq.core.metrics import spauq_eval

_FS = 8000


@pytest.fixture
def reference():
    rng = np.random.default_rng(0)
    return rng.standard_normal((2, 3 * _FS))


def _eval(reference, estimate, **kwargs):
    return spauq_eval(
        reference,
        estimate,
        _FS,
        return_shift=True,
        return_scale=True,
        verbose=False,
        **kwargs,
    )


class TestDelayedProjection:
    def test_pure_delay(self, reference):
        delay = 10
        estimate = np.roll(reference, delay, axis=-1)

        out = _eval(reference, estimate)

        # the delayed reference explains the estimate fully: no residual
        assert out["SRR"] > 70
        np.testing.assert_allclose(np.diag(out["shift"]), -delay)
        np.testing.assert_allclose(out["scale"], np.eye(2), atol=1e-6)

        # a delay is a spatial distortion: white noise vs. its delayed copy
        assert out["SSR"] == pytest.approx(10 * np.log10(0.5), abs=0.1)

    def test_pure_delay_forgiven(self, reference):
        estimate = np.roll(reference, 10, axis=-1)

        out = _eval(reference, estimate, forgive_mode="shift")

        assert out["SSR"] > 70
        assert out["SRR"] > 70

    def test_interchannel_delay_and_gain(self, reference):
        delays = (5, 12)
        gains = (0.8, 0.5)
        estimate = np.stack(
            [g * np.roll(r, d) for g, r, d in zip(gains, reference, delays)]
        )

        out = _eval(reference, estimate)

        assert out["SRR"] > 70
        np.testing.assert_allclose(np.diag(out["shift"]), [-d for d in delays])
        np.testing.assert_allclose(out["scale"], np.diag(gains), atol=1e-6)

    def test_crosschannel_delay_and_gain(self, reference):
        estimate = np.stack(
            [
                0.8 * reference[0] + 0.4 * np.roll(reference[1], 7),
                0.6 * np.roll(reference[1], 3),
            ]
        )

        out = _eval(reference, estimate)

        assert out["SRR"] > 70
        # shift is (chan of reference, chan of estimate)
        assert out["shift"][1, 0] == -7
        assert out["shift"][1, 1] == -3
        # scale is (chan of estimate, chan of reference)
        np.testing.assert_allclose(
            out["scale"], [[0.8, 0.4], [0.0, 0.6]], atol=1e-6
        )


class TestUndelayedProjection:
    def test_identity(self, reference):
        out = _eval(reference, reference.copy())

        assert out["SSR"] > 70
        assert out["SRR"] > 70

    def test_diagonal_gain(self, reference):
        # same result as before delays were applied to the projection
        gains = np.array([0.5, 2.0])
        estimate = gains[:, None] * reference

        out = _eval(reference, estimate)

        expected_ssr = 10 * np.log10(
            np.sum(reference**2) / np.sum(((gains[:, None] - 1) * reference) ** 2)
        )

        assert out["SRR"] > 70
        assert out["SSR"] == pytest.approx(expected_ssr, abs=0.1)
        np.testing.assert_allclose(out["scale"], np.diag(gains), atol=1e-6)

    def test_instantaneous_mix(self, reference):
        mix = np.array([[0.7, 0.3], [0.2, 0.9]])
        estimate = mix @ reference

        out = _eval(reference, estimate)

        expected_ssr = 10 * np.log10(
            np.sum(reference**2) / np.sum((estimate - reference) ** 2)
        )

        assert out["SRR"] > 70
        assert out["SSR"] == pytest.approx(expected_ssr, abs=0.1)
        np.testing.assert_allclose(out["scale"], mix, atol=1e-6)
