from abc import ABC, abstractmethod

import numpy as np


class Spatializer(ABC):
    def generate_one(self, signal, pan):
        raise NotImplementedError


class StereoPanLaw(Spatializer):
    def __init__(self, mode="ConstantPower") -> None:
        super().__init__()

        assert mode in ["ConstantPower"]

        self.mode = mode

        self.filter = None

    def get_stereo_multipliers(self, pan):

        assert -1 <= pan <= 1

        if self.mode == "ConstantPower":
            pan = 45 * (pan + 1)  # [-1, 1] -> [0, 90]
            L = np.cos(np.deg2rad(pan))
            R = np.sin(np.deg2rad(pan))
        else:
            raise NotImplementedError

        return np.array([L, R])

    def get_stereo_rotater(self, pan):
        multipliers = self.get_stereo_multipliers(pan)

        rotater = np.array(
            [[multipliers[0], -multipliers[1]], [multipliers[1], multipliers[0]]]
        )

        return rotater

    def generate_one(self, signal, pan):
        raise NotImplementedError


class MonoToStereoPanLaw(StereoPanLaw):
    def generate_one(self, signal, pan):

        multipliers = self.get_stereo_multipliers(pan)

        signal = multipliers[:, None] * signal

        return signal
