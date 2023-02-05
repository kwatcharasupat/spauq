import sox
import pydub as pyd
from torchaudio import functional as taF
import torch
import numpy as np

class Transform:
    def __init__(self, transform_specs, sample_rate):
        self.transforms = []
        self.sample_rate = sample_rate

        for spec in transform_specs:
            print(spec)
            backend = spec["backend"]
            name = spec["name"]
            kwargs = spec["kwargs"]
            if backend == "sox":
                self.transforms.append(SoxTransform(sample_rate, name, kwargs))
            elif backend in ["custom"]:
                if name == "delay":
                    self.transforms.append(Delay(sample_rate, name, kwargs))
                else:
                    raise NotImplementedError
            elif backend in ["ta", "torchaudio"]:
                raise NotImplementedError
            else:
                raise NotImplementedError

    def __call__(self, signal):
        for transform in self.transforms:
            signal = transform(signal)

        return signal


class SoxTransform:
    def __init__(self, sample_rate, name, kwargs) -> None:
        self.transform = sox.Transformer()
        self.sample_rate = sample_rate
        self.transform.__getattribute__(name)(**kwargs)

    def __call__(self, signal):
        return self.transform.build_array(
            input_array=signal.T, sample_rate_in=self.sample_rate
        ).T


class TorchAudioTransform:
    def __init__(self, sample_rate, name, kwargs) -> None:
        self.sample_rate = sample_rate
        self.func = taF.__getattribute__(name)
        self.kwargs = kwargs
    
    def __call__(self, signal):
        
        return self.func(torch.tensor(signal), sample_rate=self.sample_rate, **self.kwargs).numpy()


class Delay():
    def __init__(self, sample_rate, name, kwargs) -> None:
        self.delays = kwargs["positions"]
        self.sample_rate = sample_rate

    def __call__(self, signal):
        out = np.zeros_like(signal)

        for i, delay in enumerate(self.delays):
            delta = int(delay * self.sample_rate)
            out[i] = np.roll(signal[i], delta)
            out[i, :delta] = 0.0

        return out