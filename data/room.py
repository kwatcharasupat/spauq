from abc import ABC
import functools
import numpy as np

import pyroomacoustics as pra
from scipy.signal import *


def stftfilt(signal, Wn, btype, fs, **kwargs):
    
    winmult = np.power(2.0, np.ceil(fs/16000)).astype(int)
    nperseg=4096*winmult
    noverlap=nperseg//2
    
    n_sampl = signal.shape[0]
    farr, tarr, X = stft(signal, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)
    
    
    
    if btype == 'lowpass':
        mask = farr < Wn[0]
    elif btype == 'highpass':
        mask = farr > Wn[0]
    elif btype == 'bandstop':
        mask = (farr < Wn[0]) | (farr > Wn[1])
        # print(farr)
        # print(mask)
    elif btype == 'bandpass':
        mask = (farr > Wn[0]) & (farr < Wn[1])
        # print(mask)[]
        
    Xfilt = X * mask[:, None]
    
    _, sigfilt = istft(Xfilt, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap)
    
    # print(signal.shape, X.shape, Xfilt.shape, sigfilt.shape)
    sigfilt = sigfilt[:n_sampl]
    
    
    return sigfilt

def delay(signal, ldelay, rdelay, **kwargs):
    
    n_sampl = signal.shape[-1]
    L = signal[0, :]
    R = signal[1, :]
    
    L = np.concatenate([np.zeros((ldelay,)), L])[:n_sampl]
    R = stftfilt(np.concatenate([np.zeros((rdelay,)), R])[:n_sampl], [7000], 'lowpass', 16000)
    
    signal = np.stack([L, R], axis=0)
    

    return signal

class Spatializer(ABC):
    def generate_one(self, signal, angle):
        raise NotImplementedError

class StereoPanLaw(Spatializer):
    def __init__(self, mode='ConstantPower') -> None:
        super().__init__()
        
        assert mode in ['ConstantGain', 'ConstantPower']
        
        self.mode = mode
        
        self.filter = None
        
    def get_stereo_multipliers(self, angle):
        
        if self.mode == 'ConstantPower':
            # angle = 0 is center i.e. p = 0
            # angle = -45 is -0.5
            # angle = -90 is -1 
            p = angle/90
            pan = 45 * (p + 1)
            L = np.cos(np.deg2rad(pan))
            R = np.sin(np.deg2rad(pan))
        elif self.mode == 'ConstantGain':
            pan = 0.5 * ((angle/90 + 1) % 2.0) + 0.5
            L = pan
            R = 1.0 - pan
        else:
            raise NotImplementedError
        
        return np.array([L, R])
    
        
    def generate_one(self, signal, angle, filter_kwargs=None):
        
        multipliers = self.get_stereo_multipliers(angle)
        
        if filter_kwargs is not None:
            if type(filter_kwargs) is dict:
                if filter_kwargs['ftype'] == 'ideal':
                    signal = stftfilt(signal, **filter_kwargs)
            else:
                signal = sosfilt(filter_kwargs, signal)
            
            # print(signal.shape)
            
            if np.any(np.isnan(signal)):
                raise ValueError
        # print(angle, multipliers)
        
        signal = multipliers[:, None] * signal[None, :]
        
        if filter_kwargs is not None:
            if type(filter_kwargs) is dict:
                if filter_kwargs['ftype'] == 'delay':
                    signal = delay(signal, **filter_kwargs)
        
        return signal

class StereoMicInRoom(Spatializer):
    def __init__(
        self, dimensions, rt60, fs, mic_spacing, directivity, src_dist, snr
    ) -> None:

        assert len(dimensions) in [2, 3]

        self.dimensions = np.array(dimensions)
        self.rt60 = rt60
        self.fs = fs

        self.absorption, self.max_order = pra.inverse_sabine(
            rt60=rt60, room_dim=dimensions
        )
        self.center = self.dimensions/2

        self.array_geometry = pra.beamforming.linear_2D_array(
            center=self.center, M=2, phi=0.0, d=mic_spacing
        )

        self.directivity = pra.directivities.CardioidFamily(
            orientation=pra.directivities.DirectionVector(azimuth=90.0, degrees=True),
            pattern_enum=directivity,
        )

        self.src_dist = src_dist

        self.snr = snr

    def create_room(self):
        return pra.ShoeBox(
            p=self.dimensions,
            fs=self.fs,
            absorption=self.absorption,
            max_order=self.max_order,
        ).add_microphone_array(
            pra.beamforming.MicrophoneArray(
                R=self.array_geometry,
                fs=self.fs,
                directivity=self.directivity,
            )
        )

        
    def generate_one(self, signal, angle):
        room = self.create_room()
        
        room = self.add_source_rel_center(room, signal, angle)
        
        signal = room.simulate(
            snr=self.snr, return_premix=True
        )
        
        return signal

    def add_source_rel_center(self, room: pra.Room, x, angular_loc):

        pos = self.src_dist * pra.beamforming.unit_vec2D(
            0.25 * np.pi + np.deg2rad(angular_loc)
        ) + self.center[:, None]
        
        return room.add_source(position=pos, signal=x)
        