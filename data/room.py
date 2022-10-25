import numpy as np

import pyroomacoustics as pra


class StereoMicInRoom:
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
        