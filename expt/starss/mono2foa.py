import glob
import os
import numpy as np
import soundfile as sf
from tqdm.contrib.concurrent import process_map

# from acoustics.ambisonics import sn3d


def mono2foa(mono: np.ndarray, azi: float, ele: float):

    azi = np.deg2rad(azi)
    ele = np.deg2rad(ele)
    wyzx = np.array(
        [
            1/np.sqrt(2),
            np.sin(azi) * np.cos(ele),
            np.sin(ele),
            np.cos(azi) * np.cos(ele),
        ]
    )

    foa = mono[None, :] * wyzx[:, None]

    return foa
    

def process_one(
    inout
):
    reference_file, output_file = inout

    ele = 30
    azi = np.arange(-180, 180, 30)

    ref, sr = sf.read(reference_file)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for a in azi:
        foa = mono2foa(ref, a, ele)
        sf.write(output_file.replace(".wav", f"_a{int(a)}e{ele}.flac"), foa.T, sr)
    

def process(
        reference_path="/home/kwatchar3/data/timit/timit/test", 
        output_path="/home/kwatchar3/data/timit/timit/test_foa"
    ):
    
    files = glob.glob(os.path.join(reference_path, "**", "sa*.wav"), recursive=True)
    outfiles = [f.replace(reference_path, output_path) for f in files]

    inout = list(zip(files, outfiles))

    process_map(process_one, inout, max_workers=8)

if __name__ == "__main__":
    process()