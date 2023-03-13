import glob
import os
import subprocess

from tqdm import tqdm

_MusdbOutputPath = "/home/kwatchar3/spauq-home/data/musdb-hq"
_MusdbInputBase = "/home/kwatchar3/spauq-home/data/musdb-hq/raw/musdb18hq/test"
_MusdbInputGlob = "*/mixture.wav"

def mp3(cbr, mode, *, dataset, audio_glob=None, output_path=None, q=3):
    assert mode in ["j", "s", "f"]
    encode_command = [
        "lame",
        f"-b {cbr}",
        f"-m {mode}",
        f"-q {q}",
        "--resample 44100",
    ]

    print(encode_command)

    decode_command = [
        "lame",
        "--decode",
    ]

    if output_path is None:
        if dataset == "musdb":
            ds_output = _MusdbOutputPath
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        output_path = os.path.join(ds_output, "mp3", f"{cbr}-{mode}")
        mp3_output_path = os.path.join(output_path, "mp3")
        wav_output_path = os.path.join(output_path, "wav")
        os.makedirs(mp3_output_path, exist_ok=True)
        os.makedirs(wav_output_path, exist_ok=True)

    if audio_glob is None:
        if dataset == "musdb":
            audio_glob = os.path.join(_MusdbInputBase, _MusdbInputGlob)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
    audio_files = sorted(glob.glob(audio_glob, recursive=False))
    print(f"Found {len(audio_files)} audio files")
    print(audio_files)
    mp3_files = [f.replace(_MusdbInputBase, mp3_output_path).replace(".wav", ".mp3") for f in audio_files]
    wav_files = [f.replace(_MusdbInputBase, wav_output_path) for f in audio_files]

    for a, m, w in tqdm(zip(audio_files, mp3_files, wav_files), total=len(audio_files)):
        os.makedirs(os.path.dirname(m), exist_ok=True)
        os.makedirs(os.path.dirname(w), exist_ok=True)
        ecmd = " ".join(encode_command + [f"\"{a}\"", f"\"{m}\""])
        print(ecmd)
        dcmd = " ".join(decode_command + [f"\"{m}\"", f"\"{w}\""])
        print(dcmd)
        out = subprocess.run(ecmd, capture_output=True, shell=True)
        if out.returncode != 0:
            print("Encoding failed")
            print(out)
            print(out.stderr)
            print(out.stdout)
            raise RuntimeError("Encoding failed")
        out = subprocess.run(dcmd, capture_output=True, shell=True)
        if out.returncode != 0:
            print("Decoding failed")
            print(out)
            print(out.stderr)
            print(out.stdout)
            raise RuntimeError("Decoding failed")
        
def opus(cbr, mode, *, dataset, audio_glob=None, output_path=None, q=3):
    assert mode in ["j", "s", "f"]
    encode_command = [
        "opusenc",
        f"-bitrate {cbr}",
        f"--hard-cbr",
    ]

    if dataset == "musdb":
        encode_command.append("--music")
    elif dataset == "timit":
        encode_command.append("--speech")

    print(encode_command)

    decode_command = [
        "opusdec",
    ]

    if output_path is None:
        if dataset == "musdb":
            ds_output = _MusdbOutputPath
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        output_path = os.path.join(ds_output, "mp3", f"{cbr}-{mode}")
        mp3_output_path = os.path.join(output_path, "mp3")
        wav_output_path = os.path.join(output_path, "wav")
        os.makedirs(mp3_output_path, exist_ok=True)
        os.makedirs(wav_output_path, exist_ok=True)

    if audio_glob is None:
        if dataset == "musdb":
            audio_glob = os.path.join(_MusdbInputBase, _MusdbInputGlob)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
    audio_files = sorted(glob.glob(audio_glob, recursive=False))
    print(f"Found {len(audio_files)} audio files")
    print(audio_files)
    mp3_files = [f.replace(_MusdbInputBase, mp3_output_path).replace(".wav", ".mp3") for f in audio_files]
    wav_files = [f.replace(_MusdbInputBase, wav_output_path) for f in audio_files]

    for a, m, w in tqdm(zip(audio_files, mp3_files, wav_files), total=len(audio_files)):
        os.makedirs(os.path.dirname(m), exist_ok=True)
        os.makedirs(os.path.dirname(w), exist_ok=True)
        ecmd = " ".join(encode_command + [f"\"{a}\"", f"\"{m}\""])
        print(ecmd)
        dcmd = " ".join(decode_command + [f"\"{m}\"", f"\"{w}\""])
        print(dcmd)
        out = subprocess.run(ecmd, capture_output=True, shell=True)
        if out.returncode != 0:
            print("Encoding failed")
            print(out)
            print(out.stderr)
            print(out.stdout)
            raise RuntimeError("Encoding failed")
        out = subprocess.run(dcmd, capture_output=True, shell=True)
        if out.returncode != 0:
            print("Decoding failed")
            print(out)
            print(out.stderr)
            print(out.stdout)
            raise RuntimeError("Decoding failed")

    
def mp3_multiple(dataset, audio_glob=None, output_path=None):
    modes = ["j", "s", "f"]
    cbrs = [32, 40, 48, 56, 64, 80, 96, 112, 128, 256, 320]
    for cbr in cbrs:
        for mode in modes:
            mp3(cbr, mode, dataset=dataset, audio_glob=audio_glob, output_path=output_path)

if __name__ == "__main__":
    import fire
    fire.Fire()
