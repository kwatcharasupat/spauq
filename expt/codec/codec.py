import glob
import os
import subprocess

from tqdm import tqdm

_MusdbOutputPath = "/home/kwatchar3/spauq-home/data/musdb-hq"
_MusdbInputBase = "/home/kwatchar3/spauq-home/data/musdb-hq/raw/musdb18hq/test"
_MusdbInputGlob = "*/mixture.wav"

_StarssOutputPath = "/home/kwatchar3/spauq-home/data/starss"
_StarssInputBase = "/home/kwatchar3/data/starss22/mic_eval"
_StarssInputGlob = "*.wav"


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
        elif dataset == "starss":
            ds_output = _StarssOutputPath
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
        elif dataset == "starss":
            audio_glob = os.path.join(_StarssInputBase, _StarssInputGlob)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    if dataset == "musdb":
        input_base = _MusdbInputBase
    elif dataset == "starss":
        input_base = _StarssInputBase
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    audio_files = sorted(glob.glob(audio_glob, recursive=False))
    print(f"Found {len(audio_files)} audio files")
    print(audio_files)
    mp3_files = [
        f.replace(input_base, mp3_output_path).replace(".wav", ".mp3")
        for f in audio_files
    ]
    wav_files = [f.replace(input_base, wav_output_path) for f in audio_files]

    for a, m, w in tqdm(zip(audio_files, mp3_files, wav_files), total=len(audio_files)):
        os.makedirs(os.path.dirname(m), exist_ok=True)
        os.makedirs(os.path.dirname(w), exist_ok=True)
        ecmd = " ".join(encode_command + [f'"{a}"', f'"{m}"'])
        print(ecmd)
        dcmd = " ".join(decode_command + [f'"{m}"', f'"{w}"'])
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


def opus(cbr, comp, *, dataset, audio_glob=None, output_path=None):
    encode_command = [
        "opusenc",
        f"--bitrate {cbr}",
        f"--hard-cbr",
        f"--comp {comp}",
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
        elif dataset == "starss":
            ds_output = _StarssOutputPath
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        output_path = os.path.join(ds_output, "opus", f"{cbr}-{comp}")
        opus_output_path = os.path.join(output_path, "opus")
        wav_output_path = os.path.join(output_path, "wav")
        os.makedirs(opus_output_path, exist_ok=True)
        os.makedirs(wav_output_path, exist_ok=True)

    if dataset == "musdb":
        input_base = _MusdbInputBase
    elif dataset == "starss":
        input_base = _StarssInputBase
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    audio_files = sorted(glob.glob(audio_glob, recursive=False))
    print(f"Found {len(audio_files)} audio files")
    print(audio_files)
    opus_files = [
        f.replace(input_base, opus_output_path).replace(".wav", ".opus")
        for f in audio_files
    ]
    wav_files = [f.replace(input_base, wav_output_path) for f in audio_files]

    for a, m, w in tqdm(
        zip(audio_files, opus_files, wav_files), total=len(audio_files)
    ):
        os.makedirs(os.path.dirname(m), exist_ok=True)
        os.makedirs(os.path.dirname(w), exist_ok=True)
        ecmd = " ".join(encode_command + [f'"{a}"', f'"{m}"'])
        print(ecmd)
        dcmd = " ".join(decode_command + [f'"{m}"', f'"{w}"'])
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


def aac(abr, mode, *, dataset, audio_glob=None, output_path=None, q=100):
    assert mode in [0, 1, 2]
    encode_command = [
        "faac",
        f"-b {abr}",
        f"--joint {mode}",
        f"-q {q}",
        "--mpeg-vers 4",
    ]

    print(encode_command)

    decode_command = [
        "faad",
    ]

    if output_path is None:
        if dataset == "musdb":
            ds_output = _MusdbOutputPath
        elif dataset == "starss":
            ds_output = _StarssOutputPath
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        output_path = os.path.join(ds_output, "aac", f"{abr}-joint{mode}")
        aac_output_path = os.path.join(output_path, "aac")
        wav_output_path = os.path.join(output_path, "wav")
        os.makedirs(aac_output_path, exist_ok=True)
        os.makedirs(wav_output_path, exist_ok=True)

    if audio_glob is None:
        if dataset == "musdb":
            audio_glob = os.path.join(_MusdbInputBase, _MusdbInputGlob)
        elif dataset == "starss":
            audio_glob = os.path.join(_StarssInputBase, _StarssInputGlob)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    if dataset == "musdb":
        input_base = _MusdbInputBase
    elif dataset == "starss":
        input_base = _StarssInputBase
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    audio_files = sorted(glob.glob(audio_glob, recursive=False))
    print(f"Found {len(audio_files)} audio files")
    print(audio_files)
    aac_files = [
        f.replace(input_base, aac_output_path).replace(".wav", ".aac")
        for f in audio_files
    ]
    wav_files = [f.replace(input_base, wav_output_path) for f in audio_files]

    for a, m, w in tqdm(zip(audio_files, aac_files, wav_files), total=len(audio_files)):
        os.makedirs(os.path.dirname(m), exist_ok=True)
        os.makedirs(os.path.dirname(w), exist_ok=True)
        ecmd = " ".join(encode_command + [f'-o "{m}"', f'"{a}"'])
        print(ecmd)
        dcmd = " ".join(decode_command + [f'-o "{w}"', f'"{m}"'])
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
    cbrs = [160, 192, 224]  # [32, 40, 48, 56, 64, 80, 96, 112, 128, 256, 320]
    for cbr in cbrs:
        for mode in modes:
            mp3(
                cbr,
                mode,
                dataset=dataset,
                audio_glob=audio_glob,
                output_path=output_path,
            )


def aac_multiple(dataset, audio_glob=None, output_path=None):
    modes = [2, 1, 0]
    cbrs = [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320]
    for cbr in cbrs:
        for mode in modes:
            aac(
                cbr,
                mode,
                dataset=dataset,
                audio_glob=audio_glob,
                output_path=output_path,
            )


def opus_multiple(dataset, audio_glob=None, output_path=None):
    cbrs = [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320]
    comps = [0]
    for cbr in cbrs:
        for comp in comps:
            opus(
                cbr,
                comp,
                dataset=dataset,
                audio_glob=audio_glob,
                output_path=output_path,
            )


if __name__ == "__main__":
    import fire

    fire.Fire()
