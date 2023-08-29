from collections import defaultdict
import glob
import os
import numpy as np
import pandas as pd

import soundfile as sf
from tqdm import tqdm

from spauq.core.metrics import spauq_eval


def load_metadata(filename):
    df = pd.read_csv(filename, header=None, names=["frame", "class", "source_id", "azi", "ele"])
    return df

def evaluate(
        metadata_dir="/home/kwatchar3/spauq-home/data/starss22/metadata_dev/test",
        audio_dir="/home/kwatchar3/spauq-home/data/starss22/foa_dev_spat",
        reference_tag = "5ch-30-n30-0-110-n110",
        estimate_tag = "5ch-30-n30-0-110-n120",
        frame_size_seconds=0.1,
        min_segment_size_second=5.0,
    ):

    files = glob.glob(os.path.join(metadata_dir, "**", "*.csv"), recursive=True)

    allgroups = defaultdict(list)

    for f in files:
        df = load_metadata(f)
        short_f = os.sep.join(f.split(os.sep)[-2:])
        frame_by_nsrc = df.groupby("frame").count()["class"]
        n_src_change = frame_by_nsrc.diff().abs() > 0
        
        df2 = pd.DataFrame({"n_src_change": n_src_change, "n_src": frame_by_nsrc})

        groups = []

        group_start = 0
        group_end = None
        group_source = None
        for frame, row in df2.iterrows():
            if row["n_src_change"]:
                if frame_size_seconds * (group_end - group_start) > min_segment_size_second and group_source is not None:
                    groups.append((group_start, group_end, group_source))
                group_start = frame
                group_source = row["n_src"]
            group_end = frame

        allgroups[short_f] += groups

    outs = []

    # print(allgroups)

    for file, groups_ in tqdm(allgroups.items()):
        
        reffile = os.path.join(audio_dir, reference_tag, file.replace(".csv", ".wav"))
        estfile = os.path.join(audio_dir, estimate_tag, file.replace(".csv", ".wav"))

        ref, fs = sf.read(reffile)
        est, fs = sf.read(estfile)

        max_abs = max(np.max(np.abs(ref)), np.max(np.abs(est)))

        # print(max_abs)

        if max_abs < 1e-3:
            print(f"Skipping {file} due to low energy")
            continue
        # print(max_abs)

        ref /= max_abs
        est /= max_abs

        
        for group in tqdm(groups_):
            start_frame, end_frame, source = group
            start_sample = int(start_frame * frame_size_seconds * fs)
            end_sample = int((end_frame + 1)* frame_size_seconds * fs)
            ref_ = ref[start_sample:end_sample]
            est_ = est[start_sample:end_sample]

            rms = np.sqrt(np.mean(ref_**2))
            if rms < 1e-6:
                # print(f"Skipping {file} due to low energy")
                continue
            
            # print(rms)

            out = spauq_eval(
                reference=ref_.T, 
                estimate=est_.T, 
                fs=fs, 
                return_framewise=True,
            )

            # print(out["SSR"], out["SRR"])

            for k, v in out.items():
                for i in range(v.shape[0]):
                    outs.append({
                        "file": file,
                        "source": source,
                        "metric": k,
                        "value": v[i],
                    })

    np.savez("starss22.npz", **out)


def evaluate_full(
        audio_dir="/home/kwatchar3/spauq-home/data/starss22/foa_dev_spat",
        out_dir="/home/kwatchar3/spauq-home/data/starss22/foa_dev_spat_eval",
        reference_tag = "5ch-30-n30-0-110-n110",
        estimate_tag = "5ch-30-n30-0-110-n120",
    ):

    reffiles = glob.glob(os.path.join(audio_dir, reference_tag, "**", "*.wav"), recursive=True)
    
    refiles = [
        (rf, rf.replace(reference_tag, estimate_tag) )
            for rf in reffiles
        
    ]

    out_dir = os.path.join(out_dir, f"{estimate_tag}---{reference_tag}")

    os.makedirs(out_dir, exist_ok=True)

    for reffile, estfile in tqdm(refiles):


        ref, fs = sf.read(reffile)
        est, fs = sf.read(estfile)

        try:
            out = spauq_eval(
                reference=ref.T, 
                estimate=est.T, 
                fs=fs, 
                return_framewise=True,
            )

            filename = os.sep.join(reffile.split(os.sep)[-2:]).replace(".wav", ".npz")

            outpath = os.path.join(out_dir, filename)

            os.makedirs(os.path.dirname(outpath), exist_ok=True)

            np.savez(outpath, **out)
        except:
            continue

if __name__ == "__main__":
    import fire
    fire.Fire()