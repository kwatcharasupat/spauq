import os

from spauq.wrapper.cli import spauq_eval_dir

ROOT = "/home/kwatchar3/Documents/BSU"

for folder in os.listdir(os.path.join(ROOT, "test")):
    if folder.startswith("."):
        continue

    print(folder)

    spauq_eval_dir(
        os.path.join(ROOT, "test", folder,
                     f"{folder}_reference_m24.wav"),
        os.path.join(ROOT, "test", folder),
        result_dir=os.path.join(ROOT, "eval2", folder),
        save_format="npz",
        return_framewise=True,
        return_cost=True,
        return_shift=True,
        return_scale=True,
        verbose=True,
    )