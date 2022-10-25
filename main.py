import os
from data.dataloader import StereoDatamodule
import pandas as pd
from tqdm import tqdm

from museval import evaluate

from datetime import datetime

def run_eval(dataset="TIMIT",
    room_dim=(3, 3),
    snr=None,
    rt60=0.150,
    src_dist=1.0,
    mic_spacing=0.60,
    directivity="CARDIOID",
    min_angle=0,
    max_angle=0,
    angle_step=30,
    min_error=-90,
    max_error=90,
    error_step=15,):

    kwargs = locals()

    timit_dm = StereoDatamodule(
        **kwargs
    )

    timit_dl = timit_dm.train_dataloader()

    results = []
    
    is_incomplete = False

    try:
        # if True:
        # print(len(timit_dm.train_dataset))

        for item in tqdm(timit_dm.train_dataset):
            n_sampl = item["xtrue"].shape[-1]

            sdr, isr, sir, sar = evaluate(
                item["xtrue"].transpose((0, 2, 1)),  # src, chan, sampl --> src, sampl, chan
                item["xest"].transpose((0, 2, 1)),
                win=n_sampl,
                hop=n_sampl,
                mode="v3",
            )

            results.append(
                {
                    "sdr": sdr[0][0],
                    "isr": isr[0][0],
                    "sir": sir[0][0],
                    "sar": sar[0][0],
                    **{
                        k: item[k]
                        for k in ["true_angle", "est_angle", "est_deviation", "file"]
                    },
                }
            )

            # print(
            #     item['true_angle'],
            #     item['est_angle'],
            #     item['est_deviation'],
            #     isr[0][0],
            # )

            # print(results[-1])

    except KeyboardInterrupt:
        is_incomplete = True
    finally:
        
        folder = f"./results/{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        os.makedirs(folder)
        
        import json
        with open(os.path.join(folder, 'args.json'), 'w') as f:
            json.dump(dict(kwargs), f, indent=4)
        
        df = pd.DataFrame(results)
        df.to_csv(f"{folder}/results-{'incomplete' if is_incomplete else 'ok'}.csv")
        
if __name__ == "__main__":
    import fire
    fire.Fire()
