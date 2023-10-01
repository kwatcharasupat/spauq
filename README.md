# SPAUQ: Spatial Audio Quality Evaluation

[![codecov](https://codecov.io/gh/karnwatcharasupat/spauq/branch/main/graph/badge.svg?token=N6GHIM48K4)](https://codecov.io/gh/karnwatcharasupat/spauq) 
[![CircleCI](https://dl.circleci.com/status-badge/img/gh/karnwatcharasupat/spauq/tree/main.svg?style=svg&circle-token=e9a1a1f3087725f6ab4726391e79a2fd213e5e71)](https://dl.circleci.com/status-badge/redirect/gh/karnwatcharasupat/spauq/tree/main)
[![CodeFactor](https://www.codefactor.io/repository/github/karnwatcharasupat/spauq/badge)](https://www.codefactor.io/repository/github/karnwatcharasupat/spauq)
[![arXiv](https://img.shields.io/badge/arXiv-2306.08053-b31b1b.svg)](https://arxiv.org/abs/2306.08053)
[![Documentation Status](https://readthedocs.org/projects/spauq/badge/?version=latest)](https://spauq.readthedocs.io/en/latest/?badge=latest)

SPAUQ is an Implementation of 
> K. N. Watcharasupat and A. Lerch, ``Quantifying Spatial Audio Quality Impairment'', submitted to ICASSP 2024.

# Installation

```shell
pip install git+https://github.com/karnwatcharasupat/spauq.git@main
```

# CLI Usage

## Evaluating one test file against one reference file
```shell
spauq-eval-file --reference_path="path/to/ref.wav" \
  --estimate_path="path/to/est.wav"
```

## Evaluating multiple test files against one reference file
```shell
spauq-eval-dir --reference_path="path/to/ref.wav" \
  --estimate_dir="path/to/many/estimates \
  --estimate_ext=".wav"
```

# Programmatic usage

```python
from spauq.core.metrics import spauq_eval
import soundfile as sf

reference, fs = sf.read("path/to/ref.wav")
estimate, fse = sf.read("path/to/est.wav")

assert fs == fse

eval_output = spauq_eval(
    reference=reference,
    estimate=estimate,
    fs=fs
)

signal_to_spatial_distortion_ratio = eval_output["SSR"]
signal_to_residual_distortion_ratio = eval_output["SRR"]
```