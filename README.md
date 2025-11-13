# BabyLocoFormer

## Overview

BabyLocoFormer is an open-source unofficial baby version of LocoFormer. It includes the deployment of TransformerXL (not in its original form, but using flash attention, RoPE, SwiLU, etc.), multi-morph quadruped generation, limited domain randomization, and related training and evaluation.


## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    pip install -e source/babylocoformer
    cd rsl_rl && pip install -e .
    ```

## Usage

### List available tasks

```bash
python scripts/list_envs.py
```

### Run generation and conversion scripts

```bash
python model/generate_quad.py

python model/convert_to_usd.py.py --headless --input_dir <PATH>  --output_dir <PATH>
```

### Train and evaluate policies

```bash
python scripts/rsl_rl/train.py --task=Babylocoformer-v0 --num_env 2048 --headless --video

python scripts/rsl_rl/play.py --task=Babylocoformer-v0

python scripts/rsl_rl/play.py --task=Unitree-Go2-Velocity
```

## Results

[![Demo(Bilibili)](https://img.shields.io/badge/Demo-ff69b4?style=for-the-badge&logo=bilibili)](https://www.bilibili.com/video/BV14xCsB9EoY/?share_source=copy_web&vd_source=0de36dd681c4f7ffab776ec97939e21f)
[![Demo(YouTube)](https://img.shields.io/badge/Demo-FF0000?style=for-the-badge&logo=youtube)](https://youtu.be/7WqCCVZDSJo)

## 🌟 Contribute & Support
If you find this project useful or interesting, please consider **starring ⭐ this repository**!
Contributions, issues, and pull requests are all very welcome — every bit of feedback helps the project grow.
Let's make BabyLocoFormer even better together!
### Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit and run:

```bash
pip install pre-commit

pre-commit run --files <path/to/file>
```
### Some final thoughts
More robot morphologies help stabilize training, and stronger randomization improves adaptation, which makes sense.
Just messing around. Results are bad, mainly for learning. Trained on a single 5090.

## Acknowledgements
Thanks to the following projects for their great work and inspiration:


- [IsaacLab](https://github.com/isaac-sim/IsaacLab): The foundation for training and running codes.

- [Unitree_rl_lab
](https://github.com/unitreerobotics/unitree_rl_lab): The training and evaluation codes.

- [LocoFormer](https://generalist-locomotion.github.io/): The core ideas.
