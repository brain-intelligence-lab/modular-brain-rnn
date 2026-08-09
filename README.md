# Task-structured Modularity Emerges in Artificial Networks and Aligns with Brain Architecture

## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Training](#Training)
- [Plot Results](#plot-results)
- [License](./LICENSE)
- [Issues](https://github.com/brain-intelligence-lab/modular-brain-rnn/issues)

# Overview

This is the official repository for **Task-structured Modularity Emerges in Artificial Networks and Aligns with Brain Architecture**.  
In this study, we demonstrate that multitask and incremental learning enhance modularity in recurrent neural networks (RNNs) compared to single-task learning, revealing how functional demands influence the structural organization of neural networks.

![Schematics](./figures/Schematics.svg)

# Repo Contents

- [Data](./datasets/brain_hcp_data/84/): Data from The Human Connectome Project.
- [Python](./): Main Python source code (see `main.py`, `models/`, `utils/`, etc.).
- [Shell scripts](./scripts): Shell scripts are used to automate and manage multiple \
 parallel Python tasks (see `Fig2.a.sh`, `Fig2.b-h.sh`, `Fig3.a.sh`, etc.).

# System Requirements

## Hardware Requirements

- A modern CPU or GPU (NVIDIA recommended for deep learning tasks)
- Sufficient disk space for data and model checkpoints

### Dependencies

- Python 3.9
- PyTorch 1.13.1
- NumPy 1.23.5
- SciPy 1.13.1
- Bctpy 0.6.1
- (See [requirements.txt](./requirements.txt) for full list)

# Installation Guide

1. Clone the repository:
    ```bash
    git clone git@github.com:brain-intelligence-lab/modular-brain-rnn.git
    cd modular-brain-rnn
    ```
2. Create and activate a virtual environment:
    ```bash
    conda create -n mod_rnn python=3.9
    conda activate mod_rnn
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```


# Training

To reproduce the results, the shell scripts are here to automate and manage multiple parallel Python tasks.

To start single task learning, use Fig2.a.sh to train all 20 tasks independently:

```bash
./scripts/Fig2.a.sh
```
To start multi-task learning, use Fig2.b-h.sh to train multiple tasks together:

```bash
./scripts/Fig2.b-h.sh
```
For other experiments in the paper, just use the corresponding scripts.


# Plot Results
Once training is complete, you can generate the figures used in the paper by running the corresponding Python plot scripts:

```bash
python plot_Fig2.py
```

Results traning log and figures can be found in the `./runs` and `./figures` respectively.

NOTE:
Before each training session, please clear or rename the previous training's directory (e.g., mv ./runs/Fig2a ./runs/Fig2a_pre) to prevent the SummaryWriter from appending new log data, which could lead to incorrect plotting.


# Overall Workflow
![](./figures/workflow.svg)



This repository accompanies the manuscript:

Task-structured Modularity Emerges in Artificial Networks and Aligns with Brain Architecture

The manuscript is currently unpublished. Citation information for the paper will be updated upon publication.



