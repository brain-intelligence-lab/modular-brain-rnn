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
- [Citation](#citation)

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

4. Optional packages

    These packages are only necessary for some analyses and supplementary experiments mentioned in the paper.
    
    For Linear Mixed-Effects Analysis (Fig.2h) & Bipartite Network Community Detection:

    ```bash

    conda install rpy2
    R
    # Inside the R environment, install required packages:
    install.packages("lme4")
    install.packages("RLRsim")
    # Type 'quit()' to exit R

   ```
   For GNN Training, :
   ```bash
    # We need to install torch_geometric 
    # We first install pyg_lib torch_cluster torch_scatter torch_sparse torch_spline_conv 
    # from https://data.pyg.org/whl/torch-1.13.0%2Bcu117.html

    # Take pyg_lib for example:
    wget https://data.pyg.org/whl/torch-1.13.0%2Bcu117/pyg_lib-0.4.0%2Bpt113cu117-cp39-cp39-linux_x86_64.whl
    pip install pyg_lib-0.4.0+pt113cu117-cp39-cp39-linux_x86_64.whl 

    # Then we can install torch_geometric using:
    pip install torch_geometric
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





# Citation

If you use this code or data, please cite:
```
@article{YourCitation2025,
  title={Task-structured Modularity Emerges in Artificial Networks and Aligns with Brain Architecture},
  author={Your Name and Collaborators},
  journal={Journal Name},
  year={2025}
}
```



