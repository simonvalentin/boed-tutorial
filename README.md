# Designing Optimal Experiments Using Machine Learning

This repository provides notebooks for running a minimal example end-to-end and replicating all figures presented in the paper. 

## TODOs
- [ ] Finish notebooks
  - [ ] Toy-example simulation
  - [ ] Simulation study
  - [ ] Human-participant study
- [ ] Provide instructions for scaling up method
- [ ] (Optional) Build minimal version of experiment that doesn't save data and put on repo. 
- [ ] (Optional) Create static (github.io or netlify) page that provides html-rendered versions of the notebooks. 
- [ ] (Optional) Create google colab links (note might require setting up the environment appropriately). 


## CPU Setup

Install conda dependencies and the project with

```bash
conda env create -f environment.yml
conda activate modelcomp
python setup.py develop
```

If the dependencies in `environment.yml` change, update dependencies with

```bash
conda env update --file environment.yml
```

## GPU Cluster Setup

Check local versions of cuda available: ls -d /opt/cu*. You should use one of these (e.g. the latest version) for the cudatoolkit=??.? argument below.

Create a Conda environment with GPU-enabled PyTorch (with e.g. Cuda 10.1): 

```bash
conda create -n modelcomp-gpu python=3.8 pytorch torchvision cudatoolkit=10.1 -c pytorch
conda activate modelcomp-gpu
```

Then install dependencies in the GPU environment file:

```bash
conda env update --file environment-gpu.yml
```

Finally, install the Ax platform:

```bash
pip install ax-platform
´´´

The above command with the environment file can also be used to update the Conda environment when dependencies in the environment file change.
