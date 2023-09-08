# Designing Optimal Behavioral Experiments Using Machine Learning

This repository provides notebooks for running a simplified end-to-end example and for replicating all figures presented in the [paper](https://arxiv.org/pdf/2305.07721.pdf). 
See [Applying ML](practical_ml.md) for pointers on how to apply machine learning to BOED problems.

## Notebooks
* [Simplified example](notebooks/Tutorial_BOED_Example.ipynb) provides a detailed walk-through of the BOED procedure for a simplified example, where we optimize the design of one experimental block to estimate the parameters of the AEG model. 
* [Simulation study](notebooks/Tutorial_SimulationStudy.ipynb) provides code for generating the plots for figures 4 and 5.
* [Human participant study processing](notebooks/Tutorial_DataProcessing.ipynb) provides code for processing and analysing data from the human participant study.
* [Human participant study](notebooks/Tutorial_HumanParticipantExperiments.ipynb) provides code for generating the plots for figures 6, 7 and 8. 

## Scripts
* [Parameter estimation script](scripts/train_bo_pe.py) contains example code to obtain an optimized experimental design for parameter estimation. 
* [Example shell script: PE for AEG](scripts/example_job_script_pe_aeg.sh) Example script calling `train_bo_pe.py` to optimize the design for PE for the AEG model. 

## CPU Setup

Install conda dependencies and the project with

```bash
conda env create -f environment.yml
conda activate boed-tutorial
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
conda create -n boed-gpu python=3.8 pytorch torchvision cudatoolkit=10.1 -c pytorch
conda activate boed-gpu
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
