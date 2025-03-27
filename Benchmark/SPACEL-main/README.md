[![Documentation Status](https://readthedocs.org/projects/spacel/badge/?version=latest)](https://spacel.readthedocs.io/en/latest/?badge=latest)
![PyPI](https://img.shields.io/pypi/v/SPACEL)

# SPACEL: characterizing spatial transcriptome architectures by deep-learning

![](docs/_static/img/figure1.png "Overview")
SPACEL (**SP**atial **A**rchitecture **C**haracterization by d**E**ep **L**earning) is a Python package of deep-learning-based methods for ST data analysis. SPACEL consists of three modules: 
* Spoint embedded a multiple-layer perceptron with a probabilistic model to deconvolute cell type composition for each spot on single ST slice.
* Splane employs a graph convolutional network approach and an adversarial learning algorithm to identify uniform spatial domains that are transcriptomically and spatially coherent across multiple ST slices.
* Scube automatically transforms the spatial coordinate systems of consecutive slices and stacks them together to construct a three-dimensional (3D) alignment of the tissue.

## Getting started
* [Requirements](#Requirements)
* [Installation](#Installation)
* Tutorials
    * [Spoint tutorial: Deconvolution of cell types compostion on human brain Visium dataset](docs/tutorials/Visium_human_DLPFC_Spoint.ipynb)
    * [Splane tutorial: Identify uniform spatial domain on human breast cancer Visium dataset](docs/tutorials/Visium_human_breast_cancer_Splane.ipynb)
    * [Splane&Scube tutorial (1/2): Identify uniform spatial domain on human brain MERFISH dataset](docs/tutorials/MERFISH_mouse_brain_Splane.ipynb)
    * [Splane&Scube tutorial (1/2): Alignment of consecutive ST slices on human brain MERFISH dataset](docs/tutorials/MERFISH_mouse_brain_Scube.ipynb)
    * [Scube tutorial: Alignment of consecutive ST slices on mouse embryo Stereo-seq dataset](docs/tutorials/Stereo-seq_Scube.ipynb)
    * [Scube tutorial: 3D expression modeling with gaussian process regression](docs/tutorials/STARmap_mouse_brain_GPR.ipynb)
    * [SPACEL workflow (1/3): Deconvolution by Spoint on mouse brain ST dataset](docs/tutorials/ST_mouse_brain_Spoint.ipynb)
    * [SPACEL workflow (2/3): Identification of spatial domain by Splane on mouse brain ST dataset](docs/tutorials/ST_mouse_brain_Splane.ipynb)
    * [SPACEL workflow (3/3): Alignment 3D tissue by Scube on mouse brain ST dataset](docs/tutorials/ST_mouse_brain_Scube.ipynb)

Read the [documentation](https://spacel.readthedocs.io) for more information.

## Latest updates
### Version 1.1.8 2024-07-23
#### Fixed Bugs
- Fixed the conflict between optax version and phthon 3.8.

### Version 1.1.7 2024-01-16
#### Fixed Bugs
- Fixed a variable reference error in function `identify_spatial_domain`. Thanks to @tobias-zehnde for the contribution.

### Version 1.1.6 2023-07-27
#### Fixed Bugs
- Fixed a bug regarding the similarity loss weight hyperparameter `simi_l`, which in the previous version did not affect the loss value.

## Requirements
**Note**: The current version of SPACEL only supports Linux and MacOS, not Windows platform. 

To install `SPACEL`, you need to install [PyTorch](https://pytorch.org) with GPU support first. If you don't need GPU acceleration, you can just skip the installation for `cudnn` and `cudatoolkit`.
* Create conda environment for `SPACEL`:
```
conda env create -f environment.yml
```
or
```
conda create -n SPACEL -c conda-forge -c default cudatoolkit=10.2 python=3.8 rpy2 r-base r-fitdistrplus
```
You must choose correct `PyTorch`, `cudnn` and `cudatoolkit` version dependent on your graphic driver version. 

Note: If you want to run 3D expression GPR model in Scube, you need to install the [Open3D](http://www.open3d.org/docs/release/) python library first.

## Installation
* Install `SPACEL`:
```
pip install SPACEL
```
* Test if [PyTorch](https://pytorch.org) for GPU available:
```
python
>>> import torch
>>> torch.cuda.is_available()
```
If these command line have not return `True`, please check your gpu driver version and `cudatoolkit` version. For more detail, look at [CUDA Toolkit Major Component Versions](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions).
