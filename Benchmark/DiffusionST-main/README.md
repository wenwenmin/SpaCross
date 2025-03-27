# DiffusionST: A diffusion model-based framework for enhancing spatial transcriptomics data quality and identifying spatial domains!


![image](https://github.com/cuiyaxuan/DiffusionST/blob/main/workflow.png)
## Tip: To facilitate researchers' usage, examples of our project can be run in the folder's IPython notebooks (after configuring the environment dependencies as described in the README). <br>


# Identifying spatial domains
##### Using python virtual environment with conda. Please create a Pytorch environment, install Pytorch and some other packages, such as "numpy","pandas", "scikit-learn" and "scanpy". See the requirements.txt file for an overview of the packages in the environment we used to produce our results. Alternatively, you can install the environment dependencies in the following sequence to minimize environment conflicts. <br>

##### Configure the corresponding R language environment and install the mclust package.

```R
conda create -n DiffusionST
source activate DiffusionST

conda install python=3.8

conda install conda-forge::gmp

conda install conda-forge::pot
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install scanpy
pip install anndata==0.8.0
pip install pandas==1.4.2
pip install rpy2==3.5.1
pip install rpy2==3.5.1
pip install scikit-learn==1.1.1
pip install scipy==1.8.1
pip install tqdm==4.64.0
pip install einops

```
```R
import rpy2.robjects as robjects
robjects.r('''
install.packages('mclust')
           ''')

```
