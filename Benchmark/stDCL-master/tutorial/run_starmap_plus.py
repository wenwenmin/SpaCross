import os
import torch
import pandas as pd
import scanpy as sc
from sklearn import metrics
from stDCL.stDCL import *
from stDCL.utils import clustering
import argparse

if __name__ == '__main__': 

   parser = argparse.ArgumentParser(description='stDCL', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

   # setting
   parser.add_argument('--cuda', type=bool, default=True)
   parser.add_argument('--seed', type=int, default=1)
   parser.add_argument('--dataset', type=str, default="STARmap_plus")
   parser.add_argument('--datatype', type=str, default="STARmap")
   parser.add_argument('--n_clusters', type=int, default=12)
   parser.add_argument('--prelr', type=float, default=0.001)
   parser.add_argument('--lr', type=float, default=0.01)
   parser.add_argument('--pre_epochs', type=int, default=500)
   parser.add_argument('--epochs', type=int, default=1000)
   parser.add_argument('--noise', type=int, default=1)
   parser.add_argument('--lambda1', type=float, default=5)
   parser.add_argument('--lambda2', type=float, default=3)
   parser.add_argument('--lambda3', type=float, default=3)
   parser.add_argument('--input', type=int, default=3000)
   parser.add_argument('--latent_dim', type=int, default=64)
   parser.add_argument('--n_neighbors', type=int, default=2, help='parameter k in spatial graph')
   parser.add_argument('--n_neighbors_gene', type=int, default=3, help='parameter k in spatial graph')
   parser.add_argument('--clustertype', type=str, default="leiden")
   parser.add_argument('--preprocess', type=bool, default=True)
   parser.add_argument('--radius', type=int, default=50)
   args = parser.parse_args()

   # Run device, by default, the package is implemented on 'cpu'. We recommend using GPU.
   args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

   os.environ['R_HOME'] = './lib/R'
   os.environ['R_USER'] = './lib/python3.8/site-packages/rpy2'
   

   adata = sc.read_h5ad("./STARmap_PLUS_ad/8months-control-replicate_1_data.h5ad")
   adata.var_names_make_unique()
   adata = train(adata, args)

   clustering(adata, args, refinement=False,end=2.0)
   import matplotlib.pyplot as plt
   plt.rcParams["figure.figsize"] = (6, 6)
   sc.pl.embedding(adata,basis='spatial',color=['domain'],show=True,size=30)





