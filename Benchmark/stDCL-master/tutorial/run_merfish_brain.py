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
   parser.add_argument('--dataset', type=str, default="Donor_4_0")
   parser.add_argument('--datatype', type=str, default="MERFISH")
   parser.add_argument('--n_clusters', type=int, default=8)
   parser.add_argument('--prelr', type=float, default=0.05)
   parser.add_argument('--lr', type=float, default=0.05)
   parser.add_argument('--pre_epochs', type=int, default=500)
   parser.add_argument('--epochs', type=int, default=1000)
   parser.add_argument('--noise', type=int, default=2)
   parser.add_argument('--lambda1', type=float, default=5.0)
   parser.add_argument('--lambda2', type=float, default=0.1)
   parser.add_argument('--lambda3', type=float, default=0.1)
   parser.add_argument('--input', type=int, default=3000)
   parser.add_argument('--latent_dim', type=int, default=128)
   parser.add_argument('--n_neighbors', type=int, default=3, help='parameter k in spatial graph')
   parser.add_argument('--n_neighbors_gene', type=int, default=1, help='parameter k in spatial graph')
   parser.add_argument('--clustertype', type=str, default="mclust")
   parser.add_argument('--preprocess', type=bool, default=False)
   parser.add_argument('--radius', type=int, default=25)
   args = parser.parse_args()

   # Run device, by default, the package is implemented on 'cpu'. We recommend using GPU.
   args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

   os.environ['R_HOME'] = './lib/R'
   os.environ['R_USER'] = './lib/python3.8/site-packages/rpy2'

   dataset = 'Donor_4_0'

   adata = sc.read_h5ad('./MERFISH_frontal_cortex_striatum/'+dataset+'_data.h5ad')
   adata.var_names_make_unique()
   adata = train(adata, args)
   
   clustering(adata, args, refinement=True)

   # calculate metric ARI
   ARI = np.round(metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['tissue']), 2)
   NMI = np.round(metrics.normalized_mutual_info_score(adata.obs['domain'], adata.obs['tissue']), 2)
   HS = np.round(metrics.homogeneity_score(adata.obs['domain'], adata.obs['tissue']), 2)

   print('ARI:', ARI)
   print('NMI:', NMI)
   print('HS:', HS)


   
