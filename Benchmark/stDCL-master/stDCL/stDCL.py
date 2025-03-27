import torch
from stDCL.preprocess import *
import time
import random
import numpy as np
from model import Encoder
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd
from sklearn.cluster import KMeans
from stDCL.utils import *


def train(adata, args):
    
    if args.datatype == '10x':
        model = stDCL(adata, args)
        latent,imp = model.train(2)
        adata.obsm['emb'] = latent
        adata.obsm['imp'] = imp

        adata1 = adata.copy()
        model = stDCL(adata1, args)
        latent,imp = model.train(1.5)
        adata.obsm['emb1'] = latent
        adata.obsm['imp1'] = imp
        return adata
    else:
        model = stDCL(adata, args)
        latent,imp = model.train(args.noise)
        adata.obsm['emb'] = latent
        adata.obsm['imp'] = imp
        return adata

class stDCL():
    def __init__(self, adata, args):
        self.adata = adata.copy()
        self.device = args.device
        self.seed = args.seed
        self.n_clusters = args.n_clusters
        self.learning_rate_pre=args.prelr
        self.learning_rate=args.lr
        self.weight_decay=0.00
        self.pre_epochs = args.pre_epochs
        self.epochs=args.epochs
        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3
        self.input = args.input
        self.latent_dim = args.latent_dim
        self.n_neighbors = args.n_neighbors
        self.n_neighbors_gene = args.n_neighbors_gene
        self.datatype = args.datatype
        self.preprocess = args.preprocess
        
        fix_seed(self.seed)

        if self.preprocess:
            preprocess(self.adata,self.input)
           
        get_feature(self.adata,self.preprocess)

        if 'adj' not in adata.obsm.keys():
           construct_interaction(self.adata, self.datatype, self.n_neighbors_gene, self.n_neighbors)
        
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.adj = self.adata.obsm['adj']
    
        self.dim_input = self.features.shape[1]
        
        self.adj = preprocess_adj(self.adj)
        self.adj = torch.FloatTensor(self.adj).to(self.device)
        
            
    def pretrain(self):
        self.optimizer_pre = torch.optim.Adam(self.model.parameters(), self.learning_rate_pre, 
                                          weight_decay=self.weight_decay)

        for epoch in tqdm(range(self.pre_epochs)): 
            self.model.train()
        
            self.hiden_feat, self.z_a, self.emb, ret, ret_a, A_rec= self.model(self.features, self.features_a, self.adj)
            loss_rec = F.mse_loss(self.features, self.emb) + 0.5*F.mse_loss(self.adj, A_rec)
            loss_clu = self.correlation_reduction_loss(self.cross_correlation(ret.t(), ret_a.t()))
            loss = self.lambda1*loss_rec + loss_clu
            
            self.optimizer_pre.zero_grad()
            loss.backward() 
            self.optimizer_pre.step()

        with torch.no_grad():
            self.emb_rec = self.model(self.features, self.features_a, self.adj)[0].detach().cpu().numpy()
            self.adata.obsm['emb'] = self.emb_rec


    def train(self, var):
        self.model = Encoder(self.dim_input, self.latent_dim, self.n_clusters).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                          weight_decay=self.weight_decay)
        
        print('Begin to train')
        self.model.train()
        self.pretrain()
        self.features_a = self.features * torch.Tensor(np.random.normal(0, var, self.features.shape)).to(self.device)

        for epoch in tqdm(range(self.epochs)): 
            self.model.train()
            
            self.hiden_feat, self.z_a, self.emb, ret, ret_a, A_rec = self.model(self.features, self.features_a, self.adj)
            loss_rec = F.mse_loss(self.features, self.emb) + 0.5*F.mse_loss(self.adj, A_rec)
            loss_con = F.mse_loss(self.cross_correlation(self.hiden_feat, self.z_a), self.adj)
            loss_clu = self.correlation_reduction_loss(self.cross_correlation(ret.t(), ret_a.t()))
            loss = self.lambda1*loss_rec + self.lambda2*loss_con + self.lambda3*loss_clu

            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()

        print("Optimization finished")
        with torch.no_grad():
            self.model.eval()
            self.emb_rec = self.model(self.features, self.features_a, self.adj)[0].detach().cpu().numpy() 
            self.imp = self.model(self.features, self.features_a, self.adj)[2].detach().cpu().numpy()
        return self.emb_rec, self.imp
    
    def cross_correlation(self, Z_v1, Z_v2):
        """
        calculate the cross-view correlation matrix S
        Args:
            Z_v1: the first view embedding
            Z_v2: the second view embedding
        Returns: S
        """
        return torch.mm(F.normalize(Z_v1, dim=1), F.normalize(Z_v2, dim=1).t()) 

    def correlation_reduction_loss(self, S):
        """
        the correlation reduction loss L: MSE for S and I (identical matrix)
        Args:
            S: the cross-view correlation matrix S
        Returns: L
        """
        return torch.diagonal(S).add(-1).pow(2).mean() + self.off_diagonal(S).pow(2).mean()
    
    def off_diagonal(self, x):
        """
        off-diagonal elements of x
        Args:
            x: the input matrix
        Returns: the off-diagonal elements of x
        """
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()