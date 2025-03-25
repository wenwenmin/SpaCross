import os
import random

import numpy as np
import torch
import torch.nn.modules.loss
import torch.nn.functional as F
from tqdm import tqdm
from torch.backends import cudnn

from .Models import SpaCross_model
from .GLNS import GLNSampler, GLNSampler_BC

class SC_pipeline:
    def __init__(self, adata, edge_index, num_clusters, device, config, roundseed=0, imputation=False):
        seed = config['seed'] + roundseed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True)

        self.device = device
        self.adata = adata
        self.edge_index = edge_index
        self.train_config = config['train']
        self.model_config = config['model']
        self.num_clusters = num_clusters
        self.imputation = imputation

        if self.imputation:
            self.X = torch.FloatTensor(self.adata.X.copy()).to(self.device)
        else:
            self.X = torch.FloatTensor(self.adata.obsm['X_pca'].copy()).to(self.device)
        self.edge_index = self.edge_index.to(self.device)

        self.input_dim = self.X.shape[-1]
        self.model = SpaCross_model(self.input_dim, self.model_config, imputation=self.imputation).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=0.001,
            weight_decay=3e-4,
        )

        self.sampler = GLNSampler(self.num_clusters, self.device)
        self.anchor_pair = None

    def trian(self):
        neighbors = self.train_config['topk_neighs']
        pbar = tqdm(range(self.train_config['epochs']))
        for epoch in pbar:
            if epoch % self.train_config['t_step'] == 0 and epoch > 1:
                self.model.eval()
                s_rep, t_rep = self.model.std_tgt_embedding(self.X, self.edge_index)
                self.anchor_pair = self.sampler(self.edge_index, F.normalize(s_rep, dim=-1, p=2),
                                                F.normalize(t_rep, dim=-1, p=2), neighbors, cluster_method="kmeans")

            self.model.train()
            self.optimizer.zero_grad()
            mean_loss, rec_loss, tri_loss = self.model(self.X, self.edge_index, self.anchor_pair)
            loss = self.train_config['w_recon'] * rec_loss + self.train_config['w_mean'] * mean_loss + \
                   self.train_config['w_tri'] * tri_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            self.optimizer.step()
            with torch.no_grad():
                self.model.momentum_update()
            pbar.set_description(
                "Epoch {0} total loss={1:.3f} recon loss={2:.3f} mean loss={3:.3f} tri loss={4:.3f}".format(
                    epoch, loss, rec_loss, mean_loss, tri_loss),
                refresh=True)

    def process(self):
        self.model.eval()
        enc_rep, recon = self.model.evaluate(self.X, self.edge_index)
        enc_rep = enc_rep.to('cpu').detach().numpy()
        recon = recon.to('cpu').detach().numpy()
        recon[recon < 0] = 0

        self.adata.obsm['latent'] = enc_rep
        self.adata.obsm['ReX'] = recon
        return enc_rep, recon


class SC_BC_pipeline:
    def __init__(self, adata, edge_index, num_clusters, device, config, roundseed=0, imputation=False):
        seed = config['seed'] + roundseed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True)

        self.device = device
        self.adata = adata
        self.edge_index = edge_index
        self.train_config = config['train']
        self.model_config = config['model']
        self.num_clusters = num_clusters
        self.imputation = imputation
        self.batch_id = torch.tensor(adata.obs['slice_id'].to_numpy(), dtype=torch.float32)

        if self.imputation:
            self.X = torch.FloatTensor(self.adata.X.copy()).to(self.device)
        else:
            self.X = torch.FloatTensor(self.adata.obsm['X_pca'].copy()).to(self.device)
        self.edge_index = self.edge_index.to(self.device)

        self.input_dim = self.X.shape[-1]
        self.model = SpaCross_model(self.input_dim, self.model_config, imputation=self.imputation).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()),
            lr=0.001,
            weight_decay=3e-4,
        )

        self.sampler = GLNSampler_BC(self.num_clusters, self.device)
        self.anchor_pair = None

    def trian(self):
        neighbors = self.train_config['topk_neighs']
        neighbors_inter = self.train_config['topk_neighs_inter']
        pbar = tqdm(range(self.train_config['epochs']))
        for epoch in pbar:
            if epoch % self.train_config['t_step'] == 0 and epoch > 1:
                self.model.eval()
                s_rep, t_rep = self.model.std_tgt_embedding(self.X, self.edge_index)
                # (self, adj, enc_rep, batch_id, top_k, top_k_inter, cluster_method="kmeans")
                self.anchor_pair = self.sampler(self.edge_index, F.normalize(s_rep, dim=-1, p=2), self.batch_id, neighbors, neighbors_inter, cluster_method="kmeans")

            self.model.train()
            self.optimizer.zero_grad()
            mean_loss, rec_loss, tri_loss = self.model(self.X, self.edge_index, self.anchor_pair)
            loss = self.train_config['w_recon'] * rec_loss + self.train_config['w_mean'] * mean_loss + \
                   self.train_config['w_tri'] * tri_loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
            self.optimizer.step()
            with torch.no_grad():
                self.model.momentum_update()
            pbar.set_description(
                "Epoch {0} total loss={1:.3f} recon loss={2:.3f} mean loss={3:.3f} tri loss={4:.3f}".format(
                    epoch, loss, rec_loss, mean_loss, tri_loss),
                refresh=True)

    def process(self):
        self.model.eval()
        enc_rep, recon = self.model.evaluate(self.X, self.edge_index)
        enc_rep = enc_rep.to('cpu').detach().numpy()
        recon = recon.to('cpu').detach().numpy()
        recon[recon < 0] = 0

        self.adata.obsm['latent'] = enc_rep
        self.adata.obsm['ReX'] = recon
        return enc_rep, recon
