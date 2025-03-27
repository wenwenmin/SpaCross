import os
import ot
import torch
import random
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from torch.backends import cudnn
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from sklearn.neighbors import NearestNeighbors 
from stDCL.utils import *
from sklearn.cluster import KMeans


def permutation(feature):
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    
    return feature_permutated 



def construct_interaction(adata, datatype, n_neighbors_gene, n_neighbors=3):#3

    position = adata.obsm['spatial']
    # calculate distance matrix
    distance_matrix = ot.dist(position, position, metric='euclidean')
    n_spot = distance_matrix.shape[0]
    
    adata.obsm['distance_matrix'] = distance_matrix
    
    interaction = np.zeros([n_spot, n_spot])  
    for i in range(n_spot):
        vec = distance_matrix[i, :]
        distance = vec.argsort()
        for t in range(1, n_neighbors + 1):
            y = distance[t]
            interaction[i, y] = 1
         
    adata.obsm['graph_neigh'] = interaction

    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)

    if datatype == 'osmFISH':
        sc.pp.pca(adata, n_comps=30, random_state=0)
    elif datatype == 'MERFISH':
        sc.pp.pca(adata, n_comps=30, random_state=1)
    else:
        sc.pp.pca(adata, n_comps=50, random_state=1)
    pca_data = adata.obsm['X_pca']
    knn = NearestNeighbors(n_neighbors=n_neighbors_gene, metric='euclidean').fit(pca_data)
    distances, indices = knn.kneighbors(pca_data)

    for i in range(adj.shape[0]):
        for j in indices[i]:
            if i != j and adj[i, j] == 0: 
                adj[i, j] = adj[j, i] = 0.5

    adata.obsm['adj'] = adj


def preprocess(adata, top_genes):
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=top_genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    
def get_feature(adata,preprocess):
    if not preprocess:
        adata_Vars = adata
    else:   
        adata_Vars =  adata[:, adata.var['highly_variable']]
       
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
        feat = adata_Vars.X.toarray()[:, ]
    else:
        feat = adata_Vars.X[:, ] 
    
    feat_a = permutation(feat)
    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a    
    
    
def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized 

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
   

def fix_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
    
    
