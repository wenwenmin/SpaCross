import numpy as np
import torch
import scipy.sparse as sp
import anndata as ad
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

def graph_construction3D(adata_st_list,  # list of spatial transcriptomics (ST) anndata objects
               section_ids=None,
               three_dim_coor=None,  # if not None, use existing 3d coordinates in shape [# of total spots, 3]
               coor_key="spatial_aligned",  # "spatial_aligned" by default
               rad_cutoff=None,  # cutoff radius of spots for building graph
               rad_coef=1.5,  # if rad_cutoff=None, rad_cutoff is the minimum distance between spots multiplies rad_coef
               k_cutoff=12,
               slice_dist_micron=None,  # pairwise distances in micrometer for reconstructing z-axis
               c2c_dist=100,  # center to center distance between nearest spots in micrometer
               mode='KNN',
               norm=False,
               ):
    adata_st = ad.concat(adata_st_list, label="slice_name", keys=section_ids)
    # adata_st.obs['Ground Truth'] = adata_st.obs['Ground Truth'].astype('category')
    adata_st.obs["slice_name"] = adata_st.obs["slice_name"].astype('category')
    # Build a graph for spots across multiple slices
    print("Start building a graph...")
    # Build 3D coordinates
    if three_dim_coor is None:
        # The first adata in adata_list is used as a reference for computing cutoff radius of spots
        adata_st_ref = adata_st_list[0].copy()
        loc_ref = np.array(adata_st_ref.obsm[coor_key])
        pair_dist_ref = pairwise_distances(loc_ref)
        min_dist_ref = np.sort(np.unique(pair_dist_ref), axis=None)[1]

        if rad_cutoff is None:
            # The radius is computed base on the attribute "adata.obsm['spatial']"
            rad_cutoff = min_dist_ref * rad_coef
        print("Radius for graph connection is %.4f." % rad_cutoff)

        # Use the attribute "adata.obsm['spatial_aligned']" to build a global graph
        if slice_dist_micron is None:
            pair_dist_z = pairwise_distances(np.array(adata_st.obsm[coor_key]))
            min_z = np.sort(np.unique(pair_dist_z), axis=None)[1]
            print(f"min_z: {min_z}")

            loc_xy = pd.DataFrame(adata_st.obsm[coor_key]).values
            loc_z = np.zeros(adata_st.shape[0]) + min_z * rad_coef
            loc = np.concatenate([loc_xy, loc_z.reshape(-1, 1)], axis=1)
        else:
            if len(slice_dist_micron) != (len(adata_st_list) - 1):
                raise ValueError("The length of 'slice_dist_micron' should be the number of adatas - 1 !")
            else:
                loc_xy = pd.DataFrame(adata_st.obsm[coor_key]).values
                loc_z = np.zeros(adata_st.shape[0])
                dim = 0
                for i in range(len(slice_dist_micron)):
                    dim += adata_st_list[i].shape[0]
                    loc_z[dim:] += slice_dist_micron[i] * (min_dist_ref / c2c_dist)
                loc = np.concatenate([loc_xy, loc_z.reshape(-1, 1)], axis=1)

    # If 3D coordinates already exists
    else:
        if rad_cutoff is None:
            raise ValueError("Please specify 'rad_cutoff' for finding 3D neighbors!")
        loc = three_dim_coor
    adata_st.obsm['loc'] = loc

    loc = pd.DataFrame(loc)
    loc.index = adata_st.obs.index
    loc.columns = ['x', 'y', 'z']

    if mode == 'KNN':
        nbrs = NearestNeighbors(n_neighbors=k_cutoff + 1).fit(loc)
        distances, indices = nbrs.kneighbors(loc)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    else:
        nbrs = NearestNeighbors(radius=rad_cutoff).fit(loc)
        distances, indices = nbrs.radius_neighbors(loc, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it] * indices[it].shape[0], indices[it], distances[it])))

    Spatial_Net = pd.concat(KNN_list)
    Spatial_Net.columns = ['Cell1', 'Cell2', 'Distance']
    Spatial_Net = Spatial_Net.loc[Spatial_Net['Distance'] > 0,]
    id_cell_trans = dict(zip(range(loc.shape[0]), np.array(loc.index), ))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(id_cell_trans)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(id_cell_trans)

    print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], adata_st.n_obs))
    print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / adata_st.n_obs))
    #
    cells = np.array(adata_st.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    Spatial_Net['Cell1'] = Spatial_Net['Cell1'].map(cells_id_tran)
    Spatial_Net['Cell2'] = Spatial_Net['Cell2'].map(cells_id_tran)
    adj_m1 = sp.coo_matrix((np.ones(Spatial_Net.shape[0]), (Spatial_Net['Cell1'], Spatial_Net['Cell2'])), shape=(adata_st.n_obs, adata_st.n_obs))
    # Store original adjacency matrix (without diagonal entries) for later
    adj_m1 = adj_m1 - sp.dia_matrix((adj_m1.diagonal()[np.newaxis, :], [0]), shape=adj_m1.shape)
    if not norm:
        return adata_st, sparse_mx_to_torch_sparse_tensor(adj_m1)
    return adata_st, preprocess_graph(adj_m1)


def graph_construction(adata, n=6, dmax=50, mode='KNN', spatial="spatial", norm=False):
    if mode == 'KNN':
        adj_m1 = generate_adj_mat(adata, include_self=False, n=n, spatial=spatial)
    else:
        adj_m1 = generate_adj_mat_1(adata, dmax, spatial=spatial)
    adj_m1 = sp.coo_matrix(adj_m1)

    # Store original adjacency matrix (without diagonal entries) for later
    adj_m1 = adj_m1 - sp.dia_matrix((adj_m1.diagonal()[np.newaxis, :], [0]), shape=adj_m1.shape)
    adj_m1.eliminate_zeros()
    if not norm:
        return sparse_mx_to_torch_sparse_tensor(adj_m1)
    return preprocess_graph(adj_m1)

##### generate n
def generate_adj_mat(adata, include_self=False, n=6, spatial="spatial"):
    from sklearn import metrics
    assert spatial in adata.obsm, 'AnnData object should provided spatial information'

    dist = metrics.pairwise_distances(adata.obsm[spatial])
    adj_mat = np.zeros((len(adata), len(adata)))
    for i in range(len(adata)):
        n_neighbors = np.argsort(dist[i, :])[:n+1]
        adj_mat[i, n_neighbors] = 1

    if not include_self:
        x, y = np.diag_indices_from(adj_mat)
        adj_mat[x, y] = 0

    adj_mat = adj_mat + adj_mat.T
    adj_mat = adj_mat > 0
    adj_mat = adj_mat.astype(np.int64)

    return adj_mat

def generate_adj_mat_1(adata, max_dist, spatial="spatial"):
    from sklearn import metrics
    assert spatial in adata.obsm, 'AnnData object should provided spatial information'

    dist = metrics.pairwise_distances(adata.obsm[spatial], metric='euclidean')
    adj_mat = dist < max_dist
    adj_mat = adj_mat.astype(np.int64)
    return adj_mat

##### normalze graph
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_graph(adj):
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)

