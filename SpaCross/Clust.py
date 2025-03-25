import os

import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from sklearn.cluster import KMeans

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['R_HOME'] = 'D:/software/R/R-4.3.2'
os.environ['R_USER'] = 'D:/software/anaconda/anaconda3/envs/pt20cu118/Lib/site-packages/rpy2'

# Function for clustering using mclust
def __mclust_R__(enc_rep, num_cluster, modelNames='EEE', random_seed=3407):
    """
    Clustering using the mclust algorithm.
    """
    # Activate the numpy to R interface
    rpy2.robjects.numpy2ri.activate()
    # Load the R mclust package
    robjects.r.library("mclust")

    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    res = rmclust(enc_rep, num_cluster, modelNames)
    mclust_res = np.array(res[-2])  # Extract the clustering labels
    return mclust_res


def clustering(z, n_clust, method="kmeans", num_seed=5):
    if num_seed > 1:
        pred_labels = []
        if method == "mclust":
            for seed in range(num_seed):
                clust_labels = __mclust_R__(z, num_cluster=n_clust, random_seed=seed + 3407)
                pred_labels.append(clust_labels)
        else:
            for seed in range(num_seed):
                kmeans = KMeans(n_clusters=n_clust, random_state=seed + 3407)
                kmeans.fit(z)  # Fit on CPU, but consider moving to GPU if necessary
                clust_labels = kmeans.labels_
                pred_labels.append(clust_labels)
        pred_labels = np.stack(pred_labels, axis=0)
        from sklearn.cluster import SpectralClustering

        consensus_matrix = np.mean(pred_labels.T[:, np.newaxis] == pred_labels.T, axis=2)
        clustering = SpectralClustering(n_clusters=n_clust, affinity='precomputed', random_state=42)
        final_labels = clustering.fit_predict(consensus_matrix)
        return final_labels
    else:
        if method == "mclust":
            clust_labels = __mclust_R__(z, num_cluster=n_clust, random_seed=3407)
        else:
            kmeans = KMeans(n_clusters=n_clust, max_iter=100, random_state=3407)
            kmeans.fit(z)  # Fit on CPU, but consider moving to GPU if necessary
            clust_labels = kmeans.labels_

        # %%
        final_val = np.array(clust_labels)
        final_val = [str(val) for val in np.int32(final_val)]
        return final_val


import ot
def refine_label(adata, radius=30, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values

    # calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')

    n_cell = distance.shape[0]

    for i in range(n_cell):
        vec = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh + 1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)

    new_type = [str(i) for i in list(new_type)]
    # adata.obs['label_refined'] = np.array(new_type)
    return new_type
