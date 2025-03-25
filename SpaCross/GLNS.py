import os
import random

os.environ['R_HOME'] = 'D:/software/R/R-4.3.2'
os.environ['R_USER'] = 'D:/software/anaconda/anaconda3/envs/pt20cu118/Lib/site-packages/rpy2'

import torch
import torch.nn as nn
import scipy.sparse as sp
from sklearn.cluster import KMeans
from collections import OrderedDict
import numpy as np
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from typing import Optional, Dict


class GLNSampler(nn.Module):
    def __init__(
            self,
            num_centroids: int,
            device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        super().__init__()
        self.device = device
        self.num_centroids = num_centroids
        self.num_clusterings = 5

    def __get_close_nei_in_back(self, indices, each_k_idx, cluster_labels, back_nei_idxs, k):
        def repeat_1d_tensor(t, num_reps):
            return t.unsqueeze(1).expand(-1, num_reps)
        # get which neighbors are close in the background set
        batch_labels = cluster_labels[each_k_idx][indices]
        top_cluster_labels = cluster_labels[each_k_idx][back_nei_idxs]
        batch_labels = repeat_1d_tensor(batch_labels, k)

        curr_close_nei = torch.eq(batch_labels, top_cluster_labels)
        return curr_close_nei

    def forward(self, adj, student, teacher, top_k, cluster_method="kmeans"):
        n_data, d = student.shape
        similarity = torch.matmul(student, torch.transpose(teacher, 1, 0).detach())
        similarity += torch.eye(n_data, device=self.device) * 10

        _, I_knn = similarity.topk(k=top_k, dim=1, largest=True, sorted=True)
        tmp = torch.LongTensor(np.arange(n_data)).unsqueeze(-1).to(self.device)

        knn_neighbor = self.create_sparse(I_knn)
        locality = knn_neighbor * adj

        ncentroids = self.num_centroids

        pred_labels = []

        # Perform clustering based on the chosen method
        if cluster_method == "mclust":
            for seed in range(self.num_clusterings):
                clust_labels = self.__mclust_R__(teacher.cpu().numpy(), num_cluster=ncentroids, random_seed=seed + 3407)
                pred_labels.append(clust_labels)

            pred_labels = np.stack(pred_labels, axis=0)
            cluster_labels = torch.from_numpy(pred_labels).long().to(self.device)

        else:
            for seed in range(self.num_clusterings):
                kmeans = KMeans(n_clusters=ncentroids, random_state=seed + 3407)
                kmeans.fit(teacher.cpu().numpy())  # Fit on CPU, but consider moving to GPU if necessary
                clust_labels = kmeans.labels_
                pred_labels.append(clust_labels)

            pred_labels = np.stack(pred_labels, axis=0)
            cluster_labels = torch.from_numpy(pred_labels).long().to(self.device)

        all_close_nei_in_back = None
        with torch.no_grad():
            for each_k_idx in range(self.num_clusterings):
                curr_close_nei = self.__get_close_nei_in_back(tmp.squeeze(-1), each_k_idx, cluster_labels, I_knn,
                                                              I_knn.shape[1])

                if all_close_nei_in_back is None:
                    all_close_nei_in_back = curr_close_nei
                else:
                    all_close_nei_in_back = all_close_nei_in_back | curr_close_nei

        all_close_nei_in_back = all_close_nei_in_back.to(self.device)

        globality = self.create_sparse_revised(I_knn, all_close_nei_in_back)

        pos_ = locality + globality
        ind = pos_.coalesce()._indices()
        # nonzero_indices = pos_.coalesce().values().nonzero().squeeze()
        # ind = pos_.coalesce()._indices()[:, nonzero_indices]

        anchor = ind[0]
        positive = ind[1]
        negative = torch.tensor(random.choices(list(range(n_data)), k=len(anchor))).to(self.device)
        return anchor, positive, negative

    def create_sparse(self, I):

        similar = I.reshape(-1).tolist()
        index = np.repeat(range(I.shape[0]), I.shape[1])

        assert len(similar) == len(index)
        indices = torch.tensor([index, similar]).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones_like(I.reshape(-1)), [I.shape[0], I.shape[0]])

        return result

    def create_sparse_revised(self, I, all_close_nei_in_back):
        n_data, k = I.shape[0], I.shape[1]

        index = []
        similar = []
        for j in range(I.shape[0]):
            for i in range(k):
                index.append(int(j))
                similar.append(I[j][i].item())

        index = torch.masked_select(torch.LongTensor(index).to(self.device), all_close_nei_in_back.reshape(-1))
        similar = torch.masked_select(torch.LongTensor(similar).to(self.device), all_close_nei_in_back.reshape(-1))

        assert len(similar) == len(index)
        indices = torch.tensor([index.cpu().numpy().tolist(), similar.cpu().numpy().tolist()]).to(self.device)
        result = torch.sparse_coo_tensor(indices, torch.ones(len(index)).to(self.device), [n_data, n_data])

        return result

    # Function for clustering using mclust
    def __mclust_R__(self, enc_rep, num_cluster, modelNames='EEE', random_seed=3407):
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


def Integrated_3D_graph(
    batch_label: torch.Tensor,
    joint_mat: Optional[torch.Tensor] = None,
    intra_neighbors: int = 20,
    intra_metric: str = "similarity",
    inter_neighbors: int = 10,
    inter_metric: str = "similarity",
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> None:
    """
    Integrate 3D graph processing based on input batch labels and adjacency matrices.

    Parameters:
        batch_label (torch.Tensor): A tensor containing batch labels for the data.
        joint_mat (torch.Tensor, optional): A tensor representing the joint adjacency matrix. Defaults to None.
        spatial_mat (torch.Tensor, optional): A tensor representing the spatial adjacency matrix. Defaults to None.
        intra_neighbors (int, optional): Number of neighbors to consider for intra-cluster connections. Defaults to 30.
        intra_metric (str, optional): Metric to use for intra-cluster distance calculations. Defaults to "euclidean".
        inter_neighbors (int, optional): Number of neighbors to consider for inter-cluster connections. Defaults to 30.
        inter_metric (str, optional): Metric to use for inter-cluster distance calculations. Defaults to "euclidean".
        device (torch.device, optional): Device on which to perform computations (default: GPU if available).

    Returns:
        None: This function does not return any value. It performs computations in place or modifies external states.

    Raises:
        ValueError: If any of the input tensors are of incompatible shapes or types.
    """

    def find_intra_neighbor(matrix, k, metric='cosine'):
        """
        Find the intra-neighbors within the same group.

        Parameters:
            matrix (torch.Tensor): Feature matrix (on CUDA).
            k (int): Number of neighbors to find.
            metric (str): Distance metric to use ('cosine', '', or 'similarity').

        Returns:
            torch.Tensor: Indices of the nearest neighbors.
        """
        num_samples = matrix.shape[0]
        k = min(num_samples - 1, k)  # Ensure k is not larger than the number of available neighbors

        if k > 0:
            if metric == "cosine":
                # Normalize matrix for cosine similarity
                matrix_normalized = torch.nn.functional.normalize(matrix, p=2, dim=1)
                similarity = torch.matmul(matrix_normalized, matrix_normalized.T)
                cosine_dist = 1 - similarity  # Convert similarity to distance

                # Find the k nearest neighbors based on cosine distance
                _, indices = torch.topk(-cosine_dist, k + 1, dim=1)  # Negate to sort by smallest values
                return indices[:, 1:k + 1]  # Exclude the first index (self-neighbor)

            elif metric == "similarity":
                # Directly compute similarity (dot product)
                similarity = torch.matmul(matrix, matrix.T)
                _, indices = torch.topk(similarity, k + 1, dim=1)
                return indices[:, 1:k + 1]  # Exclude the first index (self-neighbor)

            else:
                # Use pairwise Euclidean or Manhattan distance
                dist = torch.cdist(matrix, matrix, p=2 if metric == "euclidean" else 1)
                _, indices = torch.topk(-dist, k + 1, dim=1)  # Negate to get smallest distances
                return indices[:, 1:k + 1]  # Exclude the first index (self-neighbor)
        else:
            return None

    def find_inter_neighbor(matrix1, matrix2, k, metric='cosine'):
        """
        Find the inter-neighbors between different groups.

        Parameters:
            matrix1 (torch.Tensor): Feature matrix of the current group (on CUDA).
            matrix2 (torch.Tensor): Feature matrix of other groups (on CUDA).
            k (int): Number of neighbors to find.
            metric (str): Distance metric to use ('cosine', 'euclidean', or 'similarity').

        Returns:
            torch.Tensor: Indices of the nearest neighbors.
        """
        k = min(matrix2.shape[0], k)  # Ensure k is not larger than the number of available neighbors

        if k > 0:
            if metric == "cosine":
                # Normalize both matrices for cosine similarity
                matrix1_normalized = torch.nn.functional.normalize(matrix1, p=2, dim=1)
                matrix2_normalized = torch.nn.functional.normalize(matrix2, p=2, dim=1)

                # Compute cosine similarity as dot product of normalized matrices
                similarity = torch.matmul(matrix1_normalized, matrix2_normalized.T)

                # Convert cosine similarity to cosine distance
                cosine_dist = 1 - similarity

                # Find the k nearest neighbors based on cosine distance
                _, indices = torch.topk(-cosine_dist, k, dim=1)  # Negate to get smallest distances
                return indices

            elif metric == "similarity":
                # Compute similarity (dot product) and find k nearest neighbors
                similarity = torch.matmul(matrix1, matrix2.T)
                _, indices = torch.topk(similarity, k, dim=1)
                return indices

            else:
                # Use pairwise Euclidean or Manhattan distance
                dist = torch.cdist(matrix1, matrix2, p=2 if metric == "euclidean" else 1)
                _, indices = torch.topk(-dist, k, dim=1)  # Negate to get smallest distances
                return indices
        else:
            return None

    # Initialize sparse matrix components
    row_index = []
    col_index = []
    data = []

    N = joint_mat.shape[0]
    if not isinstance(batch_label, np.ndarray):
        batch_label = np.array(batch_label)
    samples_indices = np.arange(N)

    # joint_mat_cuda = torch.tensor(joint_mat, device=device) if joint_mat is not None else None

    for cur in np.unique(batch_label):
        # Get indices for the current group
        cur_idx = batch_label == cur
        cur_original_idx = samples_indices[cur_idx]

        cur_joint_mat = joint_mat[cur_idx]

        # Find intra-group neighbors
        indices = find_intra_neighbor(cur_joint_mat, k=intra_neighbors, metric=intra_metric)
        intra_indices = cur_original_idx[indices.cpu().numpy()] if indices is not None else None

        # Find inter-group neighbors
        other_idx = batch_label != cur
        other_original_idx = samples_indices[other_idx]

        inter_indices = []
        cur_joint_mat = joint_mat[cur_idx]
        other_joint_mat = joint_mat[other_idx]
        indices = find_inter_neighbor(cur_joint_mat, other_joint_mat, k=inter_neighbors, metric=inter_metric)
        if indices is not None:
            inter_indices.append(other_original_idx[indices.cpu().numpy()])

        inter_indices = np.concatenate(inter_indices, axis=1) if len(inter_indices) > 0 else None

        # Populate sparse matrix with intra and inter indices
        for i in range(len(cur_original_idx)):
            if intra_indices is not None:
                cur_list = intra_indices[i].tolist()
                row_index += cur_list
                col_index += [cur_original_idx[i]] * len(cur_list)
                data += [1] * len(cur_list)
            if inter_indices is not None:
                cur_list = list(OrderedDict.fromkeys(inter_indices[i].tolist()))
                row_index += cur_list
                col_index += [cur_original_idx[i]] * len(cur_list)
                data += [1] * len(cur_list)

    # Create adjacency matrix and return as sparse COO matrix
    G_adj = sp.coo_matrix((data, (row_index, col_index)), shape=(N, N)).toarray()
    row_index, col_index = np.nonzero(G_adj)
    data = G_adj[row_index, col_index]
    return sp.coo_matrix((data, (row_index, col_index)), shape=(N, N))


class GLNSampler_BC(torch.nn.Module):
    def __init__(self, num_centroids, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.device = device
        self.num_centroids = num_centroids
        self.num_clusterings = 5

    def __get_close_nei_in_back(self, indices, each_k_idx, cluster_labels, back_nei_idxs, k, bool_mask):
        def repeat_1d_tensor(t, num_reps):
            return t.unsqueeze(1).expand(-1, num_reps)

        batch_labels = cluster_labels[each_k_idx][indices]
        top_cluster_labels = cluster_labels[each_k_idx][back_nei_idxs]
        batch_labels = repeat_1d_tensor(batch_labels, k)

        curr_close_nei = torch.eq(batch_labels, top_cluster_labels)
        curr_close_nei = curr_close_nei | bool_mask
        return curr_close_nei

    def __I_KNN_Padded(self, adj_knn):
        adj_knn = torch.tensor(adj_knn.toarray(), device=self.device)  # Directly move to device
        I_knn = torch.nonzero(adj_knn, as_tuple=False)
        neighbor_counts = adj_knn.sum(dim=1).long()
        k_max = neighbor_counts.max().item()
        I_knn_padded = torch.full((adj_knn.size(0), k_max), -1, dtype=torch.long, device=self.device)
        start_indices = torch.cat([torch.arange(count, device=self.device) for count in neighbor_counts])
        row_indices = I_knn[:, 0]
        col_indices = start_indices
        I_knn_padded[row_indices, col_indices] = I_knn[:, 1]
        return I_knn_padded

    def forward(self, adj, enc_rep, batch_id, top_k, top_k_inter, cluster_method="kmeans"):
        n_data, d = enc_rep.shape
        # Compute the integrated 3D graph adjacency matrix
        rep_adj = Integrated_3D_graph(batch_label=batch_id, joint_mat=enc_rep,
                                      intra_neighbors=top_k, inter_neighbors=top_k_inter,
                                      device=self.device)
        rep_adj_transposed = rep_adj.transpose()
        common_adj = rep_adj.multiply(rep_adj_transposed > 0)
        common_adj.data = np.ones_like(common_adj.data)
        common_adj.setdiag(0)
        common_adj.eliminate_zeros()

        # Compute locality adjacency matrix
        knn_neighbor = self.create_locality_sparse(common_adj)
        locality = knn_neighbor * adj

        I_knn_padded = self.__I_KNN_Padded(common_adj)
        bool_mask = I_knn_padded != -1
        tmp = torch.arange(n_data, device=self.device).unsqueeze(-1)

        pred_labels = []

        # Perform clustering based on the chosen method
        if cluster_method == "mclust":
            for seed in range(self.num_clusterings):
                clust_labels = self.__mclust_R__(enc_rep.cpu().numpy(), num_cluster=self.num_centroids, random_seed=seed + 3407)
                pred_labels.append(clust_labels)

            pred_labels = np.stack(pred_labels, axis=0)
            cluster_labels = torch.from_numpy(pred_labels).long().to(self.device)

        elif cluster_method == "kmeans":
            for seed in range(self.num_clusterings):
                kmeans = KMeans(n_clusters=self.num_centroids, random_state=seed + 3407)
                kmeans.fit(enc_rep.cpu().numpy())  # Fit on CPU, but consider moving to GPU if necessary
                clust_labels = kmeans.labels_
                pred_labels.append(clust_labels)

            pred_labels = np.stack(pred_labels, axis=0)
            cluster_labels = torch.from_numpy(pred_labels).long().to(self.device)

        all_close_nei_in_back = None
        with torch.no_grad():
            for each_k_idx in range(self.num_clusterings):
                curr_close_nei = self.__get_close_nei_in_back(tmp.squeeze(-1), each_k_idx, cluster_labels, I_knn_padded,
                                                              I_knn_padded.shape[1], bool_mask)

                if all_close_nei_in_back is None:
                    all_close_nei_in_back = curr_close_nei
                else:
                    all_close_nei_in_back = all_close_nei_in_back | curr_close_nei

        all_close_nei_in_back = all_close_nei_in_back.to(self.device)

        # Compute globality based on the close neighbors
        globality = self.create_globality_sparse(I_knn_padded, all_close_nei_in_back, bool_mask)

        # Combine locality and globality to form the final adjacency matrix
        pos_ = locality + globality
        nonzero_indices = pos_.coalesce().values().nonzero().squeeze()
        ind = pos_.coalesce()._indices()[:, nonzero_indices]

        anchor = ind[0]
        positive = ind[1]
        negative = torch.tensor(random.choices(list(range(n_data)), k=len(anchor))).to(self.device)
        return anchor, positive, negative

    def create_locality_sparse(self, locality_adj):
        if isinstance(locality_adj, sp.coo_matrix):
            locality_adj = locality_adj.tocsr()

        rows, cols = locality_adj.nonzero()
        data = locality_adj.data

        # 创建稀疏张量
        locality_adj = torch.sparse_coo_tensor(
            torch.tensor(np.vstack([rows, cols]), device=self.device),
            torch.tensor(data, dtype=torch.float32, device=self.device),
            size=locality_adj.shape,
            device=self.device
        )
        return locality_adj

    def create_globality_sparse(self, I_knn_padded, all_close_nei_in_back, bool_mask):
        device = self.device
        anchor = torch.arange(I_knn_padded.shape[0], device=device).unsqueeze(-1).expand_as(I_knn_padded)
        positive = I_knn_padded.clone()

        mask = bool_mask & all_close_nei_in_back
        anchor = anchor[mask]
        positive = positive[mask]

        globality = torch.sparse_coo_tensor(
            torch.stack([anchor, positive], dim=0),
            torch.ones_like(anchor, dtype=torch.float32, device=device),
            size=(I_knn_padded.size(0), I_knn_padded.size(0)),
            device=device
        )
        return globality

    # Function for clustering using mclust
    def __mclust_R__(self, enc_rep, num_cluster, modelNames='EEE', random_seed=3407):
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