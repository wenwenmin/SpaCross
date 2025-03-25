import ot
import scipy.sparse
import matplotlib
import sklearn
from matplotlib import cm
import matplotlib.pyplot as plt
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors

np.random.seed(42)
random.seed(42)

def align_spots(adata_st_list_input,  # list of spatial transcriptomics datasets
                method="icp",  # "icp" or "paste"
                data_type="Visium",
                # a spot has six nearest neighborhoods if "Visium", four nearest neighborhoods otherwise
                coor_key="spatial",  # "spatial" for visium; key for the spatial coordinates used for alignment
                tol=0.01,  # parameter for "icp" method; tolerance level
                test_all_angles=False,  # parameter for "icp" method; whether to test multiple rotation angles or not
                plot=False,
                paste_alpha=0.1,
                paste_dissimilarity="kl"
                ):
    # Align coordinates of spatial transcriptomics

    # The first adata in the list is used as a reference for alignment
    adata_st_list = adata_st_list_input.copy()

    if plot:
        # Choose colors
        cmap = cm.get_cmap('rainbow', len(adata_st_list))
        colors_list = [matplotlib.colors.rgb2hex(cmap(i)) for i in range(len(adata_st_list))]

        # Plot spots before alignment
        plt.figure(figsize=(5, 5))
        plt.title("Before alignment")
        for i in range(len(adata_st_list)):
            plt.scatter(adata_st_list[i].obsm[coor_key][:, 0],
                        adata_st_list[i].obsm[coor_key][:, 1],
                        c=colors_list[i],
                        label="Slice %d spots" % i, s=5., alpha=0.5)
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.xticks([])
        plt.yticks([])
        plt.legend(loc=(1.02, .2), ncol=(len(adata_st_list) // 13 + 1))
        plt.show()

    if (method == "icp") or (method == "ICP"):
        print("Using the Iterative Closest Point algorithm for alignemnt.")
        # Detect edges
        print("Detecting edges...")
        point_cloud_list = []
        for adata in adata_st_list:
            # Use in-tissue spots only
            if 'in_tissue' in adata.obs.columns:
                adata = adata[adata.obs['in_tissue'] == 1]
            if data_type == "Visium":
                loc_x = adata.obs.loc[:, ["array_row"]]
                loc_x = np.array(loc_x) * np.sqrt(3)
                loc_y = adata.obs.loc[:, ["array_col"]]
                loc_y = np.array(loc_y)
                loc = np.concatenate((loc_x, loc_y), axis=1)
                pairwise_loc_distsq = np.sum((loc.reshape([1, -1, 2]) - loc.reshape([-1, 1, 2])) ** 2, axis=2)
                n_neighbors = np.sum(pairwise_loc_distsq < 5, axis=1) - 1
                edge = ((n_neighbors > 1) & (n_neighbors < 5)).astype(np.float32)
            if data_type == "ST":
                loc = adata.obsm[coor_key].copy()
                pairwise_loc_distsq = np.sum((loc.reshape([1, -1, 2]) - loc.reshape([-1, 1, 2])) ** 2, axis=2)
                min_distsq = np.sort(np.unique(pairwise_loc_distsq), axis=None)[1]
                n_neighbors = np.sum(pairwise_loc_distsq < (min_distsq * 3), axis=1) - 1
                edge = ((n_neighbors > 1) & (n_neighbors < 7)).astype(np.float32)
            else:
                loc_x = adata.obs.loc[:, ["array_row"]]
                loc_x = np.array(loc_x)
                loc_y = adata.obs.loc[:, ["array_col"]]
                loc_y = np.array(loc_y)
                loc = np.concatenate((loc_x, loc_y), axis=1)
                pairwise_loc_distsq = np.sum((loc.reshape([1, -1, 2]) - loc.reshape([-1, 1, 2])) ** 2, axis=2)
                min_distsq = np.sort(np.unique(pairwise_loc_distsq), axis=None)[1]
                n_neighbors = np.sum(pairwise_loc_distsq < (min_distsq * 3), axis=1) - 1
                edge = ((n_neighbors > 1) & (n_neighbors < 7)).astype(np.float32)
            point_cloud_list.append(adata.obsm[coor_key][edge == 1].copy())

        # Align edges
        print("Aligning edges...")
        trans_list = []
        adata_st_list[0].obsm["spatial_aligned"] = adata_st_list[0].obsm[coor_key].copy()
        # Calculate pairwise transformation matrices
        for i in range(len(adata_st_list) - 1):
            if test_all_angles == True:
                for angle in [0., np.pi * 1 / 3, np.pi * 2 / 3, np.pi, np.pi * 4 / 3, np.pi * 5 / 3]:
                    R = np.array([[np.cos(angle), np.sin(angle), 0],
                                  [-np.sin(angle), np.cos(angle), 0],
                                  [0, 0, 1]]).T
                    T, distances, _ = icp(transform(point_cloud_list[i + 1], R), point_cloud_list[i], tolerance=tol)
                    if angle == 0:
                        loss_best = np.mean(distances)
                        angle_best = angle
                        R_best = R
                        T_best = T
                    else:
                        if np.mean(distances) < loss_best:
                            loss_best = np.mean(distances)
                            angle_best = angle
                            R_best = R
                            T_best = T
                T = T_best @ R_best
            else:
                T, _, _ = icp(point_cloud_list[i + 1], point_cloud_list[i], tolerance=tol)
            trans_list.append(T)
        # Tranform
        for i in range(len(adata_st_list) - 1):
            point_cloud_align = adata_st_list[i + 1].obsm[coor_key].copy()
            for T in trans_list[:(i + 1)][::-1]:
                point_cloud_align = transform(point_cloud_align, T)
            adata_st_list[i + 1].obsm["spatial_aligned"] = point_cloud_align

    elif (method == "paste") or (method == "PASTE"):
        print("Using PASTE algorithm for alignemnt.")
        # Align spots
        print("Aligning spots...")
        pis = []
        # Calculate pairwise transformation matrices
        for i in range(len(adata_st_list) - 1):
            pi = pairwise_align_paste(adata_st_list[i], adata_st_list[i + 1], coor_key=coor_key,
                                      alpha=paste_alpha, dissimilarity=paste_dissimilarity)
            pis.append(pi)
        # Tranform
        S1, S2 = generalized_procrustes_analysis(adata_st_list[0].obsm[coor_key],
                                                 adata_st_list[1].obsm[coor_key],
                                                 pis[0])
        adata_st_list[0].obsm["spatial_aligned"] = S1
        adata_st_list[1].obsm["spatial_aligned"] = S2
        for i in range(1, len(adata_st_list) - 1):
            S1, S2 = generalized_procrustes_analysis(adata_st_list[i].obsm["spatial_aligned"],
                                                     adata_st_list[i + 1].obsm[coor_key],
                                                     pis[i])
            adata_st_list[i + 1].obsm["spatial_aligned"] = S2

    if plot:
        plt.figure(figsize=(5, 5))
        plt.title("After alignment")
        for i in range(len(adata_st_list)):
            plt.scatter(adata_st_list[i].obsm["spatial_aligned"][:, 0],
                        adata_st_list[i].obsm["spatial_aligned"][:, 1],
                        c=colors_list[i],
                        label="Slice %d spots" % i, s=5., alpha=0.5)
        ax = plt.gca()
        ax.set_ylim(ax.get_ylim()[::-1])
        plt.xticks([])
        plt.yticks([])
        plt.legend(loc=(1.02, .2), ncol=(len(adata_st_list) // 13 + 1))
        plt.show()

    return adata_st_list


# Functions for the Iterative Closest Point algorithm
# Credit to https://github.com/ClayFlannigan/icp
def best_fit_transform(A, B):
    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[m - 1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m + 1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m + 1, A.shape[0]))
    dst = np.ones((m + 1, B.shape[0]))
    src[:m, :] = np.copy(A.T)
    dst[:m, :] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m, :].T, dst[:m, :].T)

        # compute the transformation between the current source and nearest destination points
        T, _, _ = best_fit_transform(src[:m, :].T, dst[:m, indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T, _, _ = best_fit_transform(A, src[:m, :].T)

    return T, distances, i


def transform(point_cloud, T):
    point_cloud_align = np.ones((point_cloud.shape[0], 3))
    point_cloud_align[:, 0:2] = np.copy(point_cloud)
    point_cloud_align = np.dot(T, point_cloud_align.T).T
    return point_cloud_align[:, :2]


# Functions for the PASTE algorithm
# Credit to https://github.com/raphael-group/paste

## Covert a sparse matrix into a dense np array
to_dense_array = lambda X: X.toarray() if isinstance(X, scipy.sparse.csr.spmatrix) else np.array(X)

## Returns the data matrix or representation
extract_data_matrix = lambda adata, rep: adata.X if rep is None else adata.obsm[rep]


def intersect(lst1, lst2):
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3


def kl_divergence_backend(X, Y):
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."

    nx = ot.backend.get_backend(X, Y)

    X = X / nx.sum(X, axis=1, keepdims=True)
    Y = Y / nx.sum(Y, axis=1, keepdims=True)
    log_X = nx.log(X)
    log_Y = nx.log(Y)
    X_log_X = nx.einsum('ij,ij->i', X, log_X)
    X_log_X = nx.reshape(X_log_X, (1, X_log_X.shape[0]))
    D = X_log_X.T - nx.dot(X, log_Y.T)
    return nx.to_numpy(D)


def my_fused_gromov_wasserstein(M, C1, C2, p, q, G_init=None, loss_fun='square_loss', alpha=0.5, armijo=False,
                                log=False, numItermax=200, use_gpu=False, **kwargs):
    p, q = ot.utils.list_to_array(p, q)

    p0, q0, C10, C20, M0 = p, q, C1, C2, M
    nx = ot.backend.get_backend(p0, q0, C10, C20, M0)

    constC, hC1, hC2 = ot.gromov.init_matrix(C1, C2, p, q, loss_fun)

    if G_init is None:
        G0 = p[:, None] * q[None, :]
    else:
        G0 = (1 / nx.sum(G_init)) * G_init
        if use_gpu:
            G0 = G0.cuda()

    def f(G):
        return ot.gromov.gwloss(constC, hC1, hC2, G)

    def df(G):
        return ot.gromov.gwggrad(constC, hC1, hC2, G)

    if log:
        # res, log = ot.gromov.cg(p, q, (1 - alpha) * M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC,
        #                         log=True, **kwargs)
        res, log = ot.gromov.fused_gromov_wasserstein((1 - alpha) * M, p=p, q=q, alpha=alpha, G0=G0, armijo=armijo, C1=C1, C2=C2, constC=constC,
                                log=True, **kwargs)

        # ot.gromov.fused_gromov_wasserstein((1 - alpha) * M, C1, C2, p, q, alpha=alpha, G0=G0, armijo=armijo, constC=constC,
        #                                    log=True, **kwargs)

        fgw_dist = log['loss'][-1]

        log['fgw_dist'] = fgw_dist
        log['u'] = log['u']
        log['v'] = log['v']
        return res, log

    else:
        # return ot.gromov.cg(p, q, (1 - alpha) * M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC,
        #                     **kwargs)
        return ot.gromov.fused_gromov_wasserstein((1 - alpha) * M, p=p, q=q, alpha=alpha, G0=G0, armijo=armijo, C1=C1, C2=C2, constC=constC,
                                log=True, **kwargs)


def pairwise_align_paste(
        sliceA,
        sliceB,
        alpha=0.1,
        dissimilarity='kl',
        use_rep=None,
        G_init=None,
        a_distribution=None,
        b_distribution=None,
        norm=False,
        numItermax=200,
        backend=ot.backend.NumpyBackend(),
        use_gpu=False,
        return_obj=False,
        verbose=False,
        gpu_verbose=False,
        coor_key="spatial",
        **kwargs):
    if use_gpu:
        try:
            import torch
        except:
            print("We currently only have gpu support for Pytorch. Please install torch.")

        if isinstance(backend, ot.backend.TorchBackend):
            if torch.cuda.is_available():
                if gpu_verbose:
                    print("gpu is available, using gpu.")
            else:
                if gpu_verbose:
                    print("gpu is not available, resorting to torch cpu.")
                use_gpu = False
        else:
            print(
                "We currently only have gpu support for Pytorch, please set backend = ot.backend.TorchBackend(). Reverting to selected backend cpu.")
            use_gpu = False
    else:
        if gpu_verbose:
            print("Using selected backend cpu. If you want to use gpu, set use_gpu = True.")

    # subset for common genes
    common_genes = intersect(sliceA.var.index, sliceB.var.index)
    sliceA = sliceA[:, common_genes]
    sliceB = sliceB[:, common_genes]

    # Backend
    nx = backend

    # Calculate spatial distances
    coordinatesA = sliceA.obsm[coor_key].copy()
    coordinatesA = nx.from_numpy(coordinatesA)
    coordinatesB = sliceB.obsm[coor_key].copy()
    coordinatesB = nx.from_numpy(coordinatesB)

    if isinstance(nx, ot.backend.TorchBackend):
        coordinatesA = coordinatesA.float()
        coordinatesB = coordinatesB.float()
    D_A = ot.dist(coordinatesA, coordinatesA, metric='euclidean')
    D_B = ot.dist(coordinatesB, coordinatesB, metric='euclidean')

    if isinstance(nx, ot.backend.TorchBackend) and use_gpu:
        D_A = D_A.cuda()
        D_B = D_B.cuda()

    # Calculate expression dissimilarity
    A_X, B_X = nx.from_numpy(to_dense_array(extract_data_matrix(sliceA, use_rep))), nx.from_numpy(
        to_dense_array(extract_data_matrix(sliceB, use_rep)))

    if isinstance(nx, ot.backend.TorchBackend) and use_gpu:
        A_X = A_X.cuda()
        B_X = B_X.cuda()

    if dissimilarity.lower() == 'euclidean' or dissimilarity.lower() == 'euc':
        M = ot.dist(A_X, B_X)
    else:
        s_A = A_X + 0.01
        s_B = B_X + 0.01
        M = kl_divergence_backend(s_A, s_B)
        M = nx.from_numpy(M)

    if isinstance(nx, ot.backend.TorchBackend) and use_gpu:
        M = M.cuda()

    # init distributions
    if a_distribution is None:
        a = nx.ones((sliceA.shape[0],)) / sliceA.shape[0]
    else:
        a = nx.from_numpy(a_distribution)

    if b_distribution is None:
        b = nx.ones((sliceB.shape[0],)) / sliceB.shape[0]
    else:
        b = nx.from_numpy(b_distribution)

    if isinstance(nx, ot.backend.TorchBackend) and use_gpu:
        a = a.cuda()
        b = b.cuda()

    if norm:
        D_A /= nx.min(D_A[D_A > 0])
        D_B /= nx.min(D_B[D_B > 0])

    # Run OT
    if G_init is not None:
        G_init = nx.from_numpy(G_init)
        if isinstance(nx, ot.backend.TorchBackend):
            G_init = G_init.float()
            if use_gpu:
                G_init.cuda()
    pi, logw = my_fused_gromov_wasserstein(M, D_A, D_B, a, b, G_init=G_init, loss_fun='square_loss', alpha=alpha,
                                           log=True, numItermax=numItermax, verbose=verbose, use_gpu=use_gpu)


    pi = nx.to_numpy(pi)
    obj = nx.to_numpy(logw['fgw_dist'])
    if isinstance(backend, ot.backend.TorchBackend) and use_gpu:
        torch.cuda.empty_cache()

    if return_obj:
        return pi, obj
    return pi


def generalized_procrustes_analysis(X, Y, pi, output_params=False, matrix=False):
    """
    Finds and applies optimal rotation between spatial coordinates of two layers (may also do a reflection).
    Args:
        X: np array of spatial coordinates (ex: sliceA.obs['spatial'])
        Y: np array of spatial coordinates (ex: sliceB.obs['spatial'])
        pi: mapping between the two layers output by PASTE
        output_params: Boolean of whether to return rotation angle and translations along with spatial coordiantes.
        matrix: Boolean of whether to return the rotation as a matrix or an angle
    Returns:
        Aligned spatial coordinates of X, Y, rotation angle, translation of X, translation of Y
    """
    assert X.shape[1] == 2 and Y.shape[1] == 2

    tX = pi.sum(axis=1).dot(X)
    tY = pi.sum(axis=0).dot(Y)
    X = X - tX
    Y = Y - tY
    H = Y.T.dot(pi.T.dot(X))
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)
    Y = R.dot(Y.T).T
    if output_params and not matrix:
        M = np.array([[0, -1], [1, 0]])
        theta = np.arctan(np.trace(M.dot(H)) / np.trace(H))
        return X, Y, theta, tX, tY
    elif output_params and matrix:
        return X, Y, R, tX, tY
    else:
        return X, Y
