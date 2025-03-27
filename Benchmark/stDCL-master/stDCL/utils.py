import numpy as np
import pandas as pd
from sklearn import metrics
import scanpy as sc
import ot
from sklearn.decomposition import PCA
from sklearn.neighbors import kneighbors_graph


def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020): 
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def clustering(adata, args, start=0.1, end=2.0, increment=0.01, refinement=False):
    
    n_clusters = args.n_clusters
    radius = args.radius
    method = args.clustertype

    if args.datatype == '10x':

        pca = PCA(n_components=20, random_state=1) 
        embedding = pca.fit_transform(adata.obsm['emb'].copy())
        adata.obsm['emb_pca'] = embedding

        pca = PCA(n_components=20, random_state=1) 
        embedding = pca.fit_transform(adata.obsm['emb1'].copy())
        adata.obsm['emb_pca1'] = embedding

        
        if method == 'mclust':
            adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters)
            adata.obs['domain'] = adata.obs['mclust']
            adata = mclust_R(adata, used_obsm='emb_pca1', num_cluster=n_clusters)
            adata.obs['domain1'] = adata.obs['mclust']

        elif method == 'leiden':
            res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
            sc.tl.leiden(adata, random_state=0, resolution=res)
            adata.obs['domain'] = adata.obs['leiden']
            res = search_res(adata, n_clusters, use_rep='emb_pca1', method=method, start=start, end=end, increment=increment)
            sc.tl.leiden(adata, random_state=0, resolution=res)
            adata.obs['domain1'] = adata.obs['leiden']
        elif method == 'louvain':
            res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
            sc.tl.louvain(adata, random_state=0, resolution=res)
            adata.obs['domain'] = adata.obs['louvain'] 
            res = search_res(adata, n_clusters, use_rep='emb_pca1', method=method, start=start, end=end, increment=increment)
            sc.tl.louvain(adata, random_state=0, resolution=res)
            adata.obs['domain1'] = adata.obs['louvain'] 

        if refinement:  
            new_type = refine_label(adata, radius, key='domain')
            adata.obs['domain'] = new_type 
            new_type = refine_label(adata, radius, key='domain1')
            adata.obs['domain1'] = new_type 

        DB = np.round(metrics.davies_bouldin_score(adata.obsm['emb_pca'], adata.obs['domain']), 4)
        DB_L = np.round(metrics.davies_bouldin_score(adata.obsm['emb_pca1'], adata.obs['domain1']), 4)

        if (DB-DB_L > 1): 
            adata.obs['domain'] = adata.obs['domain1']
            adata.obsm['emb_pca'] = adata.obsm['emb_pca1']
            adata.obsm['imp'] = adata.obsm['imp1']

    else:
            pca = PCA(n_components=20, random_state=1)
            embedding = pca.fit_transform(adata.obsm['emb'].copy())
            adata.obsm['emb_pca'] = embedding

            
            if method == 'mclust':
                adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters)
                adata.obs['domain'] = adata.obs['mclust']

            elif method == 'leiden':
                res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
                sc.tl.leiden(adata, random_state=0, resolution=res)
                adata.obs['domain'] = adata.obs['leiden']
            elif method == 'louvain':
                res = search_res(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end, increment=increment)
                sc.tl.louvain(adata, random_state=0, resolution=res)
                adata.obs['domain'] = adata.obs['louvain'] 

            if refinement:  
                new_type = refine_label(adata, radius, key='domain')
                adata.obs['domain'] = new_type 
    

def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]    
    
    return new_type


def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return res    
