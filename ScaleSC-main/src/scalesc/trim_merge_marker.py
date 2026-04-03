import os
from typing import TYPE_CHECKING, Union
import logging
import time
import json
import warnings
import scanpy as sc
import numpy as np
import cupy as cp
from cupy import _core
from cupy.cuda import device
from cupy.cuda import runtime
import pandas as pd
from cuml.ensemble import RandomForestClassifier as cumlRandomForestClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import fbeta_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.sparsefuncs import csc_median_axis_0
from scipy import sparse as sparse_cpu
from cupyx.scipy import sparse as sparse_gpu

from scalesc.kernels import *
from scalesc import util

warnings.filterwarnings("ignore")

logger = logging.getLogger('scaleSC')
GPU_ARRAY_TYPE = Union[cp.ndarray, sparse_gpu.csr_matrix, sparse_gpu.csc_matrix]
CPU_ARRAY_TYPE = Union[np.ndarray, sparse_cpu.csr_matrix, sparse_cpu.csc_matrix]

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        logger.debug(f'Time - {func.__name__}: {end-start}s')
        return res 
    return wrapper


# @timer
# def calculate_median_sparse(adata, cluster_header):
#     # copy data to gpu 
#     x = X_to_GPU(adata.X)
#     clusters = adata.obs[cluster_header].unique().tolist()
#     df_median = pd.DataFrame(0, index=clusters, columns=adata.var_names.copy())
#     for cluster in clusters:
#         x_tmp = x[adata.obs[cluster_header]==cluster, :]
#         gene_exp_count = x_tmp.sign().sum(axis=0)

#         gids = (gene_exp_count > x_tmp.shape[0]*0.5).tolist()[0]
#         genes_median_positive = adata.var_names[gids]
#         x_tmp = x_tmp[:, gids].toarray().copy()
#         genes_median = np.median(x_tmp, axis=0)

#         cols = df_median.columns[gids]

#         df_median.loc[cluster, cols] = genes_median
    
#     df_median.sort_index(inplace=True)
#     return df_median


# @timer
# def calculate_median_sparse(adata, cluster_header):
#     clusters = adata.obs[cluster_header].unique().tolist()
#     df_median = pd.DataFrame(0, index=clusters, columns=adata.var_names.copy())
#     for cluster in clusters:
#         x_tmp = adata.X[adata.obs[cluster_header]==cluster, :].copy()
#         gene_exp_count = x_tmp.sign().sum(axis=0)
#         gids = (gene_exp_count >= x_tmp.shape[0]*0.5).tolist()[0]
#         genes_median_positive = adata.var_names[gids]
#         x_tmp = x_tmp[:, gids].toarray()
#         genes_median = np.median(x_tmp, axis=0)
#         # print('in median', genes_median, (genes_median==0).all())
#         cols = df_median.columns[gids]
#         df_median.loc[cluster, cols] = genes_median
#     df_median.sort_index(inplace=True)
#     return df_median

# @timer
# def calculate_median_sparse(adata, cluster_header):
#     clusters = adata.obs[cluster_header].unique().tolist()
#     df_median = pd.DataFrame(0, index=clusters, columns=adata.var_names.copy())
#     for cluster in clusters:
#         x_tmp = adata.X[adata.obs[cluster_header]==cluster]
#         s = time.time()
#         gene_exp_count = x_tmp.sign().sum(axis=0)
#         gids = (gene_exp_count >= x_tmp.shape[0]*0.5).tolist()[0]
#         genes_median_positive = adata.var_names[gids]
#         x_tmp = x_tmp[:, gids].toarray()
#         print('opt time', time.time() - s, 'ndarray', x_tmp.shape)
#         s = time.time()
#         genes_median = np.median(x_tmp, axis=0)
#         print('median time', time.time()-s)
#         # print('in median', genes_median, (genes_median==0).all())
#         cols = df_median.columns[gids]
#         df_median.loc[cluster, cols] = genes_median
#     df_median.sort_index(inplace=True)
#     return df_median


@timer
def calculate_median_sparse_csc_median(adata, cluster_header):
    clusters = adata.obs[cluster_header].unique().tolist()
    df_median = pd.DataFrame(0.0, index=clusters, columns=adata.var_names.copy())
    # transfer to gpu 
    for cluster in clusters:
        x_tmp = adata.X[adata.obs[cluster_header]==cluster]
        genes_median = csc_median_axis_0(x_tmp.tocsc())
        df_median.loc[cluster] = genes_median
    df_median.sort_index(inplace=True)
    return df_median



# @timer
# def calculate_median_sparse(adata, cluster_header):
#     clusters = adata.obs[cluster_header].unique().tolist()
#     df_median = pd.DataFrame(0.0, index=clusters, columns=adata.var_names.copy())
#     if sparse_cpu.issparse(adata.X):
#         # x_cuda = X_to_GPU(adata.X)
#         x_cuda = adata.X
#     else:   
#         raise Exception("adata.X should be a sparse matrix to save device memory")
#     # illegal memeory access is due to the size of sparse X limited by int32
#     for cluster in clusters:
#         x_tmp = x_cuda[adata.obs[cluster_header]==cluster]
#         x_tmp = X_to_GPU(x_tmp)
#         # gene_exp_count = x_tmp.sign().sum(axis=0)
#         gene_exp_count = cp.zeros(x_tmp.shape[1], dtype=cp.float32)
#         sum_sign_elementwise_kernel(x_tmp.data, x_tmp.indices, gene_exp_count)
#         gids = (gene_exp_count >= (x_tmp.shape[0]*0.5)).get()
#         x_tmp = x_tmp[:, gids].get()
#         genes_median = csc_median_axis_0(x_tmp.tocsc())
#         cols = df_median.columns[gids]
#         df_median.loc[cluster, cols] = genes_median
#     df_median.sort_index(inplace=True)
#     del(x_cuda)
#     return df_median


@timer
def calculate_median_sparse_fast_gpu(adata, cluster_header):
    clusters = adata.obs[cluster_header].unique().tolist()
    df_median = pd.DataFrame(0.0, index=clusters, columns=adata.var_names.copy())
    # transfer to gpu 
    data = cp.asarray(adata.X.data, dtype=np.float32)
    indptr = cp.asarray(adata.X.indptr, dtype=np.int64)
    indices = cp.asarray(adata.X.indices, dtype=np.int32)
    # illegal memeory access is due to the size of sparse X limited by int32
    for cluster in clusters:
        idx = cp.asarray(np.argwhere(adata.obs[cluster_header] == cluster)).reshape(-1).astype(np.int64)
        Bx, Bj, Bp, Bi = util.csr_row_index(data, indices, indptr, idx)
        gene_exp_count = cp.zeros(adata.shape[1], dtype=cp.float32)
        sum_sign_elementwise_kernel(Bx, Bj, gene_exp_count)
        gind = gene_exp_count >= (idx.shape[0]*0.5)
        gids = cp.where(gind)[0].astype(cp.int32)
        csc = util.csr_col_index(Bx, Bj, Bi, gids, (idx.shape[0], adata.shape[1])).tocsc()
        genes_median = csc_median_axis_0(csc)
        df_median.loc[cluster] = genes_median
    df_median.sort_index(inplace=True)
    return df_median


# def get_safe_batch_size(adata):
#     n, m = adata.shape
#     return int((2**31-1) / n * 5)

# @timer
# def calculate_median_sparse_in_batch(adata, cluster_header):
#     batch_size = get_safe_batch_size(adata)
#     clusters = adata.obs[cluster_header].unique().tolist()
#     df_median = pd.DataFrame(0.0, index=clusters, columns=adata.var_names.copy())
#     x = adata.X
#     x_batches = []
#     for i in range(0, x.shape[1], batch_size):
#         x_batches.append(x[:, i: i+batch_size])
#     # illegal memeory access is due to the size of sparse X limited by int32
#     for cluster in clusters:
#         start_col = 0
#         for x_batch in x_batches:
#             x_tmp = X_to_GPU(x_batch)
#             x_tmp = x_tmp[adata.obs[cluster_header]==cluster]
#             # gene_exp_count = x_tmp.sign().sum(axis=0)
#             gene_exp_count = cp.zeros(x_tmp.shape[1], dtype=cp.float32)
#             sum_sign_elementwise_kernel(x_tmp.data, x_tmp.indices, gene_exp_count)
#             gids = (gene_exp_count >= (x_tmp.shape[0]*0.5)).get()
#             x_tmp = x_tmp.tocsc()[:, gids].get()
#             genes_median = csc_median_axis_0(x_tmp)
#             cols = df_median.columns[start_col: start_col+x_batch.shape[1]][gids]
#             df_median.loc[cluster, cols] = genes_median
#             start_col += x_batch.shape[1]
#     df_median.sort_index(inplace=True)
#     del(x)
#     return df_median


def X_to_GPU(X):
    """ Transfers matrices and arrays to the GPU.
    Args:
        X: Matrix or array to transfer to the GPU.
    """
    if isinstance(X, GPU_ARRAY_TYPE):
        pass
    elif sparse_cpu.isspmatrix_csr(X):
        X = sparse_gpu.csr_matrix(X)
    elif sparse_cpu.isspmatrix_csc(X):
        X = sparse_gpu.csc_matrix(X)
    elif isinstance(X, np.ndarray):
        X = cp.array(X)
    return X

# =======================================================================================================================


class UF:
    def __init__(self,n):
        self.p = [i for i in range(n)]
    def union(self,x,y):
        self.p[self.find(y)] = self.find(x)
    def find(self,x):
        if x != self.p[x]: self.p[x] = self.find(self.p[x])
        return self.p[x]
    def final(self):
        for i in range(len(self.p)):
            self.p[i] = self.find(i)
    def current_kids_dict(self):
        pset = set(self.p)
        self.kids_dict = {p:[] for p in pset}
        for i in range(len(self.p)):
            self.kids_dict[self.p[i]].append(i) 
            
class data2UF:

    def __init__(self, celltypes:list, merge_pairs:list[tuple]):
        self.ct = celltypes
        self.num = [i for i in range(len(celltypes))]
        self.ct2num = {self.ct[i]:i for i in range(len(celltypes))}
        self.num2ct = {i:self.ct[i] for i in range(len(celltypes))}
        self.pairs = merge_pairs
    
    def union_pairs(self) -> int:
        myuf = UF(len(self.ct))
        for pair in self.pairs:
            myuf.union(self.ct2num[pair[0]], self.ct2num[pair[1]])
        myuf.final()
        dict_merge = {self.num2ct[i]:self.num2ct[myuf.p[i]] for i in range(len(self.ct))}
        return dict_merge
    
    
def marker_filter_sort(markers, cluster, df_sp, df_frac):
    genes = markers.copy()
    genes = [_ for _ in genes if df_sp[cluster][_] >= df_sp['second'][_]]
    genes = [_ for _ in genes if df_frac[cluster][_] >= 0.4]
    genes.sort(key=lambda _: -df_sp[cluster][_] * df_frac[cluster][_])
    return genes    

def find_markers(adata, subctype_col, n_trees=100, n_top_genes=100, n_binary_genes=30, n_genes_eval=15):
    # 1. mynsforest -> marker genes, acc
    # 2. add S,F, then rank the markers
    df_nsf = myNSForest(adata, cluster_header=subctype_col, n_trees=n_trees, n_top_genes=n_top_genes, n_binary_genes=n_binary_genes, n_genes_eval=n_genes_eval)
    dict_acc = {df_nsf.iloc[i]['clusterName']:df_nsf.iloc[i]['f_score'] for i in range(df_nsf.shape[0])}
    dict_markers = {df_nsf.iloc[i]['clusterName']:df_nsf.iloc[i]['binary_genes'] for i in range(df_nsf.shape[0])}
    markers_all = []
    for key in dict_markers.keys():
        markers_all.extend(dict_markers[key])
    markers_all = list(set(markers_all))
    
    df_sp = specificity_score(adata=adata, ctype_col=subctype_col, glist=markers_all)
    df_sp['second'] = np.sort(df_sp.iloc[:,], axis=1)[:, 2]
    df_frac = fraction_cells(adata=adata, ctype_col=subctype_col, glist=markers_all)
    
    dict_markers = {cluster:marker_filter_sort(markers, cluster, df_sp, df_frac) for cluster,markers in dict_markers.items()}
    return dict_acc, dict_markers

# def stds(a, axis=None):
#     """ Variance of sparse matrix a
#     var = mean(a**2) - mean(a)**2
    
#     Standard deviation of sparse matrix a
#     std = sqrt(var(a))
#     """
#     a_squared = a.copy()
#     a_squared.data **= 2
#     a_vars = a_squared.mean(axis) - np.square(a.mean(axis))
#     return np.sqrt(a_vars)

def stds(x, axis=None):
    """ Variance of sparse matrix a
    var = mean(a**2) - mean(a)**2
    
    Standard deviation of sparse matrix a
    std = sqrt(var(a))
    """
    if sparse_gpu.issparse(x):
        x_squared = x.power(2)
        x_vars = x_squared.mean(axis) - cp.sqaure(x.mean(axis))
        res = cp.sqrt(x_vars)
    elif isinstance(x, cp.ndarray):
        x_squared = x ** 2
        x_vars = x_squared.mean(axis) - x.mean(axis) ** 2
        res = cp.sqrt(x_vars)
    elif isinstance(x, CPU_ARRAY_TYPE):
        return stds(x)
    return res


def find_cluster_pairs_to_merge(adata, x, colname, cluster, markers, threshold=0.0):
    clusters = adata.obs[colname].unique().tolist()
    merge_pairs = []
    if len(markers)==0:
        merge_pairs = [(cluster, cluster2) for cluster2 in clusters if cluster2!=cluster]
        return merge_pairs
    for cluster2 in clusters:
        if cluster2==cluster: continue
        merge = True
        # try percentage cutoff
        m_count = 0
        for gene in markers:
            x_gene = x[adata.obs[colname]==cluster, adata.var_names==gene]
            # merge_cutoff = np.mean(adata[adata.obs[colname]==cluster, gene].X) - stds(adata[adata.obs[colname]==cluster, gene].X)
            merge_cutoff = cp.mean(x_gene) - stds(x_gene)
            # if np.mean(adata[adata.obs[colname]==cluster2, gene].X) < merge_cutoff:
            if cp.mean(x[adata.obs[colname]==cluster2, adata.var_names==gene]) < merge_cutoff:
                m_count += 1
        if m_count / len(markers) <= threshold:
            merge_pairs.append((cluster, cluster2))
        # if merge: merge_pairs.append((cluster, cluster2))
    return merge_pairs

# def find_cluster_pairs_to_merge(adata, x, colname, cluster, markers):
#     clusters = adata.obs[colname].unique().tolist()
#     merge_pairs = []
#     if len(markers)==0:
#         merge_pairs = [(cluster, cluster2) for cluster2 in clusters if cluster2!=cluster]
#         return merge_pairs

#     for cluster2 in clusters:
#         if cluster2==cluster: continue
#         merge = True
#         for gene in markers:
#             x_gene = x[adata.obs[colname]==cluster, adata.var_names==gene]
#             # merge_cutoff = np.mean(adata[adata.obs[colname]==cluster, gene].X) - stds(adata[adata.obs[colname]==cluster, gene].X)
#             merge_cutoff = cp.mean(x_gene) - stds(x_gene)
#             # if np.mean(adata[adata.obs[colname]==cluster2, gene].X) < merge_cutoff:
#             if cp.mean(x[adata.obs[colname]==cluster2, adata.var_names==gene]) < merge_cutoff:
#                 merge = False
#                 break
#         if merge: merge_pairs.append((cluster, cluster2))
            
#     return merge_pairs

# def find_cluster_pairs_to_merge(adata, colname, cluster, markers):
#     clusters = adata.obs[colname].unique().tolist()
#     merge_pairs = []
#     if len(markers)==0:
#         merge_pairs = [(cluster, cluster2) for cluster2 in clusters if cluster2!=cluster]
#         return merge_pairs

#     for cluster2 in clusters:
#         if cluster2==cluster: continue
#         merge = True
#         for gene in markers:
#             merge_cutoff = np.mean(adata[adata.obs[colname]==cluster, gene].X) - stds(adata[adata.obs[colname]==cluster, gene].X)
#             if np.mean(adata[adata.obs[colname]==cluster2, gene].X) < merge_cutoff:
#                 merge = False
#                 break
#         if merge: merge_pairs.append((cluster, cluster2))
            
#     return merge_pairs

def clusters_merge(adata, 
                        subctype_col, 
                        threshold=0.0, 
                        acc_cutoff=0.3, 
                        n_trees=100, 
                        n_top_genes=100, 
                        n_binary_genes=30, 
                        n_genes_eval=15):
    """ Need a description. """
    # 1. find markers
    # 2. if no low acc, exit
    # 3. for each low acc cluster, check his pairs
    #     metrics include:
    #         pca.harmony mean, corr
    #         HVG mean, corr
    #         markers mean, corr
    #         each single marker, difference
    #     pairs for low acc clusters
    # 4. for each high acc cluster, check his pairs
    #     pairs for high acc clusters 
    # 5. finalize pair.
    #     for each pair in low, check if the oppostie in low union high
    #     if yes, final_pair.append().
    # 6. union find final pairs; then rename
    dict_acc, dict_markers = find_markers(adata, subctype_col, n_trees=n_trees, n_top_genes=n_top_genes, n_binary_genes=n_binary_genes, n_genes_eval=n_genes_eval)
    # acc_cutoff = 0.3
    low_acc_clusters = [cluster for cluster in dict_acc.keys() if dict_acc[cluster]<acc_cutoff]
    high_acc_clusters = [cluster for cluster in dict_acc.keys() if dict_acc[cluster]>=acc_cutoff]
    if len(low_acc_clusters)==0:
        logger.debug(f'{subctype_col} No need to merge')
        return subctype_col
    
    pairs_left_low = []
    pairs_left_high = []
    start_find_cluster = time.time()
    
    # combine all markers and subset it by markers
    markers = []
    for key in dict_markers:
        for marker in dict_markers[key]:
            if marker not in markers:
                markers.append(marker)
    adata_subset = adata[:, markers]
    x = X_to_GPU(adata_subset.X)
    logger.debug("starting to merge clusters")
    for cluster in low_acc_clusters:
        tmp_pairs = find_cluster_pairs_to_merge(adata_subset, x, subctype_col, cluster, dict_markers[cluster], threshold=threshold)
        pairs_left_low.extend(tmp_pairs)
        
    for cluster in high_acc_clusters:
        tmp_pairs = find_cluster_pairs_to_merge(adata_subset, x, subctype_col, cluster, dict_markers[cluster], threshold=threshold)
        pairs_left_high.extend(tmp_pairs)
    
    logger.debug(f'find cluster: {time.time() - start_find_cluster}')
    pairs_all_set = set(pairs_left_high + pairs_left_low)
    final_merge_pairs = [pair for pair in pairs_left_low if pair[::-1] in pairs_all_set]
    
    if len(final_merge_pairs)==0:
        logger.debug(f'{subctype_col} No need to merge')
        return subctype_col
        
    my_merge_data = data2UF(list(dict_acc.keys()), final_merge_pairs)
    dict_merge = my_merge_data.union_pairs()
    merged_col_name = f'{subctype_col}_merged_{threshold}'
    adata.obs[merged_col_name] = adata.obs[subctype_col].apply(lambda x: dict_merge.get(x,x))
    
    clusters = adata.obs[merged_col_name].unique().tolist()
    cluster_size = {cluster:adata.obs[adata.obs[merged_col_name]==cluster].shape[0] for cluster in clusters}
    clusters.sort(key=lambda _: -cluster_size[_])
    clusters_rename = {clusters[i]:str(i) for i in range(len(clusters))}
    
    adata.obs[merged_col_name] = adata.obs[merged_col_name].apply(lambda x: clusters_rename[x])
    return merged_col_name

# =======================================================================================================================
# Hi Haotian, please convert this myRandomForest into a gpu version
@timer
def compute_importances_cuml(clf):
    def dfs(node, importances):
        if 'children' not in node:
            return 
        feature = node['split_feature']
        gain = node['gain']
        count = node['instance_count']
        importances[feature] += gain * count / clf.n_rows
        for child in node['children']:
            dfs(child, importances)
        
    trees = json.loads(clf.get_json())
    imp_all_trees = []
    for tree in trees:
        imp = np.zeros(clf.n_features_in_, dtype=float)
        dfs(tree, imp)
        imp_all_trees.append(imp)

    importances = np.mean(np.array(imp_all_trees), axis=0)
    return importances / importances.sum()


@timer
def myRandomForestCuml(adata, adata_X_gpu, df_dummies, cl, n_trees, n_jobs, n_top_genes):
    # x_train = adata.to_df()
    x_train = adata_X_gpu
    y_train = df_dummies[cl]
    # rf_clf = RandomForestClassifier(n_estimators=n_trees, n_jobs=n_jobs, random_state=123456) #<===== criterion=“gini”, by default
    rf_clf = cumlRandomForestClassifier(n_estimators=n_trees, max_features='sqrt', split_criterion=0) #<===== criterion=“gini”, by default
    rf_clf.fit(x_train, y_train)
    ## get feature importance and rank/subset top genes
    feature_importances = compute_importances_cuml(rf_clf)
    # top_rf_genes = pd.Series(rf_clf.feature_importances_, index=adata.var_names).sort_values(ascending=False)[:n_top_genes]
    top_rf_genes = pd.Series(feature_importances, index=adata.var_names).sort_values(ascending=False)[:n_top_genes]
    del(rf_clf)
    return top_rf_genes    
# =======================================================================================================================


# ---------------------------------------------------------------------------------------------------------------------
# my XGBoost feature evaluator 
@timer 
def myXGBClassifier(dtrain, adata, df_dummies, cl, n_trees, n_top_genes):
    dtrain.set_label(df_dummies[cl])
    params = {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "device": "cuda",}
    bst = xgb.train(params, dtrain, num_boost_round=n_trees)
    score = bst.get_score(importance_type='total_gain')
    n_top_genes = min(n_top_genes, len(score))
    score_sorted = sorted(score, key=lambda x: score[x], reverse=True)[:n_top_genes]
    score_sorted_ind = [adata.var_names[int(fstr[1:])] for fstr in score_sorted]
    importance_sorted = [score[fstr] for fstr in score_sorted]
    top_rf_genes = pd.Series(importance_sorted, index=score_sorted_ind)
    del(bst)
    return top_rf_genes
   

@timer
def myRandomForestSklearn(adata, adata_X_gpu, df_dummies, cl, n_trees, n_jobs, n_top_genes):
    # x_train = adata.to_df()
    x_train = adata_X_gpu
    y_train = df_dummies[cl]
    rf_clf = RandomForestClassifier(n_estimators=n_trees, n_jobs=n_jobs, random_state=123456) #<===== criterion=“gini”, by default
    # rf_clf = cumlRandomForestClassifier(n_estimators=n_trees, max_features='sqrt', split_criterion=0) #<===== criterion=“gini”, by default
    rf_clf.fit(x_train, y_train)
    ## get feature importance and rank/subset top genes
    feature_importances = compute_importances_cuml(rf_clf)
    # top_rf_genes = pd.Series(rf_clf.feature_importances_, index=adata.var_names).sort_values(ascending=False)[:n_top_genes]
    top_rf_genes = pd.Series(feature_importances, index=adata.var_names).sort_values(ascending=False)[:n_top_genes]
    del(rf_clf)
    return top_rf_genes    
# =======================================================================================================================


## construct decision tree for each gene and evaluate the fbeta score in all combinations ==> outputs markers with max fbeta, and all scores
# =======================================================================================================================
# Hi Haotian, please convert this myDecisionTreeEvaluation into a gpu version


# @timer
# def myDecisionTreeEvaluation(adata, df_dummies, cl, genes_eval, beta):
#     x_train = adata.X[:,adata.var_names.isin(genes_eval)]
#     y_train = df_dummies[cl]
#     tree_clf = DecisionTreeClassifier(max_leaf_nodes=2)
#     tree_clf = tree_clf.fit(x_train, y_train)
#     y_pred = tree_clf.apply(x_train) - 1
#     fbeta = fbeta_score(y_train, y_pred, average='binary', beta=beta)
#     ppv = precision_score(y_train, y_pred, average='binary', zero_division=0)
#     tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
#     scores = fbeta, ppv, tn, fp, fn, tp
#     return genes_eval, scores, fbeta

# ---------------------------------------------  GPU version ------------------------------------------------------------

@timer
def myDecisionTreeEvaluation(adata, df_dummies, cl, genes_eval, beta):
    x_train = adata.X[:,adata.var_names.isin(genes_eval)].toarray()
    y_train = df_dummies[cl]
    tree_clf = cumlRandomForestClassifier(n_estimators=1, max_leaves=2, max_features=1.0, split_criterion='gini')
    tree_clf = tree_clf.fit(x_train, y_train)
    y_pred = tree_clf.predict(x_train)
    fbeta = fbeta_score(y_train, y_pred, average='binary', beta=beta)
    ppv = precision_score(y_train, y_pred, average='binary', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    scores = fbeta, ppv, tn, fp, fn, tp
    return genes_eval, scores, fbeta
# =======================================================================================================================

def specificity_score(adata=None, ctype_col:str=None, glist:list=None):
        
    ctypes = adata.obs[ctype_col].unique().tolist()
        
    if glist and len(glist):
        # adata = adata[:, adata.var_names.isin(set(glist))].copy()
        adata = adata[:, adata.var_names.isin(set(glist))]
        if adata.shape[-1]==0:
            raise ValueError("No gene found! Pls check your gene list!")

    adata_ctype_dict = {ctype:adata[adata.obs[ctype_col]==ctype] for ctype in ctypes}
    mean_exp_dict = {ctype: np.squeeze(np.asarray(adata_ctype_dict[ctype].X.sum(axis=0))) / adata_ctype_dict[ctype].n_obs for ctype in ctypes}
    df = pd.DataFrame(data=mean_exp_dict, index=adata.var_names)
    df['all'] = df.sum(axis=1)
    
    df = df.div(df['all'], axis=0)
    
    # assign nan for genes not found
    # gene in glist order
    if glist and len(glist): df = df.reindex(glist)   
    return df

def fraction_cells(adata=None, ctype_col:str=None, glist:list=None):
    """
    Given adata.X (n cells * m genes), ctype_col (a column name in adata.obs that stores the cell type annotation), and a glist (for example, [gene1, gene2, ..., genek])
    The definiation of Fraction of expression := # cells>0  / # total cells.
    Assume in total c different cell types
    for each cell type, subset the adata, and then calculate the fraction of expression of each gene  
    return the fraction dataframe, k rows, c columns.
    """
    ctypes = adata.obs[ctype_col].unique().tolist()
    # ctype_dict = {_: set([_]) for _ in ctypes}
    gset = set(glist)
    # adata = adata[:, adata.var_names.isin(gset)].copy()
    adata = adata[:, adata.var_names.isin(gset)]
    # all non-zero values to 1. because calculating fraction needs 1
    adata.X = adata.X.sign()
    # subset each cell type 
    adata_ctype_dict = {ctype:adata[adata.obs[ctype_col]==ctype] for ctype in ctypes}
    # calculate fraction
    fraction_dict = {ctype: np.squeeze(np.asarray(adata_ctype_dict[ctype].X.sum(axis=0))) / adata_ctype_dict[ctype].n_obs for ctype in ctypes}
    df = pd.DataFrame(data=fraction_dict, index=adata.var_names) 
    # re-order gene index
    if glist and len(glist): df = df.reindex(glist)
    return df

###################
## Main function ##
###################
# 1. have to re-write. because full matrix -> high mem cost, high time cost.
# 2. why the authors use dense matrix? due to dataframe, or class??
# 3. 

# results = ns.NSForest(adata, cluster_header=subctype_col, n_trees=100, n_top_genes=100, n_binary_genes=30, n_genes_eval=15, output_folder=f"{celltype}_{res}/")

def myNSForest(adata, cluster_header, cluster_list=None, medians_header=None,
             n_trees=100, n_jobs=-1, beta=0.5, n_top_genes=15, n_binary_genes=10, n_genes_eval=6,
             output_folder=".", save_results=False):
    
    ## set up outpur folder
    if save_results:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
    logger.debug("Preparing data...")
    start_time = time.time()
    
    # ## densify X from sparse matrix format
    # adata.X = adata.to_df() 

    ## categorial cluster labels
    adata.obs[cluster_header] = adata.obs[cluster_header].astype('category')
    ## dummy/indicator for one vs. all Random Forest model
    df_dummies = pd.get_dummies(adata.obs[cluster_header]) #cell-by-cluster
    ## get number of cluster
    n_total_clusters = len(df_dummies.columns)
    logger.debug(f'n_total_clusters: {n_total_clusters}')
    logger.debug(f"{(time.time() - start_time)} seconds")

    if medians_header == None:
        logger.debug("Calculating medians...")
        # =======================================================================================================================
        # Hi Haotian, please re-write this section. avoid using dense matrix.
        # https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/sparsefuncs.py#L441
        
        start_time = time.time()
        # cluster_medians = calculate_median_sparse(adata, cluster_header)
        # cluster_medians = calculate_median_sparse_in_batch(adata, cluster_header)
        cluster_medians = calculate_median_sparse_fast_gpu(adata, cluster_header)
        # cluster_medians = calculate_median_sparse_csc_median(adata, cluster_header)
        # cluster_medians.to_csv('cluster_median1.csv', index=None)
        ## get dataframes for X and cluster in a column
        # if isinstance(adata.X, np.ndarray):
        #     df_X = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names) #cell-by-gene
        # else:
        #     df_X = pd.DataFrame(adata.X.todense(), index=adata.obs_names, columns=adata.var_names) #cell-by-gene
            
        # gc.collect()
        
        # clusters = adata.obs[cluster_header]
        # df_X_clusters = pd.concat([df_X, clusters], axis=1)
        # ## get cluster medians
        # cluster_medians = df_X_clusters.groupby([cluster_header]).median() #cluster-by-gene
        # ## delete to free up memories
        # del df_X, clusters, df_X_clusters
        # =======================================================================================================================
        # translate to cudf 

        # start_time = time.time()
        # ## get dataframes for X and cluster in a column
        # if isinstance(adata.X, np.ndarray):
        #     df_X = cudf.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names) #cell-by-gene
        # else:
        #     df_X = cudf.DataFrame(adata.X.todense(), index=adata.obs_names, columns=adata.var_names) #cell-by-gene
            
        # gc.collect()
        
        # clusters = cudf.Series(adata.obs[cluster_header])
        # df_X_clusters = cudf.concat([df_X, clusters], axis=1)
        # print(df_X_clusters.shape)
        # print(df_X_clusters.head(10))
        # ## get cluster medians
        # print(time.time() - start_time)
        # cluster_medians = df_X_clusters.groupby([cluster_header]).median().to_pandas() #cluster-by-gene
        # ## delete to free up memories
        # del df_X, clusters, df_X_clusters

        # ----------------------------------------------------------------------------------------------------------------------
        logger.debug(f"{(time.time() - start_time)} seconds")
    else:
        logger.debug("Getting pre-calculated medians...")
        start_time = time.time()
        cluster_medians = adata.varm[medians_header].transpose() #cluster-by-gene
        logger.debug(f"{(time.time() - start_time)} seconds")
    
    ### START iterations ###
    if cluster_list == None:
        cluster_list = df_dummies.columns
    n_clusters = len(cluster_list)
    
    logger.debug("Number of clusters to evaluate: " + str(n_clusters))
    ct = 0
    df_supp = df_markers = df_results = pd.DataFrame()
    start_time = time.time()
    dtrain = xgb.DMatrix(adata.X)
    for cl in cluster_list:
        ct+=1
        logger.debug(str(ct) + " out of " + str(n_clusters) + ":")

        ## cluster in iteration
        logger.debug("\t" + str(cl))
        
        ##=== reset parameters for this iteration!!! (for taking care of special cases) ===##
        n_binary_genes_cl = n_binary_genes
        n_genes_eval_cl = n_genes_eval

        ## Random Forest step: get top genes ranked by Gini/feature importance
        top_rf_genes = myXGBClassifier(dtrain, adata, df_dummies, cl, n_trees, n_top_genes) # xgboost Imp
        # top_rf_genes = myRandomForestCuml(adata, adata_X_gpu, df_dummies, cl, n_trees, n_jobs, n_top_genes) # cuml Imp
        # top_rf_genes = myRandomForestSklearn(adata, df_dummies, cl, n_trees, n_jobs, n_top_genes) # sklearn Imp

        ## filter out negative genes by thresholding median>0 ==> to prevent deviding by 0 in binary score calculation
        top_gene_medians = cluster_medians.loc[cl,top_rf_genes.index]
        top_rf_genes_positive = top_gene_medians[top_gene_medians>0]
        n_positive_genes = sum(top_gene_medians>0)
    
        ##=== special cases: ===##
        if n_positive_genes == 0:
            logger.debug("\t" + "No positive genes for evaluation. Skipped. Optionally, consider increasing n_top_genes.")
            continue

        if n_positive_genes < n_binary_genes:
            logger.debug("\t" + f"Only {n_positive_genes} out of {n_top_genes} top Random Forest features with median > 0 will be further evaluated.")
            n_binary_genes_cl = n_positive_genes
            n_genes_eval_cl = min(n_positive_genes, n_genes_eval)
        ##===##
            
        ## Binary scoring step: calculate binary scores for all positive top genes
        binary_scores = [sum(np.maximum(0,1-cluster_medians[i]/cluster_medians.loc[cl,i]))/(n_total_clusters-1) for i in top_rf_genes_positive.index]
        top_binary_genes = pd.Series(binary_scores, index=top_rf_genes_positive.index).sort_values(ascending=False)

        ## Evaluation step: calculate F-beta score for gene combinations
        genes_eval = top_binary_genes.index[:n_genes_eval_cl].to_list()
        binary_genes_list = top_binary_genes.index[:n_binary_genes_cl].to_list()
        # print("before decision tree --- %s seconds ---" % (time.time() - start_time))
        markers, scores, score_max = myDecisionTreeEvaluation(adata, df_dummies, cl, binary_genes_list, beta)
        # print("after decision tree --- %s seconds ---" % (time.time() - start_time))
        logger.debug("\t" + str(markers))
        logger.debug("\t" + str(score_max))

        ## return supplementary table as csv
        df_supp_cl = pd.DataFrame({'clusterName': cl,
                                   'binary_genes': binary_genes_list,
                                   'rf_feature_importance': top_rf_genes[binary_genes_list],
                                   'cluster_median': top_gene_medians[binary_genes_list],
                                   'binary_score': top_binary_genes[binary_genes_list]}).sort_values('binary_score', ascending=False)
        df_supp = pd.concat([df_supp,df_supp_cl]).reset_index(drop=True)
        if save_results:
            df_supp.to_csv(output_folder + "NSForest_supplementary.csv", index=False)

        ## return markers table as csv
        df_markers_cl = pd.DataFrame({'clusterName': cl, 'markerGene': markers, 'score': scores[0]})
        df_markers = pd.concat([df_markers, df_markers_cl]).reset_index(drop=True)
        if save_results:
            df_markers.to_csv(output_folder + "NSForest_markers.csv", index=False)

        ## return final results as dataframe
        dict_results_cl = {'clusterName': cl,
                           'clusterSize': int(scores[4]+scores[5]),
                           'f_score': scores[0],
                           'PPV': scores[1],
                           'TN': int(scores[2]),
                           'FP': int(scores[3]),
                           'FN': int(scores[4]),
                           'TP': int(scores[5]),
                           'marker_count': len(markers),
                           'NSForest_markers': [markers],
                           'binary_genes': [df_supp_cl['binary_genes'].to_list()] #for this order is the same as the supp order
                           }
        df_results_cl = pd.DataFrame(dict_results_cl)
        df_results = pd.concat([df_results,df_results_cl]).reset_index(drop=True)

    # release resource
    del(dtrain)

    logger.debug(f"{(time.time() - start_time)} seconds")
    ### END iterations ###
    return(df_results)