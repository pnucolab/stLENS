import numpy as np
import pandas as pd
import cupy as cp
import scipy
import torch
from scipy.cluster.hierarchy import linkage, to_tree
from sklearn.metrics import silhouette_samples
from resample.bootstrap import confidence_interval
from joblib import Parallel
import contextlib
import io
import igraph as ig
import leidenalg as la
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed, wrap_non_picklable_objects
from sklearn.neighbors import NearestNeighbors
import random, math
from numba import cuda 



def snn(X, n_neighbors=20, min_weight=1/15, metric='cosine'):
    # graph = kneighbors_graph(X, n_neighbors=n_neighbors, metric=metric)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, metric=metric).fit(X)
    indices = nbrs.kneighbors(X,return_distance=False)
    indices = indices[:, 1:]

    n_samples = indices.shape[0]
    edges = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors:
            edges.append((i,neighbor))
    
    g = ig.Graph(n=n_samples,edges=edges,directed=False)
    weights = np.array(g.similarity_jaccard(pairs=g.get_edgelist()))
    g.es['weight'] = weights
    
    edges_to_delete = [i for i, w in enumerate(weights) if w < min_weight]
    g.delete_edges(edges_to_delete)
    
    return g


def find_clusters(X, 
                n_neighbors=20, 
                min_weight=1/15, 
                metric='cosine',
                res=1.2,
                n_iterations=-1):
    
    G = snn(X, n_neighbors=n_neighbors, min_weight=min_weight, metric=metric)
    partition = la.find_partition(G,
                                  la.RBConfigurationVertexPartition,
                                  weights=G.es['weight'],
                                  n_iterations=n_iterations,
                                  resolution_parameter=res)
    
    labels = np.zeros(X.shape[0])
    for i, cluster in enumerate(partition):
        for element in cluster:
            labels[element] = i + 1
    
    return labels


def construct_sample_clusters(X,
                              filler=-1,
                              reps=100,
                              size=0.8,
                              res=1.2,
                              n_jobs=None,
                              metric='cosine',
                              batch_size=20,
                              **kwargs):
    """
    Creates clusterings based on a subset of the dataset
    """
    k = int(X.shape[0] * size)
    clusters = []

    if reps is None:
        if not isinstance(res, (list, tuple, np.ndarray)):
            res_list = [res]
        else:
            res_list = res

        total_tasks = len(res_list)
        for batch_start in tqdm(range(0, total_tasks, batch_size), desc='Batched Sampling', **kwargs):
            batch_end = min(batch_start + batch_size, total_tasks)
            batch_res_list = res_list[batch_start:batch_end]

            with tqdm_joblib(desc='Constructing samples', total=len(batch_res_list), **kwargs):
                parallel = Parallel(n_jobs=n_jobs)
                batch_clusters = parallel(
                    delayed(sample_cluster)(X, k=k, res=res_i, filler=filler, sample=False, metric=metric)
                    for res_i in batch_res_list
                )
            clusters.extend(batch_clusters)
    else:
        total_tasks = reps
        for batch_start in tqdm(range(0, total_tasks, batch_size), desc='Batched Sampling', **kwargs):
            batch_end = min(batch_start + batch_size, total_tasks)
            batch_reps = batch_end - batch_start

            with tqdm_joblib(desc='Constructing samples', total=batch_reps, **kwargs):
                parallel = Parallel(n_jobs=n_jobs)
                batch_clusters = parallel(
                    delayed(sample_cluster)(X, k=k, res=res, filler=filler, metric=metric)
                    for _ in range(batch_reps)
                )
            clusters.extend(batch_clusters)

    return clusters

def chooseR(X, 
            reps=100,
            size=0.8,
            resolutions=None,
            device='gpu',
            n_jobs=None,
            silent=False,
            batch_size = 20,
            metric='cosine'):
    if resolutions is None:
        # resolutions = [0.3, 0.5, 0.8, 1, 1.2, 1.6, 2, 4, 6, 8]
        resolutions = np.arange(0.05, 2, 0.05)
    resolutions = set(resolutions)
    stats = list()
    for res in tqdm(resolutions,
                    desc='ChooseR', 
                    total=len(resolutions), 
                    disable=silent):
        stats_row = [res]
        # print("stats_row : ", stats_row)
        cls = find_clusters(X, res=res,metric=metric)
        # print("cls : ", cls)
        stats_row.append(len(np.unique(cls)))
        
        clusters = construct_sample_clusters(X, 
                                            reps=reps, 
                                            size=size, 
                                            res=res, 
                                            n_jobs=n_jobs,
                                            metric=metric,
                                            batch_size=batch_size,
                                            disable=True)
        # print("clusters:", clusters)
        score = calculate_score(clusters, X.shape[0], reps, device=device)
        
        score = 1 - score
        # print("score : ", score)
        np.fill_diagonal(score, 0)

        sil = silhouette_samples(score, cls, metric='precomputed')
        # print("sil : ", sil)
        sil_grp = group_silhouette(sil, cls)
        # print("sil_grp : ", sil_grp)  

        stats_row.append(confidence_interval(np.median, sil_grp)[0])
        stats_row.append(np.median(sil_grp))

        stats.append(stats_row) 
    
    stats = pd.DataFrame(stats, columns=['res', 'n_clusters', 'low_med', 'med']).sort_values(by=['n_clusters'], ascending=False)
    # print("stats : ", stats)
    threshold = max(stats['low_med'])
    # print("threshold : ", threshold)
    filtered_stats = stats[stats['med'] >= threshold]
    # print("filtered_stats : ", filtered_stats, "filtered_stats_len : ", len(filtered_stats))

    if len(filtered_stats) == 1:
        return filtered_stats['res'], stats
    return filtered_stats['res'].iloc[0], stats


def calculate_score(clusters, n, reps, device='cpu'):
    if device == 'gpu':
        if cuda.is_available():
            return calculate_score_gpu(clusters, n, reps)
        else:
            print('GPU is not available, function will be run in CPU')
            return calculate_score_cpu(clusters, n, reps)
    elif device == 'cpu':
        return calculate_score_cpu(clusters, n, reps)
    else:
        raise Exception("Device not recognized. Please choose one of 'cpu' or 'gpu'")
    
def group_silhouette(sil, labels):
    """
    Computes average per-cluster silhouette score 
    """
    sil_grp = list()
    for cls in set(labels):
        idx = np.where(labels == cls)
        sil_grp.append(np.mean(sil[idx]))
    return sil_grp   

def sample_cluster(X, k, res=1.2, filler=-1, sample=True, metric='cosine'):
    """
    Sample and cluster data
    """
    if not sample:
        cls = find_clusters(X, res=res, metric=metric)
        return cls
    
    row = np.zeros(X.shape[0])
    row.fill(filler)
    sample = random.sample(range(X.shape[0]), k)
    cls = find_clusters(X[sample], res=res, metric=metric)
    np.put(row, sample, cls)
    return row

def calculate_score_gpu(clusters, n, reps, batch_size=3000):
    """
    Score calculation on GPU
    """
    score = np.zeros((n, n), dtype=np.csingle)
    score_device = cuda.to_device(score)

    threadsPerBlock = (16, 16)
    blocksPerGrid_x = math.ceil(n / threadsPerBlock[0])
    blocksPerGrid_y = math.ceil(batch_size / threadsPerBlock[1])
    blocksPerGrid = (blocksPerGrid_x, blocksPerGrid_y)
    
    batch_num = math.ceil(n / batch_size)

    for row in clusters:
        for i in range(batch_num):
            x_batch_start = i * batch_size
            x_batch_end = min((i + 1) * batch_size, n)
            x_batch = row[x_batch_start:x_batch_end]
            x_device = cuda.to_device(x_batch)

            for j in range(batch_num):
                y_batch_start = j * batch_size
                y_batch_end = min((j + 1) * batch_size, n)
                y_batch = row[y_batch_start:y_batch_end]
                y_device = cuda.to_device(y_batch)
                outer_equality_kernel[blocksPerGrid, threadsPerBlock](x_device, y_device, score_device, x_batch_start, y_batch_start)

                del y_device
                cuda.current_context().memory_manager.deallocations.clear()

            del x_device
            cuda.current_context().memory_manager.deallocations.clear()
    
    score = score_device.copy_to_host()
    score = np.where(score.real > 0, percent_match(score, reps), 0)
    
    del score_device
    cuda.current_context().memory_manager.deallocations.clear()
    return score

def calculate_score_cpu(clusters, n, reps):
    score = np.zeros((n, n), dtype=np.csingle)

    for row in clusters:
        mask_valid = row != -1  
        mask_invalid = row == -1  

        equality_matrix = np.equal.outer(row, row) & mask_valid[:, None]

        score += equality_matrix.astype(np.csingle)
        score += (mask_invalid[:, None] | mask_invalid[None, :]) * 1j

    score = np.where(score.real > 0, percent_match(score, reps), 0)

    return score


@cuda.jit
def outer_equality_kernel(x, y, out, x_start, y_start):
    """
    GPU kernel score calculation algorithm
    """
    tx, ty = cuda.grid(2)

    if tx < x.shape[0] and ty < y.shape[0]:
        if x[tx] == -1 or y[ty] == -1:
            out[tx + x_start, ty + y_start] += 1j
        elif x[tx] == y[ty]:
            out[tx + x_start, ty + y_start] += 1

def percent_match(x, reps):
    """
    Percentage of co-clustering
    """
    return np.divide(x.real, (reps - x.imag), where=x.imag!=reps)

@delayed
def outer_equality(x, idx, out):
    """
    CPU score calculation algorithm
    """
    if x[idx] == -1:
        out[:, idx] += 1j
        return
    
    for i in range(x.shape[0]):
        if x[i] == x[idx]:
            out[i, idx] += 1
        elif x[i] == -1:
            out[i, idx] += 1j