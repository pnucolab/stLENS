import scanpy as sc
import cupy as cp
import pandas as pd
from scipy import stats, linalg
import scipy
import scipy.sparse as sp
import numpy as np
from tqdm.auto import tqdm
from scipy.sparse.linalg import svds
import torch
import random
import zarr
import anndata

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.cm as cm

import seaborn as sns

import psutil
import os
import gc

import dask.array as da
from dask import delayed

from .PCA import PCA
from .calc import Calc

class scLENS2():
    def __init__(self, sparsity='auto',
                 sparsity_step=0.001,
                 sparsity_threshold=0.9,
                 perturbed_n_scale=2,
                 device=None,
                 n_rand_matrix=20,
                 threshold=0.3420201433256688,
                 data=None,
                 chunk_size='auto'):  # chunk_size 추가
        self.L = None
        self.V = None
        self.L_mp = None
        self.explained_variance_ = []
        self.total_variance_ = []
        self.sparsity = sparsity
        self.sparsity_threshold = sparsity_threshold
        self.sparsity_step = sparsity_step
        self.preprocessed = False
        self._perturbed_n_scale = perturbed_n_scale
        self.device = device
        self.n_rand_matrix = n_rand_matrix
        self.threshold = threshold
        self.data = data
        self.chunk_size = chunk_size  # 청크 크기 설정

    def filtering(self, data, min_tp=0, min_genes_per_cell=200, min_cells_per_gene=15):
        if isinstance(data, pd.DataFrame):
            if not data.index.is_unique:
                print("Cell names are not unique, resetting cell names")
                data.index = range(len(data.index))

            if not data.columns.is_unique:
                print("Removing duplicate genes")
                data = data.loc[:, ~data.columns.duplicated()]

            data_array = data.values

        elif isinstance(data, sc.AnnData):
            print("adata -> sparse")
            data_array = data.X  
        else:
            data_array = data

        if isinstance(data_array, sp.spmatrix):
            print("issparse!")
            gene_sum = np.asarray(data_array.sum(axis=0)).flatten()
            cell_sum = np.asarray(data_array.sum(axis=1)).flatten()
            non_zero_genes = data_array.getnnz(axis = 0)
            non_zero_cells = data_array.getnnz(axis = 1)
        else:
            print("is not sparse")
            gene_sum = np.sum(data_array, axis=0)
            cell_sum = np.sum(data_array, axis=1)
            non_zero_genes = np.count_nonzero(data_array, axis=0)
            non_zero_cells = np.count_nonzero(data_array, axis=1)

        self.normal_genes = np.where((gene_sum > min_tp) & (non_zero_genes >= min_cells_per_gene))[0]
        self.normal_cells = np.where((cell_sum > min_tp) & (non_zero_cells >= min_genes_per_cell))[0]

        del gene_sum, cell_sum, non_zero_genes, non_zero_cells
        gc.collect()

        self._raw = data_array[self.normal_cells][:, self.normal_genes]
        del data_array
        gc.collect()

        # if isinstance(self._raw, sp.spmatrix):
        #     print("sparse -> array")
        #     self._raw = self._raw.toarray()

        print(f'Removed {data.shape[0] - len(self.normal_cells)} cells and '
                f'{data.shape[1] - len(self.normal_genes)} genes in QC')
        
        return self._raw

    def normalize(self, _raw):
            
        chunk_size = (10000, _raw.shape[1])
        if isinstance(_raw, da.core.Array):
            X = _raw
        else:
            X = da.from_array(_raw, chunks=chunk_size)
        l1_norm = da.linalg.norm(X, ord=1, axis=1, keepdims=True)
        X /= l1_norm
        del l1_norm
        gc.collect()

        X = da.log(X + 1)

        mean = da.mean(X, axis=0)
        std = da.std(X, axis=0)
        X = (X - mean) / std
        del mean, std
        gc.collect()

        l2_norm = da.linalg.norm(X, ord=2, axis=1, keepdims=True)
        X /= l2_norm
        X *= da.mean(l2_norm)
        X -= da.mean(X, axis=0)
        del l2_norm
        gc.collect()

        return X
    
    def preprocess(self, data, plot=False):
        
        print("filtering")
        self._raw = self.filtering(data) # sparse
        if sp.issparse(self._raw): 
            # raw_ann = anndata.AnnData(X=self._raw)
            # raw_ann.write(f"../{self._raw.shape[0]}/file/raw_ann.h5ad")
            print("sparse -> dense")
            self._raw = self._raw.toarray() # dense
        normalized_X = self.normalize(self._raw)
        normalized_X.to_zarr(f"../{self._raw.shape[0]}/file/normalized_X.zarr")

        # self.X = normalized_X.compute()
        
        data = data[self.normal_cells, self.normal_genes] # anndata 크기 업데이트
        del self.normal_cells, self.normal_genes
        gc.collect()

        if plot:
            print("plotting")
            self.X = normalized_X.compute()
            fig1, axs1 = plt.subplots(1, 2, figsize=(10, 5))

            axs1[0].hist(np.mean(self._raw, axis=1), bins=100)
            axs1[1].hist(cp.mean(self.X, axis=1), bins=100) 
            fig1.suptitle('Mean of Gene Expression along Cells')
            fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))
            axs2[0].hist(np.std(self._raw, axis=0), bins=100)
            axs2[1].hist(cp.std(self.X, axis=0), bins=100) 
            fig2.suptitle('SD of Gene Expression for each Gene')

            if isinstance(data, sc.AnnData):
                data.uns['preprocess_mean_plot'] = fig1
                data.uns['preprocess_sd_plot'] = fig2

        self.data = data
        self.preprocessed = True

        del normalized_X, data
        gc.collect()
    
    def _preprocess_rand(self, X, inplace=True, chunk_size = 'auto'):
        if not inplace:
            X = X.copy()
        if sp.issparse(X): 
            X = X.toarray() # dense
        X = self.normalize(X)
        return X.compute()
    
    def fit_transform(self, data=None, device=None, eigen_solver='wishart', plot_mp = False):
        
        if data is None and not self.preprocessed:
            raise Exception('No data has been provided. Provide data directly or through the preprocess function')
        if not self.preprocessed:
            if isinstance(data, pd.DataFrame):
                self._raw = data.values
                self.X = data.values
            elif isinstance(data, cp.ndarray):
                self._raw = data
                self.X = data
            elif isinstance(data, sc.AnnData):
                self._raw = data.X
                self.X = data.X
            else:
                raise ValueError("Data must be a pandas DataFrame or cupy ndarray")
        
        self.X = da.from_zarr(f"../{self._raw.shape[0]}/file/normalized_X.zarr")
        pca_result = self._PCA(self.X, device = None, plot_mp = plot_mp)

        eigenvalue, eigenvector = pca_result

        if self.X.shape[0] <= self.X.shape[1]:
            self._signal_components = eigenvector
        else:
            eigenvalue = cp.asnumpy(eigenvalue)
            eigenvector = cp.asnumpy(eigenvector)
            cbyc = sclens.X @ eigenvector @ np.diag(np.sqrt(eigenvalue))
            self._signal_components, _ = np.linalg.qr(cbyc)
            self._signal_components = self._signal_components.compute()
            self._signal_components = cp.asarray(self._signal_components)

        print(self._signal_components.shape)

        del eigenvalue, eigenvector
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        if self.sparsity == 'auto':
            self._calculate_sparsity()
            
        # if self.preprocessed:
        #     raw = self._raw

        pert_vecs = list()
        for _ in tqdm(range(self.n_rand_matrix), total=self.n_rand_matrix):
            rand = scipy.sparse.rand(self.X.shape[0], self.X.shape[1], 
                                    density=1-self.sparsity, 
                                    format='csr')
            rand.data[:] = 1
            rand = rand.toarray()
            rand += self._raw
            rand = self._preprocess_rand(rand)
            n = min(sclens._signal_components.shape[1] * sclens._perturbed_n_scale, sclens.X.shape[1])

            if self.device == 'cpu':
                pass
            elif self.device == 'gpu':
                perturbed = self._PCA_rand(rand, n)
                perturbed = cp.asarray(perturbed)

            # if self.device == 'cpu':
            #     U, S, VT = svds(rand, k=2*k)
            #     if rand.shape[0] <= rand.shape[1]:
            #         perturbed = np.array(U)
            #     else:
            #         perturbed = np.array(VT)
            
            # elif self.device == 'gpu':
            #     X = torch.tensor(rand, device="cuda", dtype=torch.float32)
            #     U, S, VT = torch.svd_lowrank(X, q=2*k)
            #     perturbed = cp.asarray(U)
            #     # if rand.shape[0] <= rand.shape[1]:
            #     #     perturbed = cp.asarray(U)
            #     # else:
            #     #     perturbed = cp.asarray(VT)
            # else:
            #     print("The device can be either cpu or gpu.")
            
            # del U, S, VT
            # gc.collect()
            # cp.get_default_memory_pool().free_all_blocks()
            # cp.get_default_pinned_memory_pool().free_all_blocks()

            pert_select = self._signal_components.T @ perturbed
            pert_select = cp.abs(pert_select)
            pert_select = cp.argmax(pert_select, axis = 1)
            pert_vecs.append(perturbed[:, pert_select])

            del rand, perturbed, pert_select
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

        pert_scores = list()
        for i in range(self.n_rand_matrix):
            for j in range(i+1, self.n_rand_matrix):
                dots = pert_vecs[i].T @ pert_vecs[j]
                corr = cp.max(cp.abs(dots), axis = 1)
                pert_scores.append(corr.get())

        pert_scores = cp.array(pert_scores)
        pvals = cp.sum(pert_scores < self.threshold, axis=0) / pert_scores.shape[0]
        self._robust_idx = pvals < 0.01
        self.X_transform = pca_result[1][:, self._robust_idx] * cp.sqrt(pca_result[0][self._robust_idx]).reshape(1, -1)
        self.robust_scores = pert_scores

        del pert_scores, pert_vecs
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        if self.X.shape[0] <= self.X.shape[1]:
            self.data.obsm['PCA_scLENS'] = self.X_transform
        else:
            self.data.varm['PCA_scLENS'] = self.X_transform

        return self.X_transform
    
        
    def _calculate_sparsity(self):
        """Automatic sparsity level calculation"""
        sparse = 0.999
        shape_row, shape_col = self._raw.shape
        n_len = shape_row * shape_col
        n_zero = n_len - self._raw.size

        rng = np.random.default_rng()
        # Calculate threshold for correlation
        n_sampling = min(self._raw.shape)
        thresh = np.mean([max(np.abs(rng.normal(0, np.sqrt(1/n_sampling), n_sampling)))
                            for _ in range(5000)]).item()
        print(f'sparsity_th: {thresh}')

        zero_indices_dict = {}
        for row in range(shape_row):
            col = self._raw[row, :]  # CSR 형식 (1, shape_col)
            zero_indices = np.setdiff1d(np.arange(shape_col), col.indices, assume_unique=True).astype(np.int32)
            if zero_indices.size > 0:
                zero_indices_dict[row] = zero_indices  # 0이 있는 row만 저장

        _raw_ann.X = None
        del _raw_ann

        # Construct binarized data matrix
        bin = scipy.sparse.csr_array(self._raw)
        bin.data[:] = 1.
        # 16일 수정
        # bin = cp.array(bin.toarray(), dtype=cp.float32)
        bin = bin.toarray()
        Vb = self._PCA_rand(self._preprocess_rand(bin, inplace=False), bin.shape[0])
        n_vbp = Vb.shape[1]//2
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        n_buffer = 5
        buffer = [1] * n_buffer
        while sparse > self.sparsity_threshold:
            n_pert = int((1-sparse) * n_len)
            selection = np.random.choice(n_zero,n_pert,replace=False)
            idx = [x[selection] for x in zero_idx]

            # Construct perturbed data matrix
            # 16일 수정
            # pert = cp.zeros_like(bin)
            pert = np.zeros_like(bin)
            pert[tuple(idx)] = 1
            pert += bin

            del idx
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            
            
            pert = self._preprocess_rand(pert)
            if pert.shape[0] <= pert.shape[1]:
                pert = pert @ pert.T
            else:
                pert = pert.T @ pert
            pert /= pert.shape[1]  # numpy

            # 18일 수정
            # pert가 numpy여서 cupy로 바꾼 후 eigenvalue decomposition
            pert = cp.asarray(pert)
            Vbp = cp.linalg.eigh(pert)[1][:, -n_vbp:]
            
            del pert
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

            corr_arr = cp.max(cp.abs(Vb.T @ Vbp), axis=0).get()
            del Vbp  
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

            corr = np.sort(corr_arr)[1]
            
            buffer.pop(0)
            buffer.append(corr)
            

            print(f'Min(corr): {corr}, sparsity: {sparse}, add_ilen: {selection.shape}')
            if all([x < thresh for x in buffer]):
                 self.sparsity = sparse + self.sparsity_step * (n_buffer - 1)
                 break
            
            sparse -= self.sparsity_step
        del bin, Vb, selection
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        
    def _PCA(self, X, device = None, plot_mp = False):
        pca = PCA(device = self.device)
        pca.fit(X)
        
        if plot_mp:
            pca.plot_mp(comparison = False)
            plt.show()
        
        comp = pca.get_signal_components()

        del pca
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()
    
        return comp
    
    # def _PCA_rand(self, X, n, use_numpy_if_oom=True, batch_size=5000):
    #     try:
    #         W = (X.T @ X) / X.shape[1]
    #         _, V = cp.linalg.eigh(W)  # 기본적으로 CuPy 연산 수행
    #     except cp.cuda.memory.OutOfMemoryError:
    #         if use_numpy_if_oom:
    #             print("Out of memory detected, switching to NumPy.")
    #             X = cp.asnumpy(X)  # NumPy 변환
    #             W = (X.T @ X) / X.shape[1]
    #             _, V = np.linalg.eigh(W)  # NumPy 연산 사용
    #             return cp.asarray(V)  # 다시 CuPy 배열로 변환
    #         else:
    #             raise  # OOM 발생 시 예외를 그대로 전달

    #     return V

    def _PCA_rand(self, X, n):
        print("pca - binary matrix")
        pca = PCA(device = self.device)
        L, V = pca._get_eigen(X)
        V = V[:, -n:]

        # # SRT에서 gene x gene matrix로 하는 경우
        # if X.shape[0] > X.shape[1]:
        #     L = L[-n:]
        #     V = cp.asnumpy(V)
        #     L = cp.asnumpy(L)
        #     vec = X @ V @ np.diag(np.sqrt(L))
        #     V, _ = np.linalg.qr(vec)

        #     del vec, _
        
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        return L, V
    

    def plot_robust_score(self):
        fig1, axs1 = plt.subplots(1, 1, figsize=(10, 5))
        for i in range(self.robust_scores.shape[1]):
            if i in np.where(self._robust_idx)[0]:
                axs1.scatter(i*np.ones_like(self.robust_scores[:, i].get()), self.robust_scores[:, i].get(), c='g', alpha=0.1)
            else:
                axs1.scatter(i*np.ones_like(self.robust_scores[:, i].get()), self.robust_scores[:, i].get(), c='r', alpha=0.1)
        axs1.axhline(y=self.threshold, color='k', linestyle='--')
        axs1.set_ylabel('Robustness Score')
        axs1.set_title('Signal Component Robustness')
    
        if isinstance(self.data, sc.AnnData):
            self.data.uns['robust_score_plot'] = fig1

            import numpy as np


    # def plot_robust_score():
    #     m_scores = sclens.robust_scores.mean(axis=0).get()  # mean stability per component
    #     sd_scores = sclens.robust_scores.std(axis=0).get()   # std deviation per component
    #     nPC = np.arange(1, len(m_scores)+1)

    #     # colormap (inverse of m_scores like Julia code: 1 - m_scores)
    #     colors = cm.get_cmap("RdBu")(1.0 - m_scores)

    #     fig, ax = plt.subplots(figsize=(10, 5))
    #     ax.set_xlabel("nPC")
    #     ax.set_ylabel("Stability")
    #     ax.set_title(f"{np.sum(sclens._robust_idx)} robust signals were detected")

    #     # Scatter plot with color encoding
    #     ax.scatter(nPC, m_scores, c=colors, s=50, edgecolor='k')  # size ~ markersize=10 in Julia
    #     ax.errorbar(nPC, m_scores, yerr=sd_scores, fmt='none', color='gray', capsize=4, linewidth=1)

    #     ax.axhline(y=sclens.threshold, color='k', linestyle='--')  # threshold line


        
