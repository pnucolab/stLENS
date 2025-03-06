import scanpy as sc
import cupy as cp
import pandas as pd
from scipy import stats, linalg
import scipy
import scipy.sparse as sp
import numpy as np
from tqdm.auto import tqdm

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import seaborn as sns

import psutil
import os
import gc

import dask.array as da
import dask.distributed
from dask import delayed

from .PCA import PCA

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
        self.chunk_size = (5000, 5000)

    def preprocess(self, data, plot=False):
        def filtering(data, min_tp=0, min_genes_per_cell=200, min_cells_per_gene=15):
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

            self._raw = data_array[self.normal_cells][:, self.normal_genes]
            del data_array

            print("sparse -> array")
            if isinstance(data, sc.AnnData):
                if scipy.sparse.issparse(self._raw):
                    self._raw = self._raw.toarray()  # df는 필요 없음

            print(f'Removed {data.shape[0] - len(self.normal_cells)} cells and '
                    f'{data.shape[1] - len(self.normal_genes)} genes in QC')
            
            return self._raw

        def normalize(_raw):
            chunk_size = (8000,8000)
            X = da.from_array(_raw, chunks=chunk_size)
            l1_norm = da.linalg.norm(X, ord=1, axis=1, keepdims=True)
            X /= l1_norm
            del l1_norm

            X = da.log(X + 1)

            mean = da.mean(X, axis=0)
            std = da.std(X, axis=0)
            X = (X - mean) / std
            del mean, std

            l2_norm = da.linalg.norm(X, ord=2, axis=1, keepdims=True)
            X /= l2_norm
            X *= da.mean(l2_norm)
            X -= da.mean(X, axis=0)
            del l2_norm

            return X
        
        _raw = filtering(data)

        normalized_X = normalize(_raw)
        np_X = normalized_X.compute()
        self.X = np_X
        
        data = data[self.normal_cells, self.normal_genes] # anndata 크기 업데이트

        if plot:
            fig1, axs1 = plt.subplots(1, 2, figsize=(10, 5))
            raw = _raw
            clean = self.X

            # CuPy 배열을 바로 가져와서 NumPy로 변환 없이 처리
            axs1[0].hist(np.mean(raw, axis=1), bins=100)
            axs1[1].hist(cp.mean(clean, axis=1), bins=100) 
            fig1.suptitle('Mean of Gene Expression along Cells')
            fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))
            axs2[0].hist(np.std(raw, axis=0), bins=100)
            axs2[1].hist(cp.std(clean, axis=0), bins=100) 
            fig2.suptitle('SD of Gene Expression for each Gene')

            if isinstance(data, sc.AnnData):
                data.uns['preprocess_mean_plot'] = fig1
                data.uns['preprocess_sd_plot'] = fig2

            self.data = data
            self.preprocessed = True
        
        return self.X , data

    def _preprocess_rand(self, X, inplace=True, batch_size=10000):
        """Preprocessing that does not save data statistics using batch processing"""
        if not inplace:
            X = X.copy()

        num_samples = X.shape[0]

        # L1 정규화 및 로그 변환 (배치 처리)
        for i in range(0, num_samples, batch_size):
            batch = X[i:i + batch_size]
            l1_norm = cp.linalg.norm(batch, ord=1, axis=1, keepdims=True)
            batch /= l1_norm
            del l1_norm
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.Device(0).synchronize()
            
            batch += 1
            # 17일 수정
            # cp -> np
            X[i:i + batch_size] = np.log(batch)
            del batch
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.Device(0).synchronize()

        # Z-score 정규화
        mean = cp.mean(X, axis=0, keepdims=True)
        std = cp.std(X, axis=0, keepdims=True)

        for i in range(0, num_samples, batch_size):
            batch = X[i:i + batch_size]
            batch = (batch - mean) / std
            X[i:i + batch_size] = batch
            del batch
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.Device(0).synchronize()

        del mean, std
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Device(0).synchronize()
        # L2 정규화 (배치 처리)
        for i in range(0, num_samples, batch_size):
            # 17일 수정
            batch = cp.asarray(X[i:i + batch_size])
            l2_norm = cp.linalg.norm(batch, ord=2, axis=1, keepdims=True)
            mean_l2 = cp.mean(l2_norm)
            batch /= l2_norm
            batch *= mean_l2
            # 17일 수정
            # batch가 cupy여서 get 추가
            X[i:i + batch_size] = batch.get()
            del batch, l2_norm, mean_l2
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.Device(0).synchronize()

        # 평균 제거
        mean_X = cp.mean(X, axis=0, keepdims=True)
        for i in range(0, num_samples, batch_size):
            batch = X[i:i + batch_size]
            batch -= mean_X
            X[i:i + batch_size] = batch
            del batch
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.Device(0).synchronize()

        del mean_X
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.cuda.Device(0).synchronize()

        return X
       

    def fit_transform(self, data=None, eigen_solver='wishart', plot_mp = False):
        
        if data is None and not self.preprocessed:
            raise Exception('No data has been provided. Provide data directly or through the preprocess function')
        if not self.preprocessed:
            if isinstance(data, pd.DataFrame):
                self._raw = data.values
                self.X = data.values
            elif isinstance(data, cp.ndarray):
                self._raw = data
                self.X = data
            else:
                raise ValueError("Data must be a pandas DataFrame or cupy ndarray")
        # 16일 수정
        pca_result = self._PCA(self.X, plot_mp = plot_mp)
        self._signal_components = pca_result[1]

        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        if self.sparsity == 'auto':
            self._calculate_sparsity()
            
        if self.preprocessed:
            # 16일 수정
            # raw = cp.array(self._raw, dtype=cp.float32)
            raw = self._raw

        n = min(self._signal_components.shape[1] * self._perturbed_n_scale, self.X.shape[1])
        
        pert_vecs = list()
        for _ in tqdm(range(self.n_rand_matrix), total=self.n_rand_matrix):
            # Construct random matrix
            rand = scipy.sparse.rand(self._raw.shape[0], self._raw.shape[1], 
                                    density=1-self.sparsity, 
                                    format='csr')
            rand.data[:] = 1
            # 16일 수정
            # rand = cp.array(rand.toarray())
            rand = rand.toarray()

            # Construct perturbed components
            rand += raw
            rand = self._preprocess_rand(rand)
            perturbed = self._PCA_rand(rand, n)

            # Select the most correlated components for each perturbation
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

        del raw, pert_scores, pert_vecs
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
        zero_idx = np.nonzero(self._raw == 0)
        n_len = self.X.shape[0]*self.X.shape[1]
        n_zero = zero_idx[0].shape[0]
        
        rng = np.random.default_rng()
        # Calculate threshold for correlation
        n_sampling = min(self.X.shape)
        thresh = np.mean([max(np.abs(rng.normal(0, np.sqrt(1/n_sampling), n_sampling)))
                            for _ in range(5000)]).item()
        print(f'sparsity_th: {thresh}')

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
        
    def _PCA(self, X, plot_mp = False):
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
        # W = (X @ X.T)
        if X.shape[0] <= X.shape[1]:
            W = (X @ X.T)
        else:
            W = (X.T @ X)
        W /= X.shape[1]
        
        del X
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        W = cp.asarray(W)
        _, V = cp.linalg.eigh(W)

        del W
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        V = V[:, -n:]

        del _
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        return V



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
        
