import scanpy as sc
import cupy as cp
import pandas as pd
from scipy import stats, linalg
import scipy
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

from PCA import PCA


class scLENS2():
    def __init__(self, sparsity='auto', 
                 sparsity_step = 0.001, 
                 sparsity_threshold=0.9, 
                 perturbed_n_scale = 2, 
                 device = None,
                 n_rand_matrix=20,
                 threshold=0.3420201433256688,
                 data = None):
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


    def preprocess(self, data, min_tp=0, min_genes_per_cell=200, min_cells_per_gene=15, plot=False):
        self.data = data
        # DataFrame 처리
        if isinstance(data, pd.DataFrame):
            if not data.index.is_unique:
                print("Cell names are not unique, resetting cell names")
                data.index = range(len(data.index))

            if not data.columns.is_unique:
                print("Removing duplicate genes")
                data = data.loc[:, ~data.columns.duplicated()]

            # DataFrame을 numpy로 변환
            data_array = data.values  
        elif isinstance(data, sc.AnnData):
            data_array = data.X  # AnnData의 .X 속성은 numpy 배열로 사용
        else:
            data_array = data

        # 정상 유전자 및 세포 필터링
        self.normal_genes = np.where((np.sum(data_array, axis=0) > min_tp) &
                                (np.count_nonzero(data_array, axis=0) >= min_cells_per_gene))[0]
        self.normal_cells = np.where((np.sum(data_array, axis=1) > min_tp) &
                                (np.count_nonzero(data_array, axis=1) >= min_genes_per_cell))[0]

        # 필터링된 데이터
        self._raw = data_array[self.normal_cells][:, self.normal_genes]

        print(f'Removed {data_array.shape[0] - len(self.normal_cells)} cells and '
              f'{data_array.shape[1] - len(self.normal_genes)} genes in QC')

        # cupy 사용 (torch 대신)
        X = cp.array(self._raw, dtype=cp.float64)

        # L1 정규화 및 로그 변환
        l1_norm = cp.linalg.norm(X, ord=1, axis=1)
        X = X / l1_norm[:, cp.newaxis]
        X += 1
        X = cp.log(X)

        # Z-score 정규화
        mean = cp.mean(X, axis=0)
        std = cp.std(X, axis=0)
        X = (X - mean) / std

        # L2 정규화
        l2_norm = cp.linalg.norm(X, ord=2, axis=1)
        X = X / l2_norm[:, cp.newaxis]
        X *= cp.mean(l2_norm)
        X -= cp.mean(X, axis=0)

        # cupy -> numpy 변환
        self.X = cp.asnumpy(X)
        self.preprocessed = True

        if plot:
            # 그림 생성
            fig1, axs1 = plt.subplots(1, 2, figsize=(10, 5))
            raw = self._raw
            clean = X  # 수정: numpy 상태에서 바로 사용

            axs1[0].hist(np.average(raw, axis=1), bins=100)
            axs1[1].hist(np.average(clean.get(), axis=1), bins=100)
            fig1.suptitle('Mean of Gene Expression along Cells')

            fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))
            axs2[0].hist(np.std(raw, axis=0), bins=100)
            axs2[1].hist(np.std(clean.get(), axis=0), bins=100)
            fig2.suptitle('SD of Gene Expression for each Gene')

            # anndata에 그림 저장
            if isinstance(data, sc.AnnData):
                data.uns['preprocess_mean_plot'] = fig1
                data.uns['preprocess_sd_plot'] = fig2

        del X, l1_norm, l2_norm, mean, std

        return pd.DataFrame(self.X)
       
    
    def _preprocess_rand(self, X, inplace=True):
        """Preprocessing that does not save data statistics"""
        if not inplace:
            X = X.copy()
    
        # L1 정규화 및 로그 변환
        l1_norm = cp.linalg.norm(X, ord=1, axis=1)
        X = X / l1_norm[:, cp.newaxis]
        X += 1
        X = cp.log(X)
    
        # Z-score 정규화
        mean = cp.mean(X, axis=0)
        std = cp.std(X, axis=0)
        X = (X - mean) / std
    
        # L2 정규화
        l2_norm = cp.linalg.norm(X, ord=2, axis=1)
        X = X / l2_norm[:, cp.newaxis]
        X *= cp.mean(l2_norm)
        X -= cp.mean(X, axis=0)

        del l1_norm, l2_norm, mean, std
        return X

    def fit_transform(self, data=None, eigen_solver='wishart', plot_mp = False):
        #print(self.get_cpu_memory_usage()) 
        #print(self.get_gpu_memory_usage())  
        
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
        
        X = cp.array(self.X, dtype=cp.float64)
       
        pca_result = self._PCA(X, plot_mp = plot_mp)
        self._signal_components = pca_result[1]

        if self.sparsity == 'auto':
            self._calculate_sparsity()
            
        if self.preprocessed:
            raw = cp.array(self._raw, dtype=cp.float64)

        n = min(self._signal_components.shape[1] * self._perturbed_n_scale, self.X.shape[1])
        
        pert_vecs = list()
        for _ in tqdm(range(self.n_rand_matrix), total=self.n_rand_matrix):
            # Construct random matrix
            rand = scipy.sparse.rand(self._raw.shape[0], self._raw.shape[1], 
                                    density=1-self.sparsity, 
                                    format='csr')
            rand.data[:] = 1
            rand = cp.array(rand.toarray())
        
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
        print(self.get_cpu_memory_usage())  # CPU 메모리 사용량 출력
        print(self.get_gpu_memory_usage())  # GPU 메모리 사용량 출력

        self.data.obsm['PCA_scLENS'] = self.X_transform

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
        bin = cp.array(bin.toarray(), dtype=cp.float64)
        Vb = self._PCA_rand(self._preprocess_rand(bin, inplace=False), bin.shape[0])
        n_vbp = Vb.shape[1]//2

        n_buffer = 5
        buffer = [1] * n_buffer
        while sparse > self.sparsity_threshold:
            n_pert = int((1-sparse) * n_len)
            selection = np.random.choice(n_zero,n_pert,replace=False)
            idx = [x[selection] for x in zero_idx]

            # Construct perturbed data matrix
            pert = cp.zeros_like(bin)
            pert[tuple(idx)] = 1
            pert += bin
            
            pert = self._preprocess_rand(pert)
            pert = pert @ pert.T
            pert /= pert.shape[1]
            
            Vbp = cp.linalg.eigh(pert)[1][:, -n_vbp:]
            
            del pert

            corr_arr = cp.max(cp.abs(Vb.T @ Vbp), axis=0).get()
            corr = np.sort(corr_arr)[1]
            
            buffer.pop(0)
            buffer.append(corr)
            

            print(f'Min(corr): {corr}, sparsity: {sparse}, add_ilen: {selection.shape}')
            if all([x < thresh for x in buffer]):
                 self.sparsity = sparse + self.sparsity_step * (n_buffer - 1)
                 break
            
            sparse -= self.sparsity_step
        del bin, Vb
    
    def _PCA(self, X, plot_mp = False):
        pca = PCA(device = self.device)
        pca.fit(X)
        
        if plot_mp:
            pca.plot_mp(comparison = False)
            plt.show()
        comp = pca.get_signal_components()
    
        return comp
    
    def _PCA_rand(self, X, n):
        W = (X @ X.T)
        W /= X.shape[1]
        _, V = cp.linalg.eigh(W)
        V = V[:, -n:]

        del W, _
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

    def get_cpu_memory_usage(self):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return f"CPU Memory Usage: {mem_info.rss / 1024 ** 2:.2f} MB"

    def get_gpu_memory_usage(self):
        device = cp.cuda.Device(0)
        # 메모리 풀 확인
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()

        # 현재 사용 중인 메모리 (bytes)
        used_bytes = mempool.used_bytes() / 1024 ** 2
        total_bytes = mempool.total_bytes() / 1024 ** 2

        return used_bytes, total_bytes
        

    