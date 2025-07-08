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
import ctypes

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.cm as cm

import seaborn as sns

import multiprocessing as mp
from multiprocessing import Process, Queue
import sys
import psutil
import os
import gc
import time
import dask.array as da
from dask import delayed

from .PCA import PCA
from .calc import Calc

class find_optimal_pc():
	def __init__(self, sparsity='auto',
				 sparsity_step=0.001,
				 sparsity_threshold=0.9,
				 perturbed_n_scale=2,
				 device='gpu',
				 n_rand_matrix=20,
				 threshold=np.cos(np.deg2rad(60)),
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
		self.directory = None


	# process 1 : preprocessing
	def _run_in_process(self, target, args=()):

		p = mp.Process(target=target, args=args)
		p.start()
		p.join()
		if p.exitcode != 0:
			raise RuntimeError(f"{target.__name__} failed with exit code {p.exitcode}")
		
	# process 2 : calculate sparsity
	def _run_in_process_value(self, target, args=()):
		queue = Queue()
		def wrapper(queue, *args):
			result = target(*args)
			queue.put(result)
		p = mp.Process(target=wrapper, args=(queue, *args))
		p.start()
		p.join()
		if p.exitcode != 0:
			raise RuntimeError(f"{target.__name__} failed with exit code {p.exitcode}")
		return queue.get()
	

	# preprocessing - filtering
	def filtering(self,
		data,
		min_tp_c=0, min_tp_g=0, max_tp_c=np.inf, max_tp_g=np.inf,
		min_genes_per_cell=200, max_genes_per_cell=0,
		min_cells_per_gene=15, mito_percent=5., ribo_percent=0.
	):

		if isinstance(data, sc.AnnData):
			cell_names = data.obs_names.to_numpy()
			gene_names = data.var_names.to_numpy()

			try:
				data_array = data.raw.X
			except AttributeError:
				data_array = data.X

		# is_sparse = sp.issparse(data_array)
		X = data_array.astype(np.float32)

		n_cell_counts = (X != 0).sum(axis=0)  
		n_cell_sums = X.sum(axis=0)

		bidx_1 = np.array(n_cell_sums > min_tp_g).flatten()
		bidx_2 = np.array(n_cell_sums < max_tp_g).flatten()
		bidx_3 = np.array(n_cell_counts >= min_cells_per_gene).flatten()

		self.fg_idx = bidx_1 & bidx_2 & bidx_3  

		n_gene_counts = (X != 0).sum(axis=1) 
		n_gene_sums = X.sum(axis=1)

		cidx_1 = np.array(n_gene_sums > min_tp_c).flatten()
		cidx_2 = np.array(n_gene_sums < max_tp_c).flatten()
		cidx_3 = np.array(n_gene_counts >= min_genes_per_cell).flatten()

		mito_mask = pd.Series(gene_names).str.contains(r"^MT-", case=False, regex=True).to_numpy()
		if mito_percent == 0:
			cidx_4 = np.ones_like(cidx_1, dtype=bool)
		else:
			mito_sum = X[:, mito_mask].sum(axis=1)
			total_sum = X.sum(axis=1)
			mito_ratio = np.array(mito_sum / total_sum).flatten()
			cidx_4 = mito_ratio < (mito_percent / 100)

		ribo_mask = pd.Series(gene_names).str.contains(r"^RP[SL]", case=False, regex=True).to_numpy()
		if ribo_percent == 0:
			cidx_5 = np.ones_like(cidx_1, dtype=bool)
		else:
			ribo_sum = X[:, ribo_mask].sum(axis=1)
			total_sum = X.sum(axis=1)
			ribo_ratio = np.array(ribo_sum / total_sum).flatten()
			cidx_5 = ribo_ratio < (ribo_percent / 100)

		if max_genes_per_cell == 0:
			cidx_6 = np.ones_like(cidx_1, dtype=bool)
		else:
			cidx_6 = n_gene_counts < max_genes_per_cell

		self.fc_idx = cidx_1 & cidx_2 & cidx_3 & cidx_4 & cidx_5 & cidx_6  # cell index

		if self.fc_idx.sum() > 0 and self.fg_idx.sum() > 0:
			Xf = X[self.fc_idx][:, self.fg_idx]

			
			valid_gene_mask = np.array(Xf.sum(axis=0) != 0).flatten()
			Xf = Xf[:, valid_gene_mask]
			self.final_gene_names = gene_names[self.fg_idx][valid_gene_mask]
			self.final_cell_names = cell_names[self.fc_idx]

			if sp.issparse(Xf):
				self._raw = Xf
			else:
				self._raw = pd.DataFrame(Xf, index=self.final_cell_names, columns=self.final_gene_names)
				self._raw = sp.csr_matrix(self._raw)

			raw_anndata = sc.AnnData(self._raw)
			raw_anndata.write_zarr(f"{self.directory}/raw_anndata.zarr")

			print(f"After filtering >> shape: {self._raw.shape}")

			raw_anndata.X = None
			del raw_anndata

			return self._raw
		else:
			print("There is no high quality cells and genes")
			return None


	# preprocessing - normalizing
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
	

	def preprocess_stage(self, data, filter, plot=False):

		if filter ==True:  
			if isinstance(data, pd.DataFrame):
				obs = pd.DataFrame(data['cell']) 
				X = sp.csr_matrix(data.iloc[:, 1:].values) 
				var = pd.DataFrame(data.columns[1:])
				var.columns = ['gene'] 
				data = sc.AnnData(X, obs=obs, var=var)

			# if isinstance(data, sc.AnnData):
			#     cell_names = data.obs_names.to_numpy()
			#     gene_names = data.var_names.to_numpy()

			#     try:
			#         data_array = data.raw.X 
			#     except AttributeError:
			#         data_array = data.X
				
			self._raw = self.filtering(data) # sparse

			if sp.issparse(self._raw):
				normalized_X = self.normalize(self._raw.toarray())
			else:
				normalized_X = self.normalize(self._raw)
			normalized_X.to_zarr(f"{self.directory}/normalized_X.zarr")

			self.fg_idx = np.where(np.isin(data.var_names, self.final_gene_names))[0]
			self.fc_idx = np.where(np.isin(data.obs_names, self.final_cell_names))[0]

			data = data[self.fc_idx, self.fg_idx]
			data.var_names = self.final_gene_names
			data.obs_names = self.final_cell_names
		
		else:
			print('without filtering')
			self._raw = data.X # scanpy로 filtering
			if sp.issparse(self._raw):
				normalized_X = self.normalize(self._raw.toarray())
			else:
				normalized_X = self.normalize(self._raw)
			normalized_X.to_zarr(f"{self.directory}/normalized_X.zarr")


		# if sp.issparse(self._raw):
		#     normalized_X = self.normalize(self._raw.toarray())
		# else:
		#     normalized_X = self.normalize(self._raw)
		# normalized_X.to_zarr(f"{self.directory}/normalized_X.zarr")

		# data = data[self.fc_idx, self.fg_idx]
		# data.var_names = self.final_gene_names
		# data.obs_names = self.final_cell_names
	
		if plot:
			print("plotting the result of preprocessing")
			normalized_X = da.from_zarr(f"{self.directory}/normalized_X.zarr")
			self.X = normalized_X.compute()

			_raw = self._raw.toarray()
			raw_mean = np.mean(_raw, axis=1)  
			raw_std = np.std(_raw, axis=0)    
			X_mean = np.mean(self.X, axis=1) 
			X_std = np.std(self.X, axis=0)

			df_mean = pd.DataFrame({
				'raw_mean': raw_mean,
				'X_mean': X_mean
			})

			df_std = pd.DataFrame({
				'raw_std': raw_std,
				'X_std': X_std
			})
	
			fig1, axs1 = plt.subplots(1, 2, figsize=(10, 5))
			axs1[0].hist(raw_mean, bins=100)
			axs1[1].hist(X_mean, bins=100)
			fig1.suptitle('Mean of Gene Expression along Cells')

			fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5))
			axs2[0].hist(raw_std, bins=100)
			axs2[1].hist(X_std, bins=100)
			fig2.suptitle('SD of Gene Expression for each Gene')
			plt.show()
			
		self.data = data
		self.data.write_zarr(f"{self.directory}/preprocessed_anndata.zarr")
		self.preprocessed = True

		data.X = None
		data.raw = None
		data.obsm.clear()
		data.varm.clear()
		data.layers.clear()
		data.uns.clear()

		del data
		del normalized_X
		gc.collect()

	def preprocess(self, data, filter=False, plot=False):
		self._run_in_process(self.preprocess_stage, args=(data, filter, plot))


	def _preprocess_rand(self, X, inplace=True, chunk_size = 'auto'):
		if not inplace:
			X = X.copy()
		if sp.issparse(X): 
			X = X.toarray() 
		X = self.normalize(X)
		return X.compute()
	
	def fit_transform(self, data=None, device=None, eigen_solver='wishart', plot_mp = False):

		_path = f"{self.directory}/preprocessed_anndata.zarr"
		if os.path.exists(_path):
			self.data = anndata.read_zarr(f"{self.directory}/preprocessed_anndata.zarr")
			self._raw = anndata.read_zarr(f"{self.directory}/raw_anndata.zarr").X
			# self._raw = raw_anndata.X

		else:
			if isinstance(data, pd.DataFrame):
				obs = pd.DataFrame(data['cell']) 
				X = sp.csr_matrix(data.iloc[:, 1:].values) 
				var = pd.DataFrame(data.columns[1:])
				var.columns = ['gene'] 
				self.data = sc.AnnData(X, obs=obs, var=var)
				self._raw = X
				self.X = X

			elif isinstance(data, sc.AnnData):
				# cell_names = data.obs_names.to_numpy()
				# gene_names = data.var_names.to_numpy()
				# data_array = data.X
				self.data = data
				self.X = data.X
				self._raw = data.X
				
			else:
				raise ValueError("Data must be a pandas DataFrame or Anndata")
		

		# calculate sparsity
		if self.sparsity == 'auto':
			self.sparsity = self._run_in_process_value(self._calculate_sparsity)


		# RMT
		self.X = da.from_zarr(f"{self.directory}/normalized_X.zarr")
		pca_result = self._PCA(self.X, device = None, plot_mp = plot_mp)

		if self.X.shape[0] <= self.X.shape[1]:
			self._signal_components = pca_result[1]
			self.eigenvalue = pca_result[0]
		else:
			eigenvalue = cp.asnumpy(pca_result[0])
			_signal_components = cp.asnumpy(pca_result[1])

			re = self.X @ _signal_components @ np.diag(1/np.sqrt(eigenvalue))
			re /= np.sqrt(self.X.shape[1])
			self._signal_components = cp.asarray(re)
			self.eigenvalue = cp.asarray(eigenvalue)

			del _signal_components, re
			gc.collect()
			cp._default_memory_pool.free_all_blocks()


		# SRT
		class CSRMatrix(ctypes.Structure):
			_fields_ = [
				("indptr", ctypes.POINTER(ctypes.c_int)),
				("indices", ctypes.POINTER(ctypes.c_int)),
				("data", ctypes.POINTER(ctypes.c_float)),
				("nnz", ctypes.c_int),
				("n_rows", ctypes.c_int),
				("n_cols", ctypes.c_int),
			]
		module_dir = os.path.dirname(os.path.abspath(__file__))
		lib_path = os.path.join(module_dir, "random_matrix.so")
		lib = ctypes.CDLL(lib_path)
		lib.sparse_rand_csr.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double]
		lib.sparse_rand_csr.restype = ctypes.POINTER(CSRMatrix)
		lib.free_csr.argtypes = [ctypes.POINTER(CSRMatrix)]

		n_rows, n_cols = self._raw.shape
		density = 1 - self.sparsity

		self.pert_vecs = list()
		for _ in tqdm(range(self.n_rand_matrix), total=self.n_rand_matrix):

			mat_ptr = lib.sparse_rand_csr(n_rows, n_cols, density)
			if not bool(mat_ptr):
				raise MemoryError("Failed to allocate CSR matrix.")
			mat = mat_ptr.contents

			# Convert to scipy CSR
			indptr = np.ctypeslib.as_array(mat.indptr, shape=(mat.n_rows + 1,))
			indices = np.ctypeslib.as_array(mat.indices, shape=(mat.nnz,))
			data = np.ctypeslib.as_array(mat.data, shape=(mat.nnz,))
			rand = sp.csr_matrix((data, indices, indptr), shape=(mat.n_rows, mat.n_cols)).copy()

			block_size = 10000
			shape = rand.shape
			rand_zarr_path = f"./{self.directory}/srt_perturbed.zarr"
			zarr_out = zarr.open(rand_zarr_path, mode="w", shape=shape, dtype=np.float32, chunks=(block_size, shape[1]))

			for i in range(0, shape[0], block_size):
				end = min(i + block_size, shape[0])
				block = (rand[i:end] + self._raw[i:end]).toarray()
				zarr_out[i:end] = block

			rand = da.from_zarr(f"./{self.directory}/srt_perturbed.zarr")
			rand = self._preprocess_rand(rand)
			
			n = min(self._signal_components.shape[1] * self._perturbed_n_scale, self.X.shape[1])

			if self.device == 'cpu': 
				perturbed_L, perturbed_V = self._PCA_rand(rand, n, self.device)
			elif self.device == 'gpu':
				gb = self.estimate_matrix_memory(rand.shape, step='pca_rand')
				strategy = self.calculate_gpu_memory(gb, step = 'pca_rand') # cupy or dask
				perturbed_L, perturbed_V = self._PCA_rand(rand, n, strategy)

				gb = self.estimate_matrix_memory(self._signal_components.shape, step='srt') 
				strategy = self.calculate_gpu_memory(gb*20, step = 'srt') # gpu or cpu

			else:
				raise ValueError("The device must be either 'cpu' or 'gpu'.")


			if self.device == 'cpu':
				if rand.shape[0] <= rand.shape[1]:
					perturbed = perturbed_V
				else:
					perturbed_L = perturbed_L[-n:] 
					re = rand @ perturbed_V @ np.diag(1/np.sqrt(perturbed_L))
					re /= np.sqrt(rand.shape[1])
					perturbed = re

					del perturbed_L, perturbed_V, re
					gc.collect()

				pert_select = self._signal_components.T @ perturbed
				pert_select = np.abs(pert_select)
				pert_select = np.argmax(pert_select, axis = 1)
				self.pert_vecs.append(perturbed[:, pert_select])

				del rand, perturbed, pert_select
				gc.collect()


			elif self.device == 'gpu':
				if strategy == 'cpu':
					
					if rand.shape[0] <= rand.shape[1]:
						perturbed = perturbed_V
					else:
						perturbed_L = perturbed_L[-n:] 
						perturbed_V = cp.asnumpy(perturbed_V)  
						perturbed_L = cp.asnumpy(perturbed_L)
						re = rand @ perturbed_V @ np.diag(1/np.sqrt(perturbed_L))
						re /= np.sqrt(rand.shape[1])
						perturbed = re

						del perturbed_L, perturbed_V, re
						gc.collect()
						cp._default_memory_pool.free_all_blocks()

					if isinstance(self._signal_components, np.ndarray):
						self._signal_components = self._signal_components
					else:
						self._signal_components = self._signal_components.get()

					pert_select = self._signal_components.T @ perturbed
					pert_select = np.abs(pert_select)
					pert_select = np.argmax(pert_select, axis = 1)
					self.pert_vecs.append(perturbed[:, pert_select])

					del rand, perturbed, pert_select
					gc.collect()
					cp.get_default_memory_pool().free_all_blocks()
					cp.get_default_pinned_memory_pool().free_all_blocks()
					cp._default_memory_pool.free_all_blocks()

				elif strategy == 'gpu': 
					
					if rand.shape[0] <= rand.shape[1]:
						perturbed = cp.asarray(perturbed_V)
					else:
						perturbed_L = perturbed_L[-n:]
						perturbed_V = cp.asnumpy(perturbed_V)
						perturbed_L = cp.asnumpy(perturbed_L)
						re = rand @ perturbed_V @ np.diag(1/np.sqrt(perturbed_L))
						re /= np.sqrt(rand.shape[1])
						perturbed = cp.asarray(re)

						del perturbed_L, perturbed_V, re
						gc.collect()
						cp._default_memory_pool.free_all_blocks()
					
					if isinstance(self._signal_components, np.ndarray):
						self._signal_components = cp.asarray(self._signal_components)

					pert_select = self._signal_components.T @ perturbed
					pert_select = cp.abs(pert_select)
					pert_select = cp.argmax(pert_select, axis = 1)
					self.pert_vecs.append(perturbed[:, pert_select])

					del rand, perturbed, pert_select
					gc.collect()
					cp.get_default_memory_pool().free_all_blocks()
					cp.get_default_pinned_memory_pool().free_all_blocks()
					cp._default_memory_pool.free_all_blocks()

				else:
					raise ValueError("The device must be either 'cpu' or 'gpu'.")

		if self.device == 'gpu':
			pert_scores = list()

			if strategy == 'cpu':
				for i in range(self.n_rand_matrix):
					for j in range(i+1, self.n_rand_matrix):
						dots = self.pert_vecs[i].T @ self.pert_vecs[j]
						corr = np.max(np.abs(dots), axis=1)
						pert_scores.append(corr)
			else:  # strategy == gpu
				for i in range(self.n_rand_matrix):
					for j in range(i+1, self.n_rand_matrix):
						dots = self.pert_vecs[i].T @ self.pert_vecs[j]
						corr = cp.max(cp.abs(dots), axis=1)
						pert_scores.append(corr)

			pert_scores = cp.array(pert_scores)

			def iqr(x):
				q1 = cp.percentile(x, 25)
				q3 = cp.percentile(x, 75)
				iqr = q3 - q1
				filtered = x[(x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)]
				return cp.median(filtered)

			rob_scores = cp.array([iqr(pert_scores[:, i]) for i in range(pert_scores.shape[1])])
			robust_idx = rob_scores > self.threshold
			self._robust_idx = robust_idx

			if isinstance(self._signal_components, np.ndarray):
				self._signal_components = cp.asarray(self._signal_components)

			_ = self._signal_components[:, self._robust_idx] * cp.sqrt(self.eigenvalue[self._robust_idx]).reshape(1, -1)

		elif self.device == 'cpu':
			pert_scores = []
			for i in range(self.n_rand_matrix):
				for j in range(i+1, self.n_rand_matrix):
					dots = self.pert_vecs[i].T @ self.pert_vecs[j]
					corr = np.max(np.abs(dots), axis=1)
					pert_scores.append(corr.get())

			pert_scores = np.array(pert_scores)

			def iqr(x):
				q1 = np.percentile(x, 25)
				q3 = np.percentile(x, 75)
				iqr = q3 - q1
				filtered = x[(x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)]
				return np.median(filtered)

			rob_scores = np.array([iqr(pert_scores[:, i]) for i in range(pert_scores.shape[1])])
			robust_idx = rob_scores > self.threshold
			self._robust_idx = robust_idx

			_ = self._signal_components[:, self._robust_idx] * np.sqrt(self.eigenvalue[self._robust_idx]).reshape(1, -1)

		else:
			raise ValueError("The device must be either 'cpu' or 'gpu'.")

		if isinstance(self._robust_idx, cp.ndarray):
			robust_idx_np = self._robust_idx.get()
		else:
			robust_idx_np = self._robust_idx

		self.data.uns['sclens_optimal_pc_count'] = int(np.sum(robust_idx_np))
		self.data.uns['sclens_robust_idx'] = robust_idx_np
		self.data.uns['sclens_eigenvalues'] = self.eigenvalue.get() if isinstance(self.eigenvalue, cp.ndarray) else self.eigenvalue

		return self.data.uns['sclens_optimal_pc_count']
	

	def _calculate_sparsity(self):
		bin_matrix = sp.csr_matrix(
			(np.ones_like(self._raw.data, dtype=np.float32),
			self._raw.indices,
			self._raw.indptr),
			shape=self._raw.shape)

		bin_dask = da.from_array(bin_matrix.toarray(), chunks=(10000, bin_matrix.shape[1]))
		bin_dask.to_zarr(f"{self.directory}/bin.zarr")

		sparse = 0.999
		shape_row, shape_col = self._raw.shape
		n_len = shape_row * shape_col
		n_zero = n_len - self._raw.size

		rng = np.random.default_rng()

		# Calculate threshold for correlation
		n_sampling = min(self._raw.shape)
		thresh = np.mean([max(np.abs(rng.normal(0, np.sqrt(1 / n_sampling), n_sampling)))
						for _ in range(5000)]).item()
		print(f'sparsity_th: {thresh}')

		zero_indices_dict = {}
		for row in range(shape_row):
			col = self._raw[row, :]  # CSR 형식 (1, shape_col)
			zero_indices = np.setdiff1d(np.arange(shape_col), col.indices, assume_unique=True).astype(np.int32)
			if zero_indices.size > 0:
				zero_indices_dict[row] = zero_indices  # 0이 있는 row만 저장
		del col, zero_indices
		gc.collect()

		c_array_pointers = (ctypes.POINTER(ctypes.c_int32) * len(zero_indices_dict))()
		for i, key in enumerate(zero_indices_dict):
			c_array_pointers[i] = zero_indices_dict[key].ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

		row_sizes = []
		for i in range(len(zero_indices_dict)):
			row_sizes.append(len(zero_indices_dict[i]))
		row_sizes = np.array(row_sizes, dtype=np.int32)

		bin = da.from_zarr(f'{self.directory}/bin.zarr')

		bin_nor = self.normalize(bin)
		bin_nor = bin_nor.compute()

		if self.device == 'cpu':
			_, Vb = self._PCA_rand(bin_nor, bin.shape[0], self.device)

		elif self.device == 'gpu':
			gb = self.estimate_matrix_memory(bin.shape, step='pca_rand')
			strategy = self.calculate_gpu_memory(gb, step='pca_rand')  # cupy or dask
			_, Vb = self._PCA_rand(bin_nor, bin.shape[0], strategy)
			if isinstance(Vb, np.ndarray):
				strategy = 'cpu'
		else:
			raise ValueError("The device must be either 'cpu' or 'gpu'.")

		del bin_nor
		gc.collect()
		cp._default_memory_pool.free_all_blocks()

		n_vbp = Vb.shape[1] // 2
		n_buffer = 5
		buffer = [1] * n_buffer

		while sparse > self.sparsity_threshold:
			n_pert = int((1 - sparse) * n_len)
			p = n_pert / n_zero
			rows, cols = self._raw.shape

			module_dir = os.path.dirname(os.path.abspath(__file__))
			lib_path = os.path.join(module_dir, "perturb_omp.so")
			lib = ctypes.CDLL(lib_path)

			lib.perturb_zeros.argtypes = [
				ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # zero_list
				ctypes.POINTER(ctypes.c_int),  # row_sizes
				ctypes.c_int,  # rows
				ctypes.c_double,  # p
				ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),  # output_rows
				ctypes.POINTER(ctypes.c_int),  # output_sizes
			]

			zero_list_ptr = c_array_pointers
			row_sizes_ptr = row_sizes.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
			output_rows = (ctypes.POINTER(ctypes.c_int) * rows)()
			output_sizes = (ctypes.c_int * rows)()

			lib.perturb_zeros(zero_list_ptr, row_sizes_ptr, rows, p, output_rows, output_sizes)

			row_idx = []
			col_idx = []

			for i in range(rows):
				size = output_sizes[i]
				if size > 0:
					for j in range(size):
						row_idx.append(i)
						col_idx.append(output_rows[i][j])

			data = [1] * len(row_idx)
			pert = sp.coo_matrix((data, (row_idx, col_idx)), shape=(rows, cols)).tocsr()

			bin_sparse = bin_matrix
			shape = pert.shape
			block_size = 10000

			zarr_path = f"{self.directory}/perturbed.zarr"
			zarr_out = zarr.open(zarr_path, mode="w", shape=shape, dtype=np.float32, chunks=(block_size, shape[1]))

			for i in range(0, shape[0], block_size):
				end = min(i + block_size, shape[0])
				block = (pert[i:end] + bin_sparse[i:end]).toarray()
				zarr_out[i:end] = block

			pert = da.from_zarr(f"{self.directory}/perturbed.zarr")
			pert = self.normalize(pert).compute()

			if self.device == 'cpu' or strategy == 'cpu':
				_, Vbp = self._PCA_rand(pert, n_vbp, self.device)

			elif self.device == 'gpu':
				gb = self.estimate_matrix_memory(pert.shape, step='pca_rand')
				if strategy == 'dask' or strategy == 'cupy':
					strategy = self.calculate_gpu_memory(gb, step='pca_rand')
				_, Vbp = self._PCA_rand(pert, n_vbp, strategy)
			else:
				raise ValueError("The device must be either 'cpu' or 'gpu'.")

			del pert
			gc.collect()
			cp._default_memory_pool.free_all_blocks()

			if self.device == 'cpu' or strategy == 'cpu':
				if isinstance(Vb, cp.ndarray):
					Vb = Vb.get()
					cp.get_default_memory_pool().free_all_blocks()
					cp.get_default_pinned_memory_pool().free_all_blocks()
					cp._default_memory_pool.free_all_blocks()

				corr_arr = np.max(np.abs(Vb.T @ Vbp), axis=0)
			elif self.device == 'gpu':
				try:
					if isinstance(Vbp, np.ndarray):
						Vbp = cp.asarray(Vbp)
						corr_arr = cp.max(cp.abs(Vb.T @ Vbp), axis=0).get()
						strategy = 'cpu'
					else:
						corr_arr = cp.max(cp.abs(Vb.T @ Vbp), axis=0).get()
				except cp.cuda.memory.OutOfMemoryError:
					Vb = np.asarray(Vb)
					corr_arr = np.max(np.abs(Vb.T @ Vbp), axis=0)

			corr = np.sort(corr_arr)[1]
			buffer.pop(0)
			buffer.append(corr)
			print(f'Min(corr): {corr}, sparsity: {sparse}')
			if all([x < thresh for x in buffer]):
				self.sparsity = sparse + self.sparsity_step * (n_buffer - 1)
				break
			sparse -= self.sparsity_step

		del bin
		gc.collect()
		cp._default_memory_pool.free_all_blocks()

		return self.sparsity


		
	def _PCA(self, X, device = None, plot_mp = False):
		pca = PCA(device = self.device)
		pca.fit(X)
		
		if plot_mp:
			pca.plot_mp(comparison = False)
			plt.show()
		
		comp = pca.get_signal_components()

		del pca
		gc.collect()
		cp.get_default_memory_pool().free_all_blocks()
		cp.get_default_pinned_memory_pool().free_all_blocks()
		cp._default_memory_pool.free_all_blocks()
	
		return comp
	

	def _PCA_rand(self, X, n, strategy): # strategy = cupy, dask, cpu
		pca = PCA(device = self.device)

		if strategy == 'cupy':
			X = cp.asarray(X)
			Y = pca._wishart_matrix(X)
			L, V = cp.linalg.eigh(Y)

			del Y
			cp._default_memory_pool.free_all_blocks() 
			
		elif strategy == 'dask':
			L, V = pca._get_eigen(X)
		
		else:
			Y = pca._wishart_matrix(X)
			L, V = np.linalg.eigh(Y)

			del Y
			gc.collect()

		V = V[:, -n:]

		
		gc.collect()
		cp.get_default_memory_pool().free_all_blocks()
		cp.get_default_pinned_memory_pool().free_all_blocks()
		cp._default_memory_pool.free_all_blocks() 

		return L, V

	def estimate_matrix_memory(self, tuple, step):
		dtype = 4
		row = tuple[0]
		col = tuple[1]

		if step == 'srt':
			bytes_total = row * col * dtype
			gb = bytes_total / 1024**3

		elif step == 'pca_rand':
			gb = 0
			# pert
			bytes_total = row * col * dtype
			gb += bytes_total / 1024**3

			# Wishart matrix (X @ X.T 또는 X.T @ X)
			wishart_dim = min(row, col)
			num_elements = wishart_dim ** 2
			gb += (num_elements * dtype) / (1024 ** 3)

			# EVD 결과 (L + V)
			num_elements = wishart_dim + wishart_dim ** 2
			gb += (num_elements * dtype) / (1024 ** 3)

			# 워크스페이스
			gb += 8
		
		return gb

	def calculate_gpu_memory(self, gb, step):
		free_mem, total_mem = cp.cuda.runtime.memGetInfo()
		total_gb = total_mem / (1024 ** 3)
		free_gb = free_mem / (1024 ** 3)

		if step == 'srt':
			if gb < free_gb*0.9:
				strategy = 'gpu'
			else:
				strategy = 'cpu'

		elif step == 'pca_rand':
			if gb < free_gb*0.9:
				strategy = 'cupy'
			else:
				strategy = 'dask'
		else:
			pass

		return strategy
	

	
