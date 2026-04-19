import scanpy as sc
try:
    import cupy as cp
except ImportError:
    raise ImportError(
        "CuPy is required but not installed. "
        "Please install CuPy manually: `pip install cupy-cuda11x` or `pip install cupy-cuda12x`"
    )
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import vstack
import numpy as np
from tqdm.auto import tqdm
import zarr 
import anndata
import ctypes

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import multiprocess as mp
from multiprocess import Queue
import shutil
import os
import gc
import glob
import dask.array as da
import tempfile
import uuid
from dask.distributed import Client
import time
from numcodecs import Blosc
import numexpr as ne

from .PCA import PCA


def _get_tempdir(p):
    return tempfile.gettempdir() if p is None else p    


def _find_compiled_library(library_name):
    '''
    Find the compiled shared library, checking multiple possible locations.
    
    Args:
        library_name (str): Name of the library (e.g., 'random_matrix', 'perturb_omp')
    
    Returns:
        str: Path to the found library
        
    Raises:
        FileNotFoundError: If the library cannot be found
    '''
    import platform
    
    # Get the current module directory
    module_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Determine the shared library extension based on platform
    if platform.system() == "Windows":
        lib_ext = ".dll"
    elif platform.system() == "Darwin":
        lib_ext = ".dylib"
    else:  # Linux and other Unix-like systems
        lib_ext = ".so"
    
    # List of possible locations to search for the library
    search_paths = [
        # Direct in module directory (original location)
        os.path.join(module_dir, f"{library_name}{lib_ext}"),
        # Meson build output locations
        os.path.join(module_dir, "..", "..", "build", "src", "stLENS", f"{library_name}{lib_ext}"),
        os.path.join(module_dir, "..", "..", "builddir", "src", "stLENS", f"{library_name}{lib_ext}"),
        # Installation locations
        os.path.join(module_dir, f"lib{library_name}{lib_ext}"),
        # Check if installed via pip/conda in site-packages
        os.path.join(os.path.dirname(module_dir), "stLENS", f"{library_name}{lib_ext}"),
        os.path.join(os.path.dirname(module_dir), "stLENS", f"lib{library_name}{lib_ext}"),
    ]
    
    # Also check in the same directory with different naming conventions
    search_paths.extend([
        os.path.join(module_dir, f"lib{library_name}{lib_ext}"),
        os.path.join(module_dir, f"{library_name}.cpython-*{lib_ext}"),
    ])
    
    # Search for the library
    for lib_path in search_paths:
        # Handle glob pattern for cpython naming
        if "*" in lib_path:
            import glob
            matches = glob.glob(lib_path)
            if matches:
                lib_path = matches[0]  # Take the first match
        
        if os.path.exists(lib_path):
            return lib_path
    
    # If not found, raise an informative error
    raise FileNotFoundError(
        f"Could not find compiled library '{library_name}'. "
        f"Searched in: {search_paths}. "
        f"Please ensure the C++ extensions were compiled properly during installation."
    )    

class stLENS():
    def __init__(self, sparsity='auto',
                 sparsity_step=0.001,
                 sparsity_threshold=0.9,
                 perturbed_n_scale=2,
                 n_rand_matrix=20,
                 threshold=np.cos(np.deg2rad(60)),
                 ): 
        self.sparsity = sparsity
        self.sparsity_threshold = sparsity_threshold
        self.sparsity_step = sparsity_step
        self._perturbed_n_scale = perturbed_n_scale
        self.n_rand_matrix = n_rand_matrix
        self.threshold = threshold
        mp.set_start_method('spawn')

    @staticmethod
    def _select_best_gpu():
        n_devices = cp.cuda.runtime.getDeviceCount()
        if n_devices <= 1:
            return 0
        best_device = 0
        best_free = 0
        for d in range(n_devices):
            with cp.cuda.Device(d):
                free, _ = cp.cuda.runtime.memGetInfo()
            if free > best_free:
                best_free = free
                best_device = d
        return best_device

    @staticmethod
    def _gpu_device_order():
        """Return device ids in order: current primary first, then others by free memory desc."""
        n_devices = cp.cuda.runtime.getDeviceCount()
        primary = cp.cuda.Device().id
        if n_devices <= 1:
            return [primary]
        others = []
        for d in range(n_devices):
            if d == primary:
                continue
            with cp.cuda.Device(d):
                free, _ = cp.cuda.runtime.memGetInfo()
            others.append((free, d))
        others.sort(reverse=True)
        return [primary] + [d for _, d in others]

    @staticmethod
    def _to_current_device(arr, dtype=None):
        """Ensure arr is on the currently-active cupy device.
        Handles numpy inputs and cross-device cupy arrays (via CPU roundtrip)."""
        if isinstance(arr, np.ndarray):
            return cp.asarray(arr, dtype=dtype) if dtype else cp.asarray(arr)
        # cupy array
        if arr.device.id == cp.cuda.Device().id:
            return arr.astype(dtype, copy=False) if dtype else arr
        # cross-device: copy via CPU
        tmp = cp.asnumpy(arr)
        return cp.asarray(tmp, dtype=dtype) if dtype else cp.asarray(tmp)

    # process 1 : preprocessing
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
    def filter_cells_and_genes(self,
        data, # anndata
        min_tp_c=0, min_tp_g=0, max_tp_c=np.inf, max_tp_g=np.inf,
        min_genes_per_cell=200, max_genes_per_cell=0,
        min_cells_per_gene=15, mito_percent=5., ribo_percent=0.,
        inplace=True, use_raw=True
    ):
        """
        Filter cells and genes in the AnnData object using the scLENS approach.
        
        Parameters
        ----------
        data : anndata.AnnData
            Input AnnData object containing the data to be filtered.
        min_tp_c : int, optional
            Minimum total counts per cell. Default is 0.
        min_tp_g : int, optional
            Minimum total counts per gene. Default is 0.
        max_tp_c : int, optional
            Maximum total counts per cell. Default is np.inf.
        max_tp_g : int, optional
            Maximum total counts per gene. Default is np.inf.
        min_genes_per_cell : int, optional
            Minimum number of genes per cell. Default is 200.
        max_genes_per_cell : int, optional
            Maximum number of genes per cell. Default is 0.
        min_cells_per_gene : int, optional
            Minimum number of cells expressing each gene. Default is 15.
        mito_percent : float, optional
            Upper threshold for mitochondrial gene expression as a percentage of total cell expression. Default is 5.0.
        ribo_percent : float, optional
            Upper threshold for ribosomal gene expression as a percentage of total cell expression. Default is 0.0.
        inplace : bool, optional
            If True, modifies the input AnnData object directly. If False, returns a new AnnData object.
        use_raw : bool, optional
            If True, uses the raw attribute of the AnnData object. Default is True.

        Returns
        -------
        data_filtered : anndata.AnnData or None
            If inplace is True, returns None. If False, returns the filtered AnnData object.
        """
        is_anndata = True
        if isinstance(data, pd.DataFrame):
            is_anndata = False
            if inplace:
                print("Warning: input data is not AnnData - inplace will not work!")
            inplace = True
            obs = pd.DataFrame(data['cell'])
            X = sp.csr_matrix(data.iloc[:, 1:].values)
            var = pd.DataFrame(data.columns[1:])
            var.columns = ['gene']
            data = sc.AnnData(X, obs=obs, var=var)

        use_raw = hasattr(data.raw, 'X')

        cell_names = data.obs_names.to_numpy()
        gene_names = data.var_names.to_numpy()

        if use_raw:
            data_array = data.raw.X
        else:
            data_array = data.X

        X = data_array.astype(np.float32)

        n_cell_counts = (X != 0).sum(axis=0)
        n_cell_sums = X.sum(axis=0)

        bidx_1 = np.array(n_cell_sums > min_tp_g).flatten()
        bidx_2 = np.array(n_cell_sums < max_tp_g).flatten()
        bidx_3 = np.array(n_cell_counts >= min_cells_per_gene).flatten()

        fg_idx = bidx_1 & bidx_2 & bidx_3

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

        fc_idx = cidx_1 & cidx_2 & cidx_3 & cidx_4 & cidx_5 & cidx_6

        if fc_idx.sum() > 0 and fg_idx.sum() > 0:
            if inplace:
                data._inplace_subset_obs(fc_idx)
                data._inplace_subset_var(fg_idx)

                xsum = data.X.sum(axis=0)
                valid_gene_mask = np.array(xsum != 0).flatten()
                data._inplace_subset_var(valid_gene_mask)
                
                if not is_anndata:
                    data_filtered = data

            else:
                data_filtered = data.copy()
                data_filtered._inplace_subset_obs(fc_idx)
                data_filtered._inplace_subset_var(fg_idx)

                xsum = data_filtered.X.sum(axis=0)
                valid_gene_mask = np.array(xsum != 0).flatten()
                data_filtered._inplace_subset_var(valid_gene_mask)

            if inplace and is_anndata:
                print(f"After filtering >> shape: {data.shape}")
                return None
            else:
                print(f"After filtering >> shape: {data_filtered.shape}")
                return data_filtered
        else:
            print("There is no high quality cells and genes")
            if inplace and is_anndata:
                return None
            else:
                return data


    # preprocessing - normalizing
    def _normalize(self, _raw):
            
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

    def _normalize_inplace(self, X):
        if X.nbytes < 1024 ** 3:
            return self._normalize_inplace_np(X)
        return self._normalize_inplace_ne(X)

    def _normalize_inplace_np(self, X):
        l1 = np.linalg.norm(X, ord=1, axis=1, keepdims=True)
        X /= l1
        del l1

        np.log1p(X, out=X)

        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X -= mean
        X /= std
        del mean, std

        l2 = np.linalg.norm(X, ord=2, axis=1, keepdims=True)
        scale = l2.mean()
        X /= l2
        X *= scale
        del l2

        X -= X.mean(axis=0)
        return X

    def _normalize_inplace_ne(self, X):
        l1 = np.linalg.norm(X, ord=1, axis=1, keepdims=True)
        ne.evaluate("X / l1", out=X)
        del l1

        ne.evaluate("log1p(X)", out=X)

        mean = X.mean(axis=0)
        std = X.std(axis=0)
        ne.evaluate("(X - mean) / std", out=X)
        del mean, std

        l2 = np.linalg.norm(X, ord=2, axis=1, keepdims=True)
        scale = l2.mean()
        ne.evaluate("(X / l2) * scale", out=X)
        del l2

        mean2 = X.mean(axis=0)
        ne.evaluate("X - mean2", out=X)
        del mean2

        return X

    def _normalize_process(self, data, tmp_dir):
        raw_X = data.X
        if sp.issparse(raw_X):
            normalized_X = self._normalize(raw_X.toarray())
        else:
            normalized_X = self._normalize(raw_X)

        tmp_prefix = uuid.uuid4()
        normalized_X.to_zarr(f"{tmp_dir}/{tmp_prefix}-normalized_X.zarr")
        return da.from_zarr(f"{tmp_dir}/{tmp_prefix}-normalized_X.zarr")
    
    def _as_dask_array(self, X, path):
        if sp.issparse(X):
            zarr_out = zarr.open(path, mode="w", shape=X.shape, dtype=np.float32, chunks=(10000, X.shape[1]))

            for i in range(0, X.shape[0], 10000):
                end = min(i + 10000, X.shape[0])
                block = X[i:end].toarray()
                zarr_out[i:end] = block

            return da.from_zarr(zarr_out)
        else:
            return da.from_array(X, chunks=(10000, X.shape[1]))

    
    def _normalize_mp(self, adata, in_store, out_store):
        X_da = self._as_dask_array(adata.X, in_store)
        Xn = self._normalize(X_da)
        Xn.to_zarr(out_store)
        
    def _preprocess_rand(self, X, inplace=True, chunk_size = 'auto'):
        if not inplace:
            X = X.copy()
        if sp.issparse(X): 
            X = X.toarray() 
        X = self._normalize(X)
        return X.compute()
    
    def pca(self, adata, device='gpu'):
        """
        Perform PCA on the given AnnData object.

        Parameters
        ----------
        adata : anndata.AnnData
            Input AnnData object containing the data to be transformed.
        device : str, optional
            Device to use for computations, either 'cpu' or 'gpu'. Default is 'gpu'.
        Returns
        -------
        adata : anndata.AnnData
        """

        ri = adata.uns['stlens']['robust_idx']

        if isinstance(ri, cp.ndarray):
                ri = ri.get()

        if not hasattr(self, '_signal_components'):
            raise RuntimeError("You must run find_optimal_pc() before calc_pca().")
    
        if device == 'gpu':
            components = self._signal_components

            if isinstance(components, np.ndarray):
                    components = cp.asarray(components)

            X_transform = components[:, ri] * cp.sqrt(
                self.eigenvalue[ri]
            ).reshape(1, -1)

            if isinstance(X_transform, cp.ndarray):
                X_transform = X_transform.get()
        elif device == 'cpu':
            X_transform = self._signal_components[:, ri] * np.sqrt(
                self.eigenvalue[ri].get()
            ).reshape(1, -1)
        else:
            raise ValueError("The device must be either 'cpu' or 'gpu'.")

        adata.obsm['X_pca_stlens'] = X_transform

        return adata
        
    def find_optimal_pc(self, data, plot_mp = False, tmp_directory=None, device='gpu', cluster = None):
        """
        Find the optimal number of principal components.

        Parameters
        ----------
        data : pd.DataFrame or anndata.AnnData
            Input data, either a pandas DataFrame or an AnnData object.
        plot_mp : bool, optional
            If True, plots the results of the PCA and SRT steps.
        tmp_directory : str, optional
            Temporary directory for storing intermediate results. If None, uses the system's temporary directory.
        device : str, optional
            Device to use for computations, either 'cpu' or 'gpu'. Default is 'gpu'.
        cluster : dask.distributed.Client, optional
            Dask distributed client for parallel processing. If None, runs in single server.

        Returns
        -------
        data : anndata.AnnData
        """

        if not device in ['cpu', 'gpu']:
            raise ValueError("The device must be either 'cpu' or 'gpu'.")

        if device == 'gpu':
            best_device = self._select_best_gpu()
            cp.cuda.Device(best_device).use()
            print(f"[stLENS] Using GPU device {best_device}", flush=True)

        tmp_dir = _get_tempdir(tmp_directory)
        tmp_prefix = uuid.uuid4()
        is_anndata = True
        if isinstance(data, pd.DataFrame):
            is_anndata = False
            obs = pd.DataFrame(data['cell'])
            X = sp.csr_matrix(data.iloc[:, 1:].values) 
            var = pd.DataFrame(data.columns[1:])
            var.columns = ['gene'] 
            data = sc.AnnData(X, obs=obs, var=var)
        elif isinstance(data, sc.AnnData):
            adata = data
        else:
            raise ValueError("Data must be a pandas DataFrame or Anndata")

        if cluster is None:
            X_normalized = self._run_in_process_value(self._normalize_process, args=(adata, tmp_dir))

        else:
            if tmp_directory is None:
                raise ValueError("tmp_directory must be defined if you use client")
            client = Client(cluster)
            
            shared = tmp_directory
            out_store = f"{shared}/Xn-{uuid.uuid4()}.zarr"
            in_store = f"{shared}/X-{uuid.uuid4()}.zarr"
        
            future = client.submit(self._normalize_mp, adata, in_store, out_store, pure=False)
            future.result()
            X_normalized = da.from_zarr(out_store)

        # X_filtered = data.raw.X if hasattr(data.raw, 'X') else data.X
        X_filtered = data.X
        if isinstance(X_filtered, anndata._core.views.ArrayView):
            X_filtered = sp.csr_matrix(X_filtered)

        # calculate sparsity
        if self.sparsity == 'auto':
            self.sparsity = self._run_in_process_value(self._calculate_sparsity, args=(X_filtered, tmp_dir, device))
            print(f"Sparsity calculation completed: {self.sparsity}")
        else:
            print(f"Using predefined sparsity: {self.sparsity}")

        # RMT
        pca_result = self._PCA(X_normalized, plot_mp = plot_mp, device=device)

        if X_normalized.shape[0] <= X_normalized.shape[1]:
            self._signal_components = pca_result[1]
            self.eigenvalue = pca_result[0]
        else:
            if device == 'gpu':
                eigenvalue = cp.asnumpy(pca_result[0])
                _signal_components = cp.asnumpy(pca_result[1])

                re = X_normalized @ _signal_components @ np.diag(1/np.sqrt(eigenvalue))
                re /= np.sqrt(X_normalized.shape[1])
                self._signal_components = cp.asarray(re)
                self.eigenvalue = cp.asarray(eigenvalue)

                del _signal_components, re
                gc.collect()
                cp._default_memory_pool.free_all_blocks()
            else:
                eigenvalue = pca_result[0]
                _signal_components = pca_result[1]
                re = X_normalized @ _signal_components @ np.diag(1/np.sqrt(eigenvalue))
                re /= np.sqrt(X_normalized.shape[1])
                self._signal_components = re
                self.eigenvalue = eigenvalue

                del _signal_components, re
                gc.collect()
                
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

        # Find and load the random_matrix library
        lib_path = _find_compiled_library("random_matrix")
        lib = ctypes.CDLL(lib_path)
        lib.sparse_rand_csr.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double]
        lib.sparse_rand_csr.restype = ctypes.POINTER(CSRMatrix)
        lib.free_csr.argtypes = [ctypes.POINTER(CSRMatrix)]

        n_rows, n_cols = X_filtered.shape
        density = 1 - self.sparsity

        if sp.issparse(X_filtered):
            X_filtered = X_filtered.astype(np.float32, copy=False)
        else:
            X_filtered = np.asarray(X_filtered, dtype=np.float32)

        dense_buf = np.empty((n_rows, n_cols), dtype=np.float32)
        block_size = 10000

        if device == 'gpu':
            sig_is_gpu = isinstance(self._signal_components, cp.ndarray)
            if not sig_is_gpu:
                # Try to place signal_components on any GPU that has room
                sig_cpu = self._signal_components
                placed = False
                for gpu_id in self._gpu_device_order():
                    try:
                        with cp.cuda.Device(gpu_id):
                            self._signal_components = cp.asarray(sig_cpu)
                        placed = True
                        break
                    except cp.cuda.memory.OutOfMemoryError:
                        with cp.cuda.Device(gpu_id):
                            cp.get_default_memory_pool().free_all_blocks()
                            cp.get_default_pinned_memory_pool().free_all_blocks()
                            cp._default_memory_pool.free_all_blocks()
                if not placed:
                    # Keep on CPU; SRT body will stream chunks
                    self._signal_components = sig_cpu

        self.pert_vecs = list()
        for _ in tqdm(range(self.n_rand_matrix), total=self.n_rand_matrix):

            mat_ptr = lib.sparse_rand_csr(n_rows, n_cols, density)
            if not bool(mat_ptr):
                raise MemoryError("Failed to allocate CSR matrix.")
            mat = mat_ptr.contents

            indptr = np.ctypeslib.as_array(mat.indptr, shape=(mat.n_rows + 1,))
            indices = np.ctypeslib.as_array(mat.indices, shape=(mat.nnz,))
            mat_data = np.ctypeslib.as_array(mat.data, shape=(mat.nnz,))
            rand_sp = sp.csr_matrix((mat_data, indices, indptr), shape=(mat.n_rows, mat.n_cols)).copy()

            del indptr, indices, mat_data, mat
            lib.free_csr(mat_ptr)
            del mat_ptr

            for i in range(0, n_rows, block_size):
                end = min(i + block_size, n_rows)
                block = rand_sp[i:end] + X_filtered[i:end]
                if sp.issparse(block):
                    block.toarray(out=dense_buf[i:end])
                else:
                    dense_buf[i:end] = block

            del rand_sp
            rand = self._normalize_inplace(dense_buf)
            
            n = min(self._signal_components.shape[1] * self._perturbed_n_scale, X_normalized.shape[1])

            if device == 'cpu':
                perturbed_L, perturbed_V = self._PCA_rand(rand, n, strategy='cpu', device=device)
            elif device == 'gpu':
                gb = self._estimate_matrix_memory(rand.shape, step='pca_rand')
                pca_strategy = self._calculate_gpu_memory(gb, step='pca_rand')  # cupy or dask
                perturbed_L, perturbed_V = self._PCA_rand(rand, n, pca_strategy, device=device)
            else:
                raise ValueError("The device must be either 'cpu' or 'gpu'.")


            if device == 'cpu':
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


            elif device == 'gpu':
                n_pert = perturbed_V.shape[1]
                chunk = 10000

                pert_vec_numpy = None
                for gpu_id in self._gpu_device_order():
                    try:
                        with cp.cuda.Device(gpu_id):
                            # === GPU streaming compute on gpu_id ===
                            if rand.shape[0] <= rand.shape[1]:
                                perturbed_gpu = self._to_current_device(perturbed_V)
                            else:
                                perturbed_L_trunc = perturbed_L[-n:]
                                perturbed_V_gpu = self._to_current_device(perturbed_V, dtype=cp.float32)
                                L_gpu = self._to_current_device(perturbed_L_trunc, dtype=cp.float32)
                                inv_sqrt_L_gpu = 1.0 / cp.sqrt(L_gpu)
                                norm_factor = cp.float32(1.0 / np.sqrt(rand.shape[1]))

                                perturbed_gpu = cp.empty((rand.shape[0], n_pert), dtype=cp.float32)
                                for i in range(0, rand.shape[0], chunk):
                                    end = min(i + chunk, rand.shape[0])
                                    rand_chunk_gpu = cp.asarray(rand[i:end], dtype=cp.float32)
                                    perturbed_gpu[i:end] = rand_chunk_gpu @ perturbed_V_gpu * inv_sqrt_L_gpu * norm_factor
                                    del rand_chunk_gpu
                                del perturbed_V_gpu, L_gpu, inv_sqrt_L_gpu
                                cp.get_default_memory_pool().free_all_blocks()

                            # signal_components.T @ perturbed on current device (stream if on CPU or cross-device)
                            sig_comp = self._signal_components
                            sig_on_current = isinstance(sig_comp, cp.ndarray) and sig_comp.device.id == gpu_id
                            if sig_on_current:
                                pert_select_gpu = cp.abs(sig_comp.T @ perturbed_gpu)
                            else:
                                # Stream chunks from CPU (or cross-device cupy) onto current device
                                sig_src = sig_comp if isinstance(sig_comp, np.ndarray) else cp.asnumpy(sig_comp)
                                acc = cp.zeros((sig_src.shape[1], perturbed_gpu.shape[1]), dtype=cp.float32)
                                for i in range(0, sig_src.shape[0], chunk):
                                    end = min(i + chunk, sig_src.shape[0])
                                    sig_chunk_gpu = cp.asarray(sig_src[i:end], dtype=cp.float32)
                                    acc += sig_chunk_gpu.T @ perturbed_gpu[i:end]
                                    del sig_chunk_gpu
                                pert_select_gpu = cp.abs(acc)
                                del acc
                            pert_select_idx = cp.argmax(pert_select_gpu, axis=1)
                            del pert_select_gpu

                            pert_vec_numpy = cp.asnumpy(perturbed_gpu[:, pert_select_idx])

                            del perturbed_gpu, pert_select_idx
                            gc.collect()
                            cp.get_default_memory_pool().free_all_blocks()
                            cp.get_default_pinned_memory_pool().free_all_blocks()
                            cp._default_memory_pool.free_all_blocks()
                        break  # success, exit device loop
                    except cp.cuda.memory.OutOfMemoryError:
                        with cp.cuda.Device(gpu_id):
                            cp.get_default_memory_pool().free_all_blocks()
                            cp.get_default_pinned_memory_pool().free_all_blocks()
                            cp._default_memory_pool.free_all_blocks()
                        print(f'[Warning] GPU {gpu_id} OOM in SRT append. Trying next device...', flush=True)
                        continue

                if pert_vec_numpy is not None:
                    self.pert_vecs.append(pert_vec_numpy)
                    del rand, pert_vec_numpy
                    gc.collect()
                else:
                    # CPU fallback (all GPUs OOM)
                    print('[Warning] All GPUs OOM in SRT append. Falling back to CPU for this iteration.', flush=True)
                    pV = cp.asnumpy(perturbed_V) if isinstance(perturbed_V, cp.ndarray) else perturbed_V
                    pL = cp.asnumpy(perturbed_L) if isinstance(perturbed_L, cp.ndarray) else perturbed_L

                    if rand.shape[0] <= rand.shape[1]:
                        perturbed = pV
                    else:
                        pL = pL[-n:]
                        re = rand @ pV @ np.diag(1/np.sqrt(pL))
                        re /= np.sqrt(rand.shape[1])
                        perturbed = re
                        del re

                    sig_comp = self._signal_components
                    if isinstance(sig_comp, cp.ndarray):
                        sig_comp = cp.asnumpy(sig_comp)
                        self._signal_components = sig_comp

                    pert_select = np.argmax(np.abs(sig_comp.T @ perturbed), axis=1)
                    self.pert_vecs.append(perturbed[:, pert_select])

                    del rand, perturbed, pert_select, pV, pL
                    gc.collect()

        del dense_buf
        gc.collect()

        if device == 'gpu':
            pert_scores = list()
            for i in range(self.n_rand_matrix):
                for j in range(i+1, self.n_rand_matrix):
                    dots = self.pert_vecs[i].T @ self.pert_vecs[j]
                    corr = np.max(np.abs(dots), axis=1)
                    pert_scores.append(corr)

            pert_scores = cp.array(pert_scores)

            def iqr(x):
                q1 = cp.percentile(x, 25)
                q3 = cp.percentile(x, 75)
                iqr = q3 - q1
                filtered = x[(x >= q1 -1.5*iqr) & (x <= q3 + 1.5*iqr)]
                return cp.median(filtered)

            rob_scores = cp.array([iqr(pert_scores[:,i]) for i in range(pert_scores.shape[1])])
            robust_idx = rob_scores > self.threshold
        elif device == 'cpu':
            pert_scores = list()
            for i in range(self.n_rand_matrix):
                for j in range(i+1, self.n_rand_matrix):
                    dots = self.pert_vecs[i].T @ self.pert_vecs[j]
                    corr = np.max(np.abs(dots), axis = 1)
                    pert_scores.append(corr)

            pert_scores = np.array(pert_scores)

            def iqr(x):
                q1 = np.percentile(x, 25)
                q3 = np.percentile(x, 75)
                iqr = q3 - q1
                filtered = x[(x >= q1 -1.5*iqr) & (x <= q3 + 1.5*iqr)]
                return np.median(filtered)

            rob_scores = np.array([iqr(pert_scores[:,i]) for i in range(pert_scores.shape[1])])
            robust_idx = rob_scores > self.threshold

        if isinstance(robust_idx, cp.ndarray):
            robust_idx_np = robust_idx.get()
        else:
            robust_idx_np = robust_idx

        data.uns['stlens'] = {
            'optimal_pc_count': int(np.sum(robust_idx_np)),
            'robust_idx': robust_idx,
            'robust_scores': pert_scores,
        }
        print(f"number of filtered signals: {data.uns['stlens']['optimal_pc_count']}")

        return data

    
    def _calculate_sparsity(self, X_filtered, tmp_dir, device):
        if device == 'gpu':
            best_device = self._select_best_gpu()
            cp.cuda.Device(best_device).use()
            print(f"[stLENS:_calculate_sparsity] Using GPU device {best_device}", flush=True)
        tmp_prefix = uuid.uuid4()
        X_filtered = sp.csr_matrix(X_filtered)

        bin_matrix = sp.csr_matrix(
            (np.ones_like(X_filtered.data, dtype=np.float32),
            X_filtered.indices,
            X_filtered.indptr),
            shape=X_filtered.shape)

        sparse = 0.999
        shape_row, shape_col = X_filtered.shape
        n_len = shape_row * shape_col
        n_zero = n_len - X_filtered.size

        rng = np.random.default_rng()

        # Calculate threshold for correlation
        n_sampling = min(X_filtered.shape)
        thresh = np.mean([max(np.abs(rng.normal(0, np.sqrt(1/n_sampling), n_sampling)))
                            for _ in range(5000)]).item()
        print(f'sparsity_th: {thresh}')

        csr_indptr_np = np.ascontiguousarray(X_filtered.indptr, dtype=np.int32)
        csr_indices_np = np.ascontiguousarray(X_filtered.indices, dtype=np.int32)
        csr_indptr_ptr = csr_indptr_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        csr_indices_ptr = csr_indices_np.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

        bin_nor = bin_matrix.toarray()
        bin_nor = self._normalize_inplace(bin_nor)

        if device == 'cpu':
            _, Vb= self._PCA_rand(bin_nor, bin_nor.shape[0], strategy='cpu', device=device)

        elif device == 'gpu':
            gb = self._estimate_matrix_memory(bin_nor.shape, step='pca_rand')
            strategy = self._calculate_gpu_memory(gb, step = 'pca_rand') # cupy or dask
            _, Vb= self._PCA_rand(bin_nor, bin_nor.shape[0], strategy, device=device)
            if isinstance(Vb, np.ndarray):
                strategy = 'cpu'
        else:
            raise ValueError("The device must be either 'cpu' or 'gpu'.")

        del bin_nor
        gc.collect()        

        n_vbp = Vb.shape[1]//2
        n_buffer = 5
        buffer = [1] * n_buffer

        print(f"Initial sparse: {sparse}, threshold: {self.sparsity_threshold}")

        rows, cols = X_filtered.shape
        dense_buf = np.empty((rows, cols), dtype=np.float32)
        block_size = 10000

        # Load perturb_omp library once outside the loop
        lib_path = _find_compiled_library("perturb_omp")
        lib = ctypes.CDLL(lib_path)
        lib.perturb_zeros.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.free_perturb_output.argtypes = [
            ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
            ctypes.c_int,
        ]
        lib.free_perturb_output.restype = None

        while sparse > self.sparsity_threshold:
            n_pert = int((1-sparse) * n_len)
            p = n_pert / n_zero

            output_rows = (ctypes.POINTER(ctypes.c_int) * rows)()
            output_sizes = (ctypes.c_int * rows)()

            lib.perturb_zeros(csr_indptr_ptr, csr_indices_ptr, rows, cols, p, output_rows, output_sizes)

            sizes_np = np.ctypeslib.as_array(output_sizes)
            indptr = np.zeros(rows + 1, dtype=np.int32)
            np.cumsum(sizes_np, dtype=np.int32, out=indptr[1:])
            total_nnz = int(indptr[-1])

            indices = np.empty(total_nnz, dtype=np.int32)
            for i in range(rows):
                size = int(sizes_np[i])
                if size > 0:
                    indices[indptr[i]:indptr[i + 1]] = np.ctypeslib.as_array(
                        output_rows[i], shape=(size,)
                    )

            lib.free_perturb_output(output_rows, rows)

            data = np.ones(total_nnz, dtype=np.float32)
            pert = sp.csr_matrix((data, indices, indptr), shape=(rows, cols))

            for i in range(0, rows, block_size):
                end = min(i + block_size, rows)
                block = pert[i:end] + bin_matrix[i:end]
                if sp.issparse(block):
                    block.toarray(out=dense_buf[i:end])
                else:
                    dense_buf[i:end] = block

            del pert
            pert = self._normalize_inplace(dense_buf)

            if device == 'cpu' or strategy == 'cpu':
                _, Vbp = self._PCA_rand(pert, n_vbp, strategy ='cpu', device=device)

            elif device == 'gpu':
                gb = self._estimate_matrix_memory(pert.shape, step='pca_rand')
                if strategy == 'dask' or strategy == 'cupy':
                    strategy = self._calculate_gpu_memory(gb, step = 'pca_rand') # cupy or dask
                _, Vbp = self._PCA_rand(pert, n_vbp, strategy, device=device)
            else:
                raise ValueError("The device must be either 'cpu' or 'gpu'.")

            del pert
            gc.collect()

            if device == 'cpu' or strategy == 'cpu':
                if isinstance(Vb, cp.ndarray):
                    Vb = Vb.get()
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                    cp._default_memory_pool.free_all_blocks()

                corr_arr = np.max(np.abs(Vb.T @ Vbp), axis=0)
            elif device == 'gpu':
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

        del dense_buf
        gc.collect()

        return self.sparsity

        
    def _PCA(self, X, device, plot_mp = False):
        pca = PCA(device = device)
        pca.fit(X)
        
        if plot_mp:
            pca.plot_mp(comparison = False)
            plt.show()
        
        comp = pca.get_signal_components()

        del pca
        gc.collect()
    
        return comp
    

    def _PCA_rand(self, X, n, strategy, device):
        pca = PCA(device = device)

        if strategy == 'cupy':
            X = cp.asarray(X)
            Y = pca._wishart_matrix(X)
            L, V = cp.linalg.eigh(Y)

            del Y
            cp._default_memory_pool.free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
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

        return L, V
    

    def plot_robust_score(self, adata):
        """
        Plot the robust scores and their stability.
        
        Parameters
        ----------
        adata : anndata.AnnData
            AnnData object containing the results of the stLENS analysis.

        Returns
        -------
        scatter plot
        """
        rs = adata.uns['stlens']['robust_scores']
        ri = adata.uns['stlens']['robust_idx']
        if isinstance(ri, np.ndarray):
            m_scores = rs.mean(axis=0)
            sd_scores = rs.std(axis=0)
        else:
            m_scores = rs.mean(axis=0).get()
            sd_scores = rs.std(axis=0).get()
        nPC = np.arange(1, len(m_scores)+1)

        colors = cm.get_cmap("RdBu")(1.0 - m_scores)

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_xlabel("nPC")
        ax.set_ylabel("Stability")
        ax.set_title(f"{np.sum(ri)} robust signals were detected")

        ax.scatter(nPC, m_scores, c=colors, s=50, edgecolor='k')
        ax.errorbar(nPC, m_scores, yerr=sd_scores, fmt='none', color='gray', capsize=4, linewidth=1)
        ax.axhline(y=self.threshold, color='k', linestyle='--')

        ax.invert_xaxis()

    def _estimate_matrix_memory(self, tuple, step):
        dtype = 4
        row = tuple[0]
        col = tuple[1]

        if step == 'srt':
            bytes_total = row * col * dtype
            gb = bytes_total / 1024**3

        elif step == 'pca_rand':
            gb = 0
            bytes_total = row * col * dtype
            gb += bytes_total / 1024**3

            wishart_dim = min(row, col)
            num_elements = wishart_dim ** 2
            gb += (num_elements * dtype) / (1024 ** 3)

            num_elements = wishart_dim + wishart_dim ** 2
            gb += (num_elements * dtype) / (1024 ** 3)

            gb += 8
        
        return gb

    def _calculate_gpu_memory(self, gb, step):
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
    

    def clean_tempfiles(tmp_dir=None):
        tmp_dir = tempfile.gettempdir() if tmp_dir is None else tmp_dir

        patterns = [
            "*-perturbed.zarr",
            "*-normalized_X.zarr",
            "*-bin.zarr",
            "*-srt_perturbed.zarr"
        ]

        for pattern in patterns:
            full_pattern = os.path.join(tmp_dir, pattern)
            for path in glob.glob(full_pattern):
                if os.path.isdir(path):
                    try:
                        shutil.rmtree(path)
                        print(f"Deleted: {path}")
                    except Exception as e:
                        print(f"Failed to delete {path}: {e}")

    

    
