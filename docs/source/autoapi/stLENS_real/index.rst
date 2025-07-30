stLENS.stLENS
=============

.. py:module:: stLENS.stLENS




Module Contents
---------------

.. py:class:: stLENS(sparsity='auto', sparsity_step=0.001, sparsity_threshold=0.9, perturbed_n_scale=2, n_rand_matrix=20, threshold=np.cos(np.deg2rad(60)))

   .. py:attribute:: sparsity
      :value: 'auto'



   .. py:attribute:: sparsity_threshold
      :value: 0.9



   .. py:attribute:: sparsity_step
      :value: 0.001



   .. py:attribute:: n_rand_matrix
      :value: 20



   .. py:attribute:: threshold
      :value: np.cos(np.deg2rad(60))


   .. py:method:: filter_cells_and_genes(data, min_tp_c=0, min_tp_g=0, max_tp_c=np.inf, max_tp_g=np.inf, min_genes_per_cell=200, max_genes_per_cell=0, min_cells_per_gene=15, mito_percent=5.0, ribo_percent=0.0, inplace=True, use_raw=True)

      Filter cells and genes in the AnnData object using the scLENS approach.

      :param data: Input AnnData object containing the data to be filtered.
      :type data: anndata.AnnData
      :param min_tp_c: Minimum total counts per cell. Default is 0.
      :type min_tp_c: int, optional
      :param min_tp_g: Minimum total counts per gene. Default is 0.
      :type min_tp_g: int, optional
      :param max_tp_c: Maximum total counts per cell. Default is np.inf.
      :type max_tp_c: int, optional
      :param max_tp_g: Maximum total counts per gene. Default is np.inf.
      :type max_tp_g: int, optional
      :param min_genes_per_cell: Minimum number of genes per cell. Default is 200.
      :type min_genes_per_cell: int, optional
      :param max_genes_per_cell: Maximum number of genes per cell. Default is 0.
      :type max_genes_per_cell: int, optional
      :param min_cells_per_gene: Minimum number of cells expressing each gene. Default is 15.
      :type min_cells_per_gene: int, optional
      :param mito_percent: Upper threshold for mitochondrial gene expression as a percentage of total cell expression. Default is 5.0.
      :type mito_percent: float, optional
      :param ribo_percent: Upper threshold for ribosomal gene expression as a percentage of total cell expression. Default is 0.0.
      :type ribo_percent: float, optional
      :param inplace: If True, modifies the input AnnData object directly. If False, returns a new AnnData object.
      :type inplace: bool, optional
      :param use_raw: If True, uses the raw attribute of the AnnData object. Default is True.
      :type use_raw: bool, optional

      :returns: **data_filtered** -- If inplace is True, returns None. If False, returns the filtered AnnData object.
      :rtype: anndata.AnnData or None



   .. py:method:: pca(adata, inplace=True, device='gpu')

      Perform PCA on the given AnnData object.

      :param adata: Input AnnData object containing the data to be transformed.
      :type adata: anndata.AnnData
      :param inplace: If True, modifies the input AnnData object directly. If False, returns a new AnnData object.
      :type inplace: bool, optional
      :param device: Device to use for computations, either 'cpu' or 'gpu'. Default is 'gpu'.
      :type device: str, optional

      :returns: **adata** -- If inplace is True, returns None. If False, returns the AnnData object with PCA results stored in `obsm['X_pca_stlens']`.
      :rtype: anndata.AnnData or None



   .. py:method:: find_optimal_pc(data, inplace=True, plot_mp=False, tmp_directory=None, device='gpu')

      Find the optimal number of principal components.

      :param data: Input data, either a pandas DataFrame or an AnnData object.
      :type data: pd.DataFrame or anndata.AnnData
      :param inplace: If True, modifies the input data directly. If False, returns a new AnnData object.
      :type inplace: bool, optional
      :param plot_mp: If True, plots the results of the PCA and SRT steps.
      :type plot_mp: bool, optional
      :param tmp_directory: Temporary directory for storing intermediate results. If None, uses the system's temporary directory.
      :type tmp_directory: str, optional
      :param device: Device to use for computations, either 'cpu' or 'gpu'. Default is 'gpu'.
      :type device: str, optional

      :returns: **adata** -- If inplace is True, returns None. If False, returns the normalized AnnData object.
      :rtype: anndata.AnnData or None



   .. py:method:: plot_robust_score(adata)

      Plot the robust scores and their stability.

      :param adata: AnnData object containing the results of the stLENS analysis.
      :type adata: anndata.AnnData

      :rtype: scatter plot



   .. py:method:: clean_tempfiles()


