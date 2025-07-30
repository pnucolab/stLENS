# stLENS Documentation

Welcome to the stLENS documentation!
stLENS is a Python-based scalable tool to determine the optimal number of principal components from the spatial transcritomics data. It is fully compatible with the scverse ecosystem.

## TL;DR
### Installation
```bash
   pip install stLENS
```
### Load data
```bash
   from stLENS import stLENS
   stlens = stLENS()

   # Load data - it is recommended to use AnnData with count data stored in X
   import scanpy as sc
   adata = sc.read_h5ad("path/to/your_data.h5ad")
```
### Preprocessing with stLENS
```bash
   # Perform filtering before the steps below.

   # Find the optimal number of principal components.
   stlens.find_optimal_pc(adata)
   # Perform PCA.
   stlens.pca(adata)
```

## Contents
```{toctree}
:maxdepth: 1

installation
tutorials/index
api/index
```