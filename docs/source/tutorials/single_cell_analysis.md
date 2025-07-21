# Single-cell Analysis

This section covers tutorials for analyzing single-cell RNA-seq data using stLENS.

## Load Data
예시 데이터를 다운받을 수 있습니다.
```python
!wget -P ./data/sc_sim https://public.pnucolab.com/stlens/sim_muris_atlas.h5ad
```
데이터를 읽습니다.
```python
import scanpy as sc

adata = sc.read_h5ad("./data/sc_sim/sim_muris_atlas.h5ad")
adata
```
## Settings

```python
import stLENS

from stLENS import find_optimal_pc
stlens = find_optimal_pc.find_optimal_pc()
```
아래는 회전변환 과정까지 모두 포함하고 있습니다. 이 코드를 사용한다면 아래 PCA 과정에서 주석 처리된 코드를 사용하세요.
```python
import stLENS

from stLENS import stLENS_py
stlens = stLENS_py.stLENS_py()
```
디렉토리를 설정합니다.
```python
import os
import shutil

file_name = "test_stlens"

directory = f"./output/{file_name}"
stlens.directory = directory
```
gpu 또는 cpu 버전으로 사용 가능합니다.
```python
stlens.device = 'gpu'
```
## Quality Control & Normalization
stLENS의 filtering 방법을 사용하는 경우 :
```python
stlens.preprocess(adata, filter=True, plot=False)
```
scanpy의 filtering 방법을 사용하는 경우 :
```python
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=True, inplace=False)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=15)

stlens.preprocess(adata, filter=False, plot=False)
```
## PCA
```python
stlens.fit_transform(plot_mp = False)
sc.pp.pca(adata, n_comps=)

# PCA 전체 과정 포함 코드
# stlens.fit_transform(plot_mp = False)
# stlens.plot_robust_score()
```