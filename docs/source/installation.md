# Installation & Setup

To use stLENS, you can follow the steps below to install it and set up your environment.

## Installation

```python
pip install stLENS
```

## Setup

For distributed computing with Dask, you need to set up a Dask client. This is particularly useful when using functions like `find_optimal_pc` that support parallel processing.

```python
from dask.distributed import Client, LocalCluster

# Create a local cluster with a specific host
cluster = LocalCluster(host='{your_host_ip}')
client = Client(cluster)
```