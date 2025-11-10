# Installation & Setup

To use stLENS, you can follow the steps below to install it and set up your environment.

## Installation

```python
pip install stLENS
```

## Setup

For distributed computing with Dask, you need to set up a Dask client. This is particularly useful when using functions like `find_optimal_pc` that support parallel processing.

### Setting up Dask Client
```python
from dask.distributed import Client, LocalCluster

# Create a local cluster with a specific host
cluster = LocalCluster(host='{your_host_ip}')
client = Client(cluster)
```

### Connecting Additional Workers

To connect additional servers as Dask workers:

1. First, create the cluster and check its scheduler address:
   ```python
   cluster = LocalCluster(host='{your_host_ip}')
   print(cluster)  # This will display the scheduler address (e.g., tcp://{your_host_ip}:{port})
   ```

2. On each additional server you want to use as a worker, open a terminal and run:
   ```bash
   dask-worker tcp://{your_host_ip}:{port} --nworkers=10
   ```

   **Note:** Replace `{your_host_ip}` and `{port}` with the actual values from the scheduler address displayed when you printed the `cluster` object in step 1.