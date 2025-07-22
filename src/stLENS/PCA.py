from .calc import Calc

import dask.array as da
import scanpy as sc
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import gc


class PCA():
    def __init__(self, device = None, data = None): 
        self.device = device
        self.data = data


    def fit(self, X=None, eigen_solver = 'wishart'):
        calc = Calc()
        self.n_cells, self.n_genes = X.shape
        self.rmt_device = None

        if eigen_solver == 'wishart':
            self.L, self.V = self._get_eigen(X)
            Xr = self._random_matrix(X)

            if isinstance(self.L, np.ndarray):
                self.device = 'cpu'
                self.rmt_device = 'cpu'
            self.Lr, self.Vr = self._get_eigen(Xr)

            if self.rmt_device != 'cpu' and isinstance(self.Lr, np.ndarray):
                self.L = self.L.get()
                self.V = self.V.get()
                self.rmt_device = 'cpu'
            elif self.rmt_device == 'cpu':
                self.rmt_device = 'cpu'
            else:
                self.rmt_device = 'gpu'


            del Xr
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            cp._default_memory_pool.free_all_blocks() 
            gc.collect()

            self.explained_variance_ = (self.L**2) / self.n_cells
            self.total_variance_ = self.explained_variance_.sum()

            calc.L = self.L 
            self.L_mp = calc._mp_calculation(self.L, self.Lr, self.rmt_device)
            calc.L_mp = self.L_mp
            self.lambda_c = calc._tw(self.rmt_device)
            print("lambda_c:",self.lambda_c)
            self.peak = calc._mp_parameters(self.L_mp, self.rmt_device)['peak']

        else:
            raise ValueError("Invalid eigen_solver. Use 'wishart'.")
        
        self.Ls = self.L[self.L > self.lambda_c]
        self.Vs = self.V[:, self.L > self.lambda_c]

        noise_boolean = ((self.L < self.lambda_c) & (self.L > calc.b_minus))
        self.Vn = self.V[:, noise_boolean]
        self.Ln = self.L[noise_boolean]
    
        self.n_components = len(self.Ls)
        print(f"Number of signal: {self.n_components}")

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        gc.collect()

    def get_signal_components(self, n_components=0):
        if n_components == 0:
            comp = self.Ls,  self.Vs
            return comp
        elif n_components >= 1:
            comp = self.Ls[:n_components], self.Vs[:n_components]
            return comp
        raise ValueError('n_components must be positive')
    
    def _wishart_matrix(self, X):
        if X.shape[0] <= X.shape[1]:
            Y = (X @ X.T)
        else:
            Y = (X.T @ X)
        Y /= X.shape[1]
        return Y
    
    def to_gpu(self, Y):
        chunk_size = (10000, Y.shape[1])
        if isinstance(Y, da.core.Array):
            Y_dask = Y
        else : 
            Y_dask = da.from_array(Y, chunks=chunk_size)

        Y_gpu = cp.asarray(Y_dask.blocks[0])

        chunk = len(Y_dask.chunks[0])
        for i in range(1, chunk):
            block = cp.asarray(Y_dask.blocks[i])
            Y_gpu = cp.concatenate((Y_gpu, block), axis=0)

            del block
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
            cp._default_memory_pool.free_all_blocks() 

        del Y_dask, chunk_size, Y, chunk
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp._default_memory_pool.free_all_blocks() 

        return Y_gpu
    
    def _get_eigen(self, X):
        Y = self._wishart_matrix(X)
        if self.device=='gpu':
            try:
                Y = self.to_gpu(Y)
                L, V = cp.linalg.eigh(Y)
            except cp.cuda.memory.OutOfMemoryError:
                print('[Warning] GPU memory insufficient. Falling back to CPU computation.')
                if isinstance(Y, cp.ndarray):
                    Y = Y.get()
                    cp.get_default_memory_pool().free_all_blocks()
                    cp.get_default_pinned_memory_pool().free_all_blocks()
                    cp._default_memory_pool.free_all_blocks()
                    L, V = np.linalg.eigh(Y)

        elif self.device=='cpu':
            L, V = np.linalg.eigh(Y)
        else:
            raise ValueError("The device must be either 'cpu' or 'gpu'.")

        del Y
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp._default_memory_pool.free_all_blocks() 
        gc.collect()
        return L, V

    def _random_matrix(self, X):

        if isinstance(X, da.core.Array):
            X_dask = X
        else:
            X_dask = da.from_array(X, chunks=(10000, X.shape[1]))
        def shuffle_block(block):
            for row in block:
                np.random.shuffle(row)
            return block

        Xr_dask = X_dask.map_blocks(shuffle_block, dtype=X_dask.dtype)
        Xr = Xr_dask.compute()

        del X_dask, Xr_dask
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        cp._default_memory_pool.free_all_blocks() 
        return Xr
    

    def plot_mp(self, comparison=False, path=False, info=True, bins=None, title=None):
        calc = Calc()
        calc.style_mp_stat()

        if bins is None:
            bins = 300

        if self.device == 'gpu':
            x = np.linspace(0, int(cp.round(cp.max(self.L_mp) + 0.5)), 2000)
            y = calc._mp_pdf(x, self.L_mp, self.rmt_device).get()
            if comparison and self.Lr is not None:
                yr = calc._mp_pdf(x, self.Lr, self.rmt_device).get()
        elif self.device == 'cpu':
            x = np.linspace(0, int(np.round(np.max(self.L_mp) + 0.5)), 2000)
            y = calc._mp_pdf(x, self.L_mp, self.rmt_device)
            if comparison and self.Lr is not None:
                yr = calc._mp_pdf(x, self.Lr, self.rmt_device)
        else:
            raise ValueError("The device must be either 'cpu' or 'gpu'.")

        if info:
            fig = plt.figure(dpi=100)
            fig.set_layout_engine()

            ax = fig.add_subplot(111)

            dic = calc._mp_parameters(self.L_mp, self.rmt_device)
            info1 = (r'$\bf{Data Parameters}$' + '\n{0} cells\n{1} genes'
                    .format(self.n_cells, self.n_genes))
            info2 = ('\n' + r'$\bf{MP\ distribution\ in\ data}$'
                    + '\n$\gamma={:0.2f}$ \n$\sigma^2={:1.2f}$\
                    \n$b_-={:2.2f}$\n$b_+={:3.2f}$'
                    .format(dic['gamma'], dic['s'], dic['b_minus'],
                            dic['b_plus']))

            n_components = self.n_components if self.n_components is not None else 0
            info3 = ('\n' + r'$\bf{Analysis}$' +
                    '\n{0} eigenvalues > $\lambda_c (3 \sigma)$\
                    \n{1} noise eigenvalues'
                    .format(n_components, self.n_cells - n_components))

            if isinstance(self.L_mp, np.ndarray):
                cdf_func = calc._call_mp_cdf(self.L_mp, dic)  
                ks = stats.kstest(self.L_mp, cdf_func)
            else:
                cdf_func = calc._call_mp_cdf(self.L_mp.get(), dic)  
                ks = stats.kstest(self.L_mp.get(), cdf_func)  

            info4 = '\n'+r'$\bf{Statistics}$'+'\nKS distance ={0}'\
                .format(round(ks[0], 4))\
                + '\nKS test p-value={0}'\
                .format(round(ks[1], 2))

            infoT = info1 + info2 + info4 + info3

            ax.text(1.05, 1.02, infoT, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(facecolor='wheat', alpha=0.8, boxstyle='round,pad=0.5'))
        else:
            plt.figure(dip=100)
            
        
        if not isinstance(self.L, np.ndarray):
            self.L = self.L.get()
        if not isinstance(self.Lr, np.ndarray):
            self.Lr = self.Lr.get()
            
        plot = sns.histplot(self.L, bins=bins, stat="density",
                        kde=False, color=sns.xkcd_rgb["cornflower blue"], alpha=0.85)
    
        plt.plot(x, y, color=sns.xkcd_rgb["pale red"], lw=2, label="MP for random part in data")


        MP_data = mlines.Line2D([], [], color=sns.xkcd_rgb["pale red"], label="MP for random part in data", linewidth=2)
        MP_rand = mlines.Line2D([], [], color=sns.xkcd_rgb["sap green"], label="MP for randomized data", linewidth=1.5, linestyle='--')
        randomized = mpatches.Patch(color=sns.xkcd_rgb["apple green"], label="Randomized data", alpha=0.75, linewidth=3, fill=False)
        data_real = mpatches.Patch(color=sns.xkcd_rgb["cornflower blue"], label="Real data", alpha=0.85)

        if comparison:
            sns.histplot(self.Lr, bins=30, kde=False,
                        stat="density", color=sns.xkcd_rgb["apple green"], alpha=0.75, linewidth=3)
            
            ax.plot(x, yr, sns.xkcd_rgb["sap green"], lw=1.5, ls='--')

            ax.legend(handles=[data_real, MP_data, randomized, MP_rand], loc="upper right", frameon=True)
        else:
            ax.legend(handles=[data_real, MP_data], loc="upper right", frameon=True)

        if self.device == 'cpu':
            max_Lr = np.max(self.Lr) if self.Lr is not None else 0
            max_L_mp = np.max(self.L_mp) if self.L_mp is not None else 0

        elif self.device == 'gpu':
            max_Lr = cp.max(self.Lr) if self.Lr is not None else 0
            max_L_mp = cp.max(self.L_mp) if self.L_mp is not None else 0

        else:
            raise ValueError("The device must be either 'cpu' or 'gpu'.")

        ax.set_xlim([0, int(np.round(max(max_Lr, max_L_mp) + 1.5))])

        ax.grid(linestyle='--', lw=0.3)

        if title:
            ax.set_title(title)
        
        ax.set_xlabel('First cell eigenvalues normalized distribution')

        if self.data is not None and isinstance(self.data, sc.AnnData):
            self.data.uns['mp_plot'] = fig

        return fig