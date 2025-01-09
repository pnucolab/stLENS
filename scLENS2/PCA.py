from calc import Calc

import pandas as pd
import scanpy as sc
import torch
import numpy as np
import cupy as cp
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

class PCA():
    def __init__(self,device = None, data = None):

        # self.L = None
        # self.V = None
        # self.L_mp = None
        # self.explained_variance_ = []
        # self.total_variance_ = []
        self.device = device
        self.data = data


    def fit(self, X=None, eigen_solver = 'wishart'):
        calc = Calc()
        self.n_cells, self.n_genes = X.shape

        if eigen_solver == 'wishart':
            self.L, self.V = self._get_eigen(X)
            Xr = self._random_matrix(X)
            self.Lr, self.Vr = self._get_eigen(Xr)

            self.explained_variance_ = (self.L**2) / self.n_cells
            self.total_variance_ = self.explained_variance_.sum()

        
            calc.L = self.L
            self.L_mp = calc._mp_calculation(self.L, self.Lr)
            calc.L_mp = self.L_mp
            self.lambda_c = calc._tw()
            self.peak = calc._mp_parameters(self.L_mp)['peak']

        else:
            raise ValueError("Invalid eigen_solver. Use 'wishart'.")
        
        self.Ls = self.L[self.L > self.lambda_c]
        self.Vs = self.V[:, self.L > self.lambda_c]

        noise_boolean = ((self.L < self.lambda_c) & (self.L > calc.b_minus))
        self.Vn = self.V[:, noise_boolean]
        self.Ln = self.L[noise_boolean]
        self.n_components = len(self.Ls)
        print(f"Number of signal components: {self.n_components}")

    def get_signal_components(self, n_components=0):
        if n_components == 0:
            comp = self.Ls,  self.Vs
            return comp
        elif n_components >= 1:
            comp = self.Ls[:n_components], self.Vs[:n_components]
            return comp
        raise ValueError('n_components must be positive')

    # def _tw(self):
    #     '''Tracy-Widom critical eigenvalue'''
    #     gamma = self._mp_parameters(self.L_mp)['gamma']
    #     p = len(self.L) / gamma
    #     sigma = 1 / cp.power(p, 2/3) * cp.power(gamma, 5/6) * \
    #         cp.power((1 + cp.sqrt(gamma)), 4/3)
    #     lambda_c = cp.mean(self.L_mp) * (1 + cp.sqrt(gamma)) ** 2 + sigma
    #     return lambda_c

    # def _mp_calculation(self, L, Lr, eta=1, eps=10**-6, max_iter=1000):
    #     converged = False
    #     iter = 0
    #     loss_history = []
    
    #     b_plus = self._mp_parameters(Lr)['b_plus']
    #     b_minus = self._mp_parameters(Lr)['b_minus']
        
    #     L_updated = L[(L > b_minus) & (L < b_plus)]
    #     new_b_plus = self._mp_parameters(L_updated)['b_plus']
    #     new_b_minus = self._mp_parameters(L_updated)['b_minus']
    
    #     while not converged:
    #         loss = (1 - float(new_b_plus) / float(b_plus))**2
    #         loss_history.append(loss)
    #         iter += 1
        
    #         if loss <= eps:
    #             converged = True
    #         elif iter == max_iter:
    #             print('Max interactions exceeded!')
    #             converged = True
    #         else:
    #             gradient = new_b_plus - b_plus
    #             new_b_plus = b_plus + eta * gradient
                
    #             L_updated = L[(L > new_b_minus) & (L < new_b_plus)]
    #             self.b_plus = new_b_plus
    #             self.b_minus = new_b_minus
                
    #             new_b_plus = self._mp_parameters(L_updated)['b_plus']
    #             new_b_minus = self._mp_parameters(L_updated)['b_minus']
    
    #     indices = cp.where((L > new_b_minus) & (L < new_b_plus))
    #     L_mp = L[indices]
    #     return cp.array(L_mp)

    # def _mp_parameters(self, L):
    #     moment_1 = cp.mean(L)
    #     moment_2 = cp.mean(cp.power(L, 2))
    #     gamma = moment_2 / float(moment_1**2) - 1
    #     s = moment_1
    #     sigma = moment_2
    #     b_plus = s * (1 + cp.sqrt(gamma))**2
    #     b_minus = s * (1 - cp.sqrt(gamma))**2
    #     x_peak = s * (1.0 - gamma)**2.0 / (1.0 + gamma)
    #     return {
    #         'moment_1': moment_1,
    #         'moment_2': moment_2,
    #         'gamma': gamma, 
    #         'b_plus': b_plus,
    #         'b_minus': b_minus,
    #         's': s,
    #         'peak': x_peak,
    #         'sigma': sigma
    #     }

    def _get_eigen(self, X):
        Y = self._wishart_matrix(X)
        L, V = cp.linalg.eigh(Y)
        return L, V

    def _wishart_matrix(self, X):
        Y = (X @ X.T)
        Y /= X.shape[1]
        return Y

    def _random_matrix(self, X):
        Xr = cp.array([
            cp.random.permutation(row) for row in X
        ])
        return Xr
    
    # def style_mp_stat(self):
    #     plt.style.use("ggplot")
    #     np.seterr(invalid='ignore')
    #     sns.set_style("white")
    #     sns.set_context("paper")
    #     sns.set_palette("deep")
    #     plt.rcParams['axes.linewidth'] =0.5
    #     plt.rcParams['figure.dpi'] = 100

    # def _marchenko_pastur(self, x, dic):
    #     '''Distribution of eigenvalues'''
    #     pdf = np.sqrt((dic['b_plus'] - x) * (x-dic['b_minus']))\
    #         / float(2 * dic['s'] * np.pi * dic['gamma'] * x)
    #     return pdf
    
    # def _mp_pdf(self, x, L):
    #     '''Marchnko-Pastur PDF'''
    #     dic = self._mp_parameters(L)
    #     y = cp.empty_like(x)
    #     for i, xi in enumerate(x):
    #         y[i] = self._marchenko_pastur(xi, dic)
    #     return y
    
    # def _call_mp_cdf(self,L,dic):
    #     "CDF of Marchenko Pastur"
    #     func= lambda y: list(map(lambda x: self._cdf_marchenko(x,dic), y))
    #     return func
    
    # def _cdf_marchenko(self,x,dic):
    #     if x < dic['b_minus']: 
    #         return 0.0
    #     elif x>dic['b_minus'] and x<dic['b_plus']:
    #         return 1/float(2*dic['s']*np.pi*dic['gamma'])*\
    #         float(np.sqrt((dic['b_plus']-x)*(x-dic['b_minus']))+\
    #         (dic['b_plus']+dic['b_minus'])/2*np.arcsin((2*x-dic['b_plus']-\
    #         dic['b_minus'])/(dic['b_plus']-dic['b_minus']))-\
    #     np.sqrt(dic['b_plus']*dic['b_minus'])*np.arcsin(((dic['b_plus']+\
    #         dic['b_minus'])*x-2*dic['b_plus']*dic['b_minus'])/\
    #     ((dic['b_plus']-dic['b_minus'])*x)) )+np.arcsin(1)/np.pi
    #     else:
    #         return 1.0

    def plot_mp(self, comparison=False, path=False, info=True, bins=None, title=None):
        calc = Calc()
        calc.style_mp_stat()
        if bins is None:
            bins = 300

        x = np.linspace(0, int(cp.round(cp.max(self.L_mp) + 0.5)), 2000)
        y = calc._mp_pdf(x, self.L_mp).get()

        if comparison and self.Lr is not None:
            yr = calc._mp_pdf(x, self.Lr).get()

        # info 부분 합침
        if info:
            fig = plt.figure(dpi=100)
            fig.set_layout_engine()

            ax = fig.add_subplot(111)

            dic = calc._mp_parameters(self.L_mp)
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

            # 디버깅
            print("L_mp type:", type(self.L_mp))
            print("L_mp shape:", self.L_mp.shape if hasattr(self.L_mp, "shape") else "No shape attribute")

            # 수정
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

            # props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)

            # at = AnchoredText(infoT, loc=2, prop=dict(size=10),
            #                 frameon=True,
            #                 bbox_to_anchor=(1., 1.024),
            #                 bbox_transform=ax.transAxes)
            # at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            # lgd = ax.add_artist(at)
        else:
            plt.figure(dip=100)
            
        
        # distplot이 deprecated -> histplot으로 변경
        plot = sns.histplot(self.L.get(), bins=bins, stat="density",
                        kde=False, color=sns.xkcd_rgb["cornflower blue"], alpha=0.85)
    
        # MP 분포 선 (랜덤 데이터)
        plt.plot(x, y, color=sns.xkcd_rgb["pale red"], lw=2, label="MP for random part in data")


        # 범례 설정
        MP_data = mlines.Line2D([], [], color=sns.xkcd_rgb["pale red"], label="MP for random part in data", linewidth=2)
        MP_rand = mlines.Line2D([], [], color=sns.xkcd_rgb["sap green"], label="MP for randomized data", linewidth=1.5, linestyle='--')
        randomized = mpatches.Patch(color=sns.xkcd_rgb["apple green"], label="Randomized data", alpha=0.75, linewidth=3, fill=False)
        data_real = mpatches.Patch(color=sns.xkcd_rgb["cornflower blue"], label="Real data", alpha=0.85)

        # 비교가 필요한 경우
        if comparison:
            sns.histplot(self.Lr.get(), bins=30, kde=False,
                        stat="density", color=sns.xkcd_rgb["apple green"], alpha=0.75, linewidth=3)
            
            ax.plot(x, yr, sns.xkcd_rgb["sap green"], lw=1.5, ls='--')

            ax.legend(handles=[data_real, MP_data, randomized, MP_rand], loc="upper right", frameon=True)
        else:
            ax.legend(handles=[data_real, MP_data], loc="upper right", frameon=True)

        # x축 범위 설정
        max_Lr = cp.max(self.Lr) if self.Lr is not None else 0
        max_L_mp = cp.max(self.L_mp) if self.L_mp is not None else 0
        ax.set_xlim([0, int(np.round(max(max_Lr, max_L_mp) + 1.5))])

        # 격자 스타일 설정
        ax.grid(linestyle='--', lw=0.3)

        # 제목 설정
        if title:
            ax.set_title(title)
        
        # x축 레이블 설정
        ax.set_xlabel('First cell eigenvalues normalized distribution')

        if self.data is not None and isinstance(self.data, sc.AnnData):
            self.data.uns['mp_plot'] = fig

        # if path:
        #     plt.savefig(path, bbox_inches="tight")
        return fig