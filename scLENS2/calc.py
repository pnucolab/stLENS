import scanpy as sc
import cupy as cp
import pandas as pd
import numpy as np
from scipy import stats, linalg
import scipy

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

class Calc():
    def __init__(self,device = None, data = None, L=None, L_mp=None):
        
        self.L = []
        self.V = None
        self.L_mp = None
        self.explained_variance_ = []
        self.total_variance_ = []
        self.device = device
        self.data = data

    def style_mp_stat(self):
        plt.style.use("ggplot")
        # np.seterr(invalid='ßignore')
        np.seterr(invalid='ignore')
        sns.set_style("white")
        sns.set_context("paper")
        sns.set_palette("deep")
        plt.rcParams['axes.linewidth'] =0.5
        plt.rcParams['figure.dpi'] = 100

    def _tw(self):
        '''Tracy-Widom critical eigenvalue'''
        if self.L is None or len(self.L) == 0:
            raise ValueError("self.L must not be empty or None.")
        if self.L_mp is None or len(self.L_mp) == 0:
            raise ValueError("self.L_mp must not be empty or None.")

        gamma = self._mp_parameters(self.L_mp)['gamma']
        p = len(self.L) / gamma
        sigma = 1 / cp.power(p, 2/3) * cp.power(gamma, 5/6) * \
            cp.power((1 + cp.sqrt(gamma)), 4/3)
        lambda_c = cp.mean(self.L_mp) * (1 + cp.sqrt(gamma)) ** 2 + sigma
        return lambda_c

    def _mp_parameters(self, L):
        moment_1 = cp.mean(L)
        moment_2 = cp.mean(cp.power(L, 2))
        gamma = moment_2 / float(moment_1**2) - 1
        s = moment_1
        sigma = moment_2
        b_plus = s * (1 + cp.sqrt(gamma))**2
        b_minus = s * (1 - cp.sqrt(gamma))**2
        x_peak = s * (1.0 - gamma)**2.0 / (1.0 + gamma)
        return {
            'moment_1': moment_1,
            'moment_2': moment_2,
            'gamma': gamma, 
            'b_plus': b_plus,
            'b_minus': b_minus,
            's': s,
            'peak': x_peak,
            'sigma': sigma
        }

    def _marchenko_pastur(self, x, dic):
        '''Distribution of eigenvalues'''
        pdf = np.sqrt((dic['b_plus'] - x) * (x-dic['b_minus']))\
            / float(2 * dic['s'] * np.pi * dic['gamma'] * x)
        return pdf

    def _mp_pdf(self, x, L):
        '''Marchnko-Pastur PDF'''
        dic = self._mp_parameters(L)
        y = cp.empty_like(x)
        for i, xi in enumerate(x):
            y[i] = self._marchenko_pastur(xi, dic)
        return y

    def _mp_calculation(self, L, Lr, eta=1, eps=10**-6, max_iter=1000):
        converged = False
        iter = 0
        loss_history = []
    
        b_plus = self._mp_parameters(Lr)['b_plus']
        b_minus = self._mp_parameters(Lr)['b_minus']
        
        L_updated = L[(L > b_minus) & (L < b_plus)]
        new_b_plus = self._mp_parameters(L_updated)['b_plus']
        new_b_minus = self._mp_parameters(L_updated)['b_minus']
    
        while not converged:
            loss = (1 - float(new_b_plus) / float(b_plus))**2
            loss_history.append(loss)
            iter += 1
        
            if loss <= eps:
                converged = True
            elif iter == max_iter:
                print('Max interactions exceeded!')
                converged = True
            else:
                gradient = new_b_plus - b_plus
                new_b_plus = b_plus + eta * gradient
                
                L_updated = L[(L > new_b_minus) & (L < new_b_plus)]
                self.b_plus = new_b_plus
                self.b_minus = new_b_minus
                
                new_b_plus = self._mp_parameters(L_updated)['b_plus']
                new_b_minus = self._mp_parameters(L_updated)['b_minus']
    
        indices = cp.where((L > new_b_minus) & (L < new_b_plus))
        L_mp = L[indices]
        return cp.array(L_mp)

    def _cdf_marchenko(self,x,dic):
        if x < dic['b_minus']: 
            return 0.0
        elif x>dic['b_minus'] and x<dic['b_plus']:
            return 1/float(2*dic['s']*np.pi*dic['gamma'])*\
            float(np.sqrt((dic['b_plus']-x)*(x-dic['b_minus']))+\
            (dic['b_plus']+dic['b_minus'])/2*np.arcsin((2*x-dic['b_plus']-\
            dic['b_minus'])/(dic['b_plus']-dic['b_minus']))-\
        np.sqrt(dic['b_plus']*dic['b_minus'])*np.arcsin(((dic['b_plus']+\
            dic['b_minus'])*x-2*dic['b_plus']*dic['b_minus'])/\
        ((dic['b_plus']-dic['b_minus'])*x)) )+np.arcsin(1)/np.pi
        else:
            return 1.0

    def _call_mp_cdf(self,L,dic):
        "CDF of Marchenko Pastur"
        func= lambda y: list(map(lambda x: self._cdf_marchenko(x,dic), y))
        return func