# This is a Python library for Monte Carlo simulations in pricing an Asian option.
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.stats.qmc import Sobol
from scipy.stats import norm

def draw_pseudo_random_numbers(seed, N, M):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(size=(N, M))

def draw_quasi_random_numbers(seed, N, M):
    sobol_engine = Sobol(d=M, scramble=True, seed=seed)
    U = sobol_engine.random_base2(m=20)  # 2^20 
    return norm.ppf(np.clip(U, 1e-12, 1-1e-12)) # avoid inf values

def euler_path(Z, S0, r, sigma, gamma, T):
    N, M = Z.shape
    dt = T / M
    logS = np.empty((N, M + 1), dtype=float)
    logS[:, 0] = np.log(S0)

