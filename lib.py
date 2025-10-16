# This is a Python library for Monte Carlo simulations in pricing an Asian option.
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.stats.qmc import Sobol
from scipy.stats import norm, t


class MonteCarloSimulation:
    def __init__(self, S0, K, r, sigma, gamma, T, M, base_seed=42):
        """Initialize Monte Carlo simulation parameters.
        
        Args:
            S0 (float): Initial stock price
            K (float): Strike price
            r (float): Risk-free rate
            sigma (float): Volatility parameter
            gamma (float): CEV model parameter
            T (float): Time to maturity
            M (int): Number of time steps
            base_seed (int, optional): Base seed for random number generation. Defaults to 42.
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.gamma = gamma
        self.T = T
        self.M = M
        self.base_seed = base_seed
        self.dt = T / M
        self.sqrt_dt = np.sqrt(self.dt)

    def draw_pseudo_random_numbers(self, seed, N, M):
        """Generate pseudo-random standard normal variables."""
        rng = np.random.default_rng(seed)
        return rng.standard_normal(size=(N, M))

    def draw_quasi_random_numbers(self, seed, N, M):
        """Generate quasi-random numbers using Sobol sequence."""
        m = int(np.ceil(np.log2(N)))
        if 2**m != N:
            raise ValueError("For Sobol QMC, set N to 2**m.")
        sobol_engine = Sobol(d=M, scramble=True, seed=seed)
        U = sobol_engine.random_base2(m=m)
        return norm.ppf(np.clip(U, 1e-12, 1-1e-12))

    def draw_pseudo_antithetic_numbers(self, seed, N, M):
        """Generate antithetic pairs of random numbers."""
        rng = np.random.default_rng(seed)
        half_N = (N + 1) // 2
        Z_half = rng.standard_normal(size=(half_N, M))
        Z = np.vstack([Z_half, -Z_half])[:N]
        return Z

    def _cev_arith_from_Z(self, Z, euler=True):
        N = Z.shape[0]
        S = self.sim_CEV_paths(N, Z, euler=euler)
        avg = S[:, 1:].mean(axis=1)
        return np.exp(-self.r * self.T) * np.maximum(avg - self.K, 0.0)

    def diagnose_once(self, Z, euler=True, beta_fixed=None):
        # Target X
        X = self._cev_arith_from_Z(Z, euler)
        # Control Y (your GBM routine returns pathwise payoffs already)
        Y = self.sim_GBM_paths(Z.shape[0], Z)
        muY = self.geometric_asian_closed_form()

        if beta_fixed is None:
            Xc, Yc = X - X.mean(), Y - Y.mean()
            varY = Yc.var(ddof=1); beta = 0.0 if varY == 0 else (np.cov(X, Y, ddof=1)[0,1] / varY)
        else:
            beta = float(beta_fixed)

        adj = X + beta*(muY - Y)
        corr = float(np.corrcoef(X, Y)[0,1])
        vr = float(1.0 - adj.var(ddof=1) / X.var(ddof=1))  # variance reduction fraction
        return {"corr_XY": corr, "beta": float(beta), "VR_fraction": vr,
                "SE_plain": float(X.std(ddof=1)/np.sqrt(X.size)),
                "SE_cv": float(adj.std(ddof=1)/np.sqrt(X.size))}


    def sim_GBM_paths(self, N, Z):
        """Simulate Geometric Brownian Motion paths."""
        drift = (self.r - 0.5 * self.sigma**2) * self.dt
        vol   = self.sigma * self.sqrt_dt

        logS = np.full(N, np.log(self.S0), dtype=float)
        sum_logS = np.zeros(N, dtype=float)

        for n in range(self.M):
            logS += drift + vol * Z[:, n]
            sum_logS += logS

        G = np.exp(sum_logS / self.M)
        return np.exp(-self.r * self.T) * np.maximum(G - self.K, 0.0)

    def geometric_asian_closed_form(self):
        """
        Closed-form mean of the geometric-Asian call under GBM with discrete monitoring.
        """
        m = np.log(self.S0) + (self.r - 0.5 * self.sigma**2) * self.T * (self.M + 1) / (2 * self.M)
        v = (self.sigma**2) * self.T * ((self.M + 1) * (2 * self.M + 1)) / (6 * self.M**2)
        s = np.sqrt(max(v, 0.0))
        if s == 0.0:
            G = np.exp(m)
            return float(np.exp(-self.r*self.T) * max(G - self.K, 0.0))
        d1 = (m - np.log(self.K) + v) / s
        d2 = d1 - s
        undiscounted = np.exp(m + 0.5*v) * norm.cdf(d1) - self.K * norm.cdf(d2)
        return float(np.exp(-self.r*self.T) * undiscounted)

    def sim_CEV_paths(self, N, Z, euler=True):
        """Simulate CEV model paths."""
        logS = np.empty((N, self.M+1), dtype=float)
        logS[:, 0] = np.log(self.S0)

        for n in range(self.M):
            logS_n = logS[:, n]
            S_n = np.exp(logS_n)
            dW = self.sqrt_dt * Z[:, n]
            drift = (self.r - 0.5 * self.sigma**2 * np.power(S_n, 2*self.gamma-2)) * self.dt
            diffusion = self.sigma * np.power(S_n, self.gamma-1) * dW
            
            if euler:
                logS[:, n+1] = logS_n + drift + diffusion
            else:
                correction = 0.5 * self.sigma**2 * (self.gamma - 1) * np.power(S_n, 2*self.gamma-2) * (dW**2 - self.dt)
                logS[:, n+1] = logS_n + drift + diffusion + correction
                
        return np.exp(logS)

    def price_asian_option_from_Z(self, Z, euler=True):
        """Price Asian option using provided random numbers."""
        N = Z.shape[0]
        S = self.sim_CEV_paths(N, Z, euler=euler)
        avg = S[:, 1:].mean(axis=1)
        disc_payoff = np.exp(-self.r * self.T) * np.maximum(avg - self.K, 0.0)
        return float(disc_payoff.mean())

    def price_asian_option_cv_from_Z(self, Z, euler=True, beta_fixed=None):
        """
        Control variate pricing:
        - X: arithmetic Asian call under CEV (target), built from Z
        - Y: geometric Asian call under GBM (control), built from SAME Z
        - mu_Y: closed-form mean of Y
        If beta_fixed is None -> estimate beta in-sample (good for MC).
        If beta_fixed is set   -> use fixed beta (recommended for RQMC).
        """
        N = Z.shape[0]
        # Target payoff X (CEV arithmetic)
        S_cev = self.sim_CEV_paths(N, Z, euler=euler)          # (N, M+1)
        arith_avg = S_cev[:, 1:].mean(axis=1)
        X = np.exp(-self.r * self.T) * np.maximum(arith_avg - self.K, 0.0)

        # Control payoff Y (GBM geometric) using SAME Z (your GBM routine is streaming & returns Y directly)
        Y = self.sim_GBM_paths(N, Z)                           # (N,)

        mu_Y = self.geometric_asian_closed_form()

        if beta_fixed is None:
            # sample-optimal beta: Cov(X,Y)/Var(Y)
            Xbar, Ybar = X.mean(), Y.mean()
            Xc, Yc = X - Xbar, Y - Ybar
            varY = Yc.var(ddof=1)
            beta = 0.0 if varY == 0 else ( (Xc*Yc).mean() * N / max(N-1,1) ) / varY
        else:
            beta = float(beta_fixed)

        adj = X + beta * (mu_Y - Y)
        price = float(adj.mean())        
        return price

    def estimate_beta_pilot(self, Npilot, seed=None, euler=True):
        """
        Small IID MC pilot to get a fixed beta for RQMC.
        """
        seed = self.base_seed + 987654 if seed is None else seed
        Zp = self.draw_pseudo_random_numbers(seed, Npilot, self.M)
        N = Npilot

        # Build X and Y on the same Z
        S_cev = self.sim_CEV_paths(N, Zp, euler=euler)
        X = np.exp(-self.r * self.T) * np.maximum(S_cev[:,1:].mean(axis=1) - self.K, 0.0)
        Y = self.sim_GBM_paths(N, Zp)
        Xc, Yc = X - X.mean(), Y - Y.mean()
        varY = Yc.var(ddof=1)
        if varY == 0:
            return 0.0
        covXY = (Xc*Yc).mean() * N / max(N-1,1)
        return float(covXY / varY)

    @staticmethod
    def ci_from_replicates(estimates, alpha=0.05):
        """Calculate confidence interval from replicate estimates."""
        est = np.asarray(estimates, dtype=float)
        K = est.size
        mean = est.mean()
        var_between = est.var(ddof=1) if K > 1 else np.nan
        se_mean = np.sqrt(var_between / K) if K > 1 else np.nan
        
        df = K - 1
        if K > 1:
            tcrit = t.ppf(1 - alpha/2, df=df)
            ci = (mean - tcrit * se_mean, mean + tcrit * se_mean)
        else:
            ci = (np.nan, np.nan)
        return mean, se_mean, ci, var_between

    def rqmc_with_scrambles(self, N, scrambles, euler=True, CV=False, beta_fixed=None, pilot_N=16384, pilot_seed=None):
        """Run RQMC with multiple scrambles."""
        if beta_fixed is None:
            beta_fixed = self.estimate_beta_pilot(pilot_N, seed=pilot_seed, euler=euler)

        estimates = []
        for s in range(scrambles):
            Z = self.draw_quasi_random_numbers(self.base_seed + s, N, self.M)
            if not CV:
                estimates.append(self.price_asian_option_from_Z(Z, euler))
            else:
                estimates.append(self.price_asian_option_cv_from_Z(Z, euler, beta_fixed=beta_fixed))
        return self.ci_from_replicates(estimates)

    def mc_with_batches(self, N, batches, euler=True, CV=False):
        """Run standard MC with multiple batches."""
        estimates = []
        for b in range(batches):
            Z = self.draw_pseudo_random_numbers(self.base_seed + b, N, self.M)
            if not CV:
                estimates.append(self.price_asian_option_from_Z(Z, euler))
            else:
                estimates.append(self.price_asian_option_cv_from_Z(Z, euler, beta_fixed=None))
        return self.ci_from_replicates(estimates)

    def mc_with_antithetic(self, N, batches, euler=True, CV=False):
        """Run MC with antithetic variates."""
        estimates = []
        for b in range(batches):
            Z = self.draw_pseudo_antithetic_numbers(self.base_seed + b, N, self.M)
            if not CV:    
                estimates.append(self.price_asian_option_from_Z(Z, euler))
            else:
                estimates.append(self.price_asian_option_cv_from_Z(Z, euler, beta_fixed=None))
        return self.ci_from_replicates(estimates)

    def mc_with_antithetic_cv_pairavg(self, N, batches, euler=True):
        estimates = []
        muY = self.geometric_asian_closed_form()
        for b in range(batches):
            # Generate explicit pairs
            Z1 = self.draw_pseudo_random_numbers(self.base_seed + b, N//2, self.M)
            Z2 = -Z1

            X1 = self._cev_arith_from_Z(Z1, euler)
            X2 = self._cev_arith_from_Z(Z2, euler)
            Y1 = self.sim_GBM_paths(Z1.shape[0], Z1)
            Y2 = self.sim_GBM_paths(Z2.shape[0], Z2)

            # Pair-average
            Xp = 0.5*(X1 + X2)
            Yp = 0.5*(Y1 + Y2)

            # In-sample beta for MC
            Xc, Yc = Xp - Xp.mean(), Yp - Yp.mean()
            varY = Yc.var(ddof=1)
            beta = 0.0 if varY == 0 else (np.cov(Xp, Yp, ddof=1)[0,1] / varY)

            adj = Xp + beta*(muY - Yp)
            estimates.append(float(adj.mean()))
        return self.ci_from_replicates(estimates)

    def compare_methods(self, N, batches=8, euler=True):
        """Compare different Monte Carlo methods."""
        # Standard MC
        mc_mean, mc_se, mc_ci, mc_var = self.mc_with_batches(N, batches, euler)
        
        # RQMC
        qmc_mean, qmc_se, qmc_ci, qmc_var = self.rqmc_with_scrambles(N, batches, euler)
        
        # MC + Antithetic
        anti_mean, anti_se, anti_ci, anti_var = self.mc_with_antithetic(N, batches, euler)
        
        # MC + CV
        mc_cv_mean, mc_cv_se, mc_cv_ci, mc_cv_var = self.mc_with_batches(N, batches, euler, CV=True)
        # MC + CV + Antithetic
        mc_cv_anti_mean, mc_cv_anti_se, mc_cv_anti_ci, mc_cv_anti_var = self.mc_with_antithetic(N, batches, euler, CV=True)
        # CV RQMC
        qmc_cv_mean, qmc_cv_se, qmc_cv_ci, qmc_cv_var = self.rqmc_with_scrambles(N, batches, euler, CV=True)


        results = {
            'Standard MC': {'mean': mc_mean, 'SE': mc_se, 'CI': mc_ci, 'var': mc_var},
            'Standard MC + Antithetic': {'mean': anti_mean, 'SE': anti_se, 'CI': anti_ci, 'var': anti_var},
            'Standard MC + CV': {'mean': mc_cv_mean, 'SE': mc_cv_se, 'CI': mc_cv_ci, 'var': mc_cv_var},
            'Standard MC + CV + Antithetic': {'mean': mc_cv_anti_mean, 'SE': mc_cv_anti_se, 'CI': mc_cv_anti_ci, 'var': mc_cv_anti_var},
            'RQMC': {'mean': qmc_mean, 'SE': qmc_se, 'CI': qmc_ci, 'var': qmc_var},
            'RQMC + CV': {'mean': qmc_cv_mean, 'SE': qmc_cv_se, 'CI': qmc_cv_ci, 'var': qmc_cv_var}
        }
    
        return results

    def plot_convergence(self, N_values, method='mc', batches=8, euler=True):
        """Plot convergence of different methods."""
        errors = []
        prices = []
        
        method_map = {
            'mc': self.mc_with_batches,
            'mc + antithetic': self.mc_with_antithetic,
            'mc + cv': self.mc_with_batches(N, batches, euler, CV=True),
            'mc + cv + antithetic': self.mc_with_antithetic(N, batches, euler, CV=True),
            'qmc': self.rqmc_with_scrambles,
            'qmc + cv': self.rqmc_with_scrambles(N, batches, euler, CV=True)
        }
        
        if method not in method_map:
            raise ValueError(f"Method must be one of {list(method_map.keys())}")
            
        simulation_method = method_map[method]
        
        for N in N_values:
            mean, se, _, _ = simulation_method(N, batches, euler)
            prices.append(mean)
            errors.append(se)
            
        return prices, errors