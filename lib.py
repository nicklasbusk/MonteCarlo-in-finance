import numpy as np
import matplotlib.pyplot as plt
from scipy.stats.qmc import Sobol
from scipy.stats import norm, t
import time


class MonteCarloSimulation:
    def __init__(self, S0, K, r, sigma, gamma, T, M, base_seed=42):
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
        """Generate standard normal pseudo-random draws."""
        rng = np.random.default_rng(seed)
        return rng.standard_normal(size=(N, M))

    def draw_quasi_random_numbers(self, seed, N, M):
        """Generate Sobol-based quasi-random normals."""
        m = int(np.ceil(np.log2(N)))
        if 2**m != N:
            raise ValueError("For Sobol QMC, set N to 2**m.")
        sobol_engine = Sobol(d=M, scramble=True, seed=seed)
        U = sobol_engine.random_base2(m=m)
        # Avoid extreme quantiles that can blow up norm.ppf
        epsilon = 1e-10
        U = np.clip(U, epsilon, 1.0 - epsilon)
        return norm.ppf(U)

    @staticmethod
    def dyadic_midpoint_order(M):
        """Return dyadic midpoint order for Brownian-bridge construction."""
        order = []
        stack = [(0, M)]
        while stack:
            l, r = stack.pop()
            if r - l <= 1:
                continue
            mid = (l + r) // 2
            order.append(mid)
            # push right then left so left is processed next
            stack.append((mid, r))
            stack.append((l, mid))
        return order

    def draw_quasi_random_numbers_bb(self, seed, N, M, use_antithetic=False):
        """Sobol + Brownian-bridge normals for time-ordered increments."""
        epsilon = 1e-10
        
        if use_antithetic:
            if N < 2:
                raise ValueError("For antithetic pairing, N must be at least 2.")

            m = int(np.ceil(np.log2(N)))
            sobol_engine = Sobol(d=M, scramble=True, seed=seed)
            U_half = sobol_engine.random_base2(m=m-1)
            U = np.vstack([U_half, 1.0 - U_half])
            U = np.clip(U, epsilon, 1.0 - epsilon)
            Z_std = norm.ppf(U)
        else:
            m = int(np.ceil(np.log2(N)))
            if 2**m != N:
                raise ValueError("For Sobol QMC, set N to 2**m.")
            sobol_engine = Sobol(d=M, scramble=True, seed=seed)
            U = sobol_engine.random_base2(m=m)
            U = np.clip(U, epsilon, 1.0 - epsilon)
            Z_std = norm.ppf(U)  # shape (N, M)

        # Brownian-bridge mapping -> W(t) in dyadic order, then to increments
        Np = N
        dt = self.dt
        sqrt_dt = self.sqrt_dt
        Mloc = M

        order = self.dyadic_midpoint_order(Mloc)

        sampled = np.empty((Np, Mloc+1), dtype=float)
        sampled.fill(np.nan)
        sampled[:, 0] = 0.0

        sampled[:, Mloc] = Z_std[:, 0] * np.sqrt(self.T)

        z_col = 1
        for mpos in order:
            # Find nearest filled neighbors l < mpos < r (same for all paths
            # because positions are filled deterministically).
            l = mpos - 1
            while l >= 0 and np.isnan(sampled[0, l]):
                l -= 1
            r = mpos + 1
            while r <= Mloc and np.isnan(sampled[0, r]):
                r += 1

            denom = float(r - l)
            mean = (sampled[:, l] * (r - mpos) + sampled[:, r] * (mpos - l)) / denom
            var = ((mpos - l) * (r - mpos) / denom) * dt
            sd = np.sqrt(max(var, 0.0))

            sampled[:, mpos] = mean + Z_std[:, z_col] * sd
            z_col += 1

        Wvals = sampled[:, 1:]
        Wprev = np.concatenate([np.zeros((Np, 1), dtype=float), Wvals[:, :-1]], axis=1)
        dW = Wvals - Wprev
        Z_increments = dW / sqrt_dt
        return Z_increments

    def draw_pseudo_antithetic_numbers(self, seed, N, M):
        """Generate antithetic pseudo-random normals."""
        rng = np.random.default_rng(seed)
        half_N = (N + 1) // 2
        Z_half = rng.standard_normal(size=(half_N, M))
        Z = np.vstack([Z_half, -Z_half])[:N]
        return Z
    
    def draw_quasi_antithetic_numbers(self, seed, N, M):
        """Sobol normals with antithetic pairing."""
        if N < 2:
            raise ValueError("N must be at least 2.")

        m = int(np.ceil(np.log2(N)))
        if 2**m != N:
            raise ValueError("set N to 2**m.")

        sobol_engine = Sobol(d=M, scramble=True, seed=seed)
        U_half = sobol_engine.random_base2(m=m-1)

        U = np.vstack([U_half, 1.0 - U_half])
        
        # Avoid extreme quantiles that can blow up norm.ppf
        epsilon = 1e-10
        U = np.clip(U, epsilon, 1.0 - epsilon)

        return norm.ppf(U)

    def _cev_arith_from_Z(self, Z, euler=False):
        N = Z.shape[0]
        S = self.sim_CEV_paths(N, Z, euler=euler)
        avg = S[:, 1:].mean(axis=1)
        return np.exp(-self.r * self.T) * np.maximum(avg - self.K, 0.0)

    def sim_GBM_paths(self, N, Z):
        """Simulate GBM paths and return discounted payoffs."""
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
        """Closed-form mean for a discrete geometric-Asian call under GBM."""
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

  
    def sim_CEV_paths(self, N, Z, euler=False):
        """Simulate CEV paths."""
        # Recompute in case self.M was changed externally
        dt = self.T / self.M
        sqrt_dt = np.sqrt(dt)

        logS = np.empty((N, self.M+1), dtype=float)
        logS[:, 0] = np.log(self.S0)

        S_floor = 1e-16

        for n in range(self.M):
            logS_n = logS[:, n]
            S_n = np.exp(logS_n)
            S_n_safe = np.maximum(S_n, S_floor)
            dW = sqrt_dt * Z[:, n]

            log_S_n_safe = np.log(S_n_safe)
            exp1 = (2.0 * self.gamma - 2.0) * log_S_n_safe
            exp2 = (self.gamma - 1.0) * log_S_n_safe
            pow_term = np.exp(exp1)
            pow_term_diff = np.exp(exp2)

            drift = (self.r - 0.5 * self.sigma**2 * pow_term) * dt
            diffusion = self.sigma * pow_term_diff * dW

            if euler:
                logS_next = logS_n + drift + diffusion
            else:
                correction = 0.5 * self.sigma**2 * (self.gamma - 1.0) * pow_term * (dW**2 - dt)
                logS_next = logS_n + drift + diffusion + correction

            logS[:, n+1] = logS_next
            
        return np.exp(logS)

    def price_asian_option_from_Z(self, Z, euler=False):
        """Price arithmetic Asian call using provided normals."""
        N = Z.shape[0]
        S = self.sim_CEV_paths(N, Z, euler=euler)
        avg = S[:, 1:].mean(axis=1)
        disc_payoff = np.exp(-self.r * self.T) * np.maximum(avg - self.K, 0.0)
        return float(disc_payoff.mean())

    def price_asian_option_from_Z_raw(self, Z, euler=False):
        """Return discounted payoffs for the arithmetic Asian call."""
        N = Z.shape[0]
        S = self.sim_CEV_paths(N, Z, euler=euler)
        avg = S[:, 1:].mean(axis=1)
        disc_payoff = np.exp(-self.r * self.T) * np.maximum(avg - self.K, 0.0)
        return disc_payoff

    def _payoffs_from_Z_with_params(self, Z, euler=False, S0_override=None, sigma_override=None):
        """Return discounted payoffs with optional S0/sigma overrides."""
        N = Z.shape[0]
        S0 = self.S0 if S0_override is None else float(S0_override)
        sigma = self.sigma if sigma_override is None else float(sigma_override)

        logS = np.empty((N, self.M+1), dtype=float)
        logS[:, 0] = np.log(S0)
        # Recompute in case self.M was changed externally
        dt = self.T / self.M
        sqrt_dt = np.sqrt(dt)

        S_floor = 1e-16

        for n in range(self.M):
            logS_n = logS[:, n]
            S_n = np.exp(logS_n)
            S_n_safe = np.maximum(S_n, S_floor)
            dW = sqrt_dt * Z[:, n]

            log_S_n_safe = np.log(S_n_safe)
            exp1 = (2.0 * self.gamma - 2.0) * log_S_n_safe
            exp2 = (self.gamma - 1.0) * log_S_n_safe
            pow_term = np.exp(exp1)
            pow_term_diff = np.exp(exp2)

            drift = (self.r - 0.5 * sigma**2 * pow_term) * dt
            diffusion = sigma * pow_term_diff * dW

            if euler:
                logS_next = logS_n + drift + diffusion
            else:
                correction = 0.5 * sigma**2 * (self.gamma - 1.0) * pow_term * (dW**2 - dt)
                logS_next = logS_n + drift + diffusion + correction

            logS[:, n+1] = logS_next

        S = np.exp(logS)
        arith_avg = S[:, 1:].mean(axis=1)
        return np.exp(-self.r * self.T) * np.maximum(arith_avg - self.K, 0.0)



    def price_asian_option_cv_from_Z(self, Z, euler=False, beta_fixed=None):
        """Control-variate pricing with geometric Asian under GBM."""
        N = Z.shape[0]
        S_cev = self.sim_CEV_paths(N, Z, euler=euler)          # (N, M+1)
        arith_avg = S_cev[:, 1:].mean(axis=1)
        X = np.exp(-self.r * self.T) * np.maximum(arith_avg - self.K, 0.0)

        Y = self.sim_GBM_paths(N, Z)                           # (N,)

        mu_Y = self.geometric_asian_closed_form()

        if beta_fixed is None:
            Xbar, Ybar = X.mean(), Y.mean()
            Xc, Yc = X - Xbar, Y - Ybar
            varY = Yc.var(ddof=1)
            beta = 0.0 if varY == 0 else ( (Xc*Yc).mean() * N / max(N-1,1) ) / varY
        else:
            beta = float(beta_fixed)

        adj = X + beta * (mu_Y - Y)
        price = float(adj.mean())        
        return price


    def estimate_beta_pilot_rqmc(self, Npilot, scrambles=4, euler=False, pilot_seed_offset=100000, use_bb=True):
        """Estimate a fixed beta using independent RQMC pilots."""
        betas = []
        for s in range(scrambles):
            seed = self.base_seed + pilot_seed_offset + s
            if use_bb:
                Z = self.draw_quasi_random_numbers_bb(seed, Npilot, self.M)
            else:
                Z = self.draw_quasi_random_numbers(seed, Npilot, self.M)
            S_cev = self.sim_CEV_paths(Npilot, Z, euler=euler)
            X = np.exp(-self.r*self.T)*np.maximum(S_cev[:,1:].mean(1)-self.K, 0.0)
            Y = self.sim_GBM_paths(Npilot, Z)
            Xc, Yc = X - X.mean(), Y - Y.mean()
            denom = (Yc**2).sum()
            if denom == 0 or not np.isfinite(denom):
                beta_s = 0.0
            else:
                beta_s = (Xc*Yc).sum() / denom
            betas.append(beta_s)
        return float(np.mean(betas))


    @staticmethod
    def ci_from_replicates(estimates, alpha=0.05):
        """Confidence interval from replicate estimates."""
        est = np.asarray(estimates, dtype=float)
        K = est.size
        mean = est.mean()
        var_between = est.var(ddof=1) if K > 1 else np.nan
        se_mean = np.sqrt(var_between / K) if K > 1 else np.nan
        
        df = K - 1
        if K > 1:
            tcrit = t.ppf(1 - alpha/2, df=df)
            ci = ((mean - tcrit * se_mean).round(4), (mean + tcrit * se_mean).round(4))
        else:
            ci = (np.nan, np.nan)
        return mean, se_mean, ci, var_between

    def rqmc_with_scrambles(self, N, scrambles, euler=False, CV=False, beta_fixed=None,
                            pilot_N=16384, pilot_scrambles=4, pilot_seed_offset=100000,
                            use_bb=False, use_antithetic=False):
        """Run RQMC with Owen scrambles and optional control variates."""

        # If CV requested but no fixed beta provided, estimate pilot beta once.
        if CV and beta_fixed is None:
            beta_fixed = self.estimate_beta_pilot_rqmc(pilot_N, scrambles=pilot_scrambles,
                                                     euler=euler, pilot_seed_offset=pilot_seed_offset)
            #print(f"Estimated fixed beta from pilot RQMC: {beta_fixed:.6f}")

        estimates = []
        for s in range(scrambles):
            seed = self.base_seed + s
            if use_bb:
                if use_antithetic:
                    Z = self.draw_quasi_random_numbers_bb(seed, N, self.M, use_antithetic=True)
                else:
                    Z = self.draw_quasi_random_numbers_bb(seed, N, self.M)
            else:
                if use_antithetic:
                    Z = self.draw_quasi_antithetic_numbers(seed, N, self.M)
                else:
                    Z = self.draw_quasi_random_numbers(seed, N, self.M)
            if not CV:
                estimates.append(self.price_asian_option_from_Z(Z, euler))
            else:
                estimates.append(self.price_asian_option_cv_from_Z(Z, euler, beta_fixed=beta_fixed))

        mean, se, ci, var_between = self.ci_from_replicates(estimates)
        return mean, se, ci, var_between

    def mc_with_batches(self, N, batches, euler=False, CV=False):
        """Run standard MC with multiple batches."""
        estimates = []
        for b in range(batches):
            Z = self.draw_pseudo_random_numbers(self.base_seed + b, N, self.M)
            if not CV:
                estimates.append(self.price_asian_option_from_Z(Z, euler))
            else:
                estimates.append(self.price_asian_option_cv_from_Z(Z, euler, beta_fixed=None))
        return self.ci_from_replicates(estimates)

    def mc_with_antithetic(self, N, batches, euler=False, CV=False):
        """Run MC with antithetic variates."""
        estimates = []
        for b in range(batches):
            Z = self.draw_pseudo_antithetic_numbers(self.base_seed + b, N, self.M)
            if not CV:    
                estimates.append(self.price_asian_option_from_Z(Z, euler))
            else:
                estimates.append(self.price_asian_option_cv_from_Z(Z, euler, beta_fixed=None))
        return self.ci_from_replicates(estimates)
    

    def compare_methods(self, N, batches=8, euler=False):
        """Compare MC, RQMC, and variance-reduction variants."""
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

        # RQMC + BB
        qmc_bb_mean, qmc_bb_se, qmc_bb_ci, qmc_bb_var = self.rqmc_with_scrambles(N, batches, euler, CV=False, use_antithetic=False, use_bb=True)
        # RQMC + BB + Antithetic
        qmc_cv_mean, qmc_cv_se, qmc_cv_ci, qmc_cv_var = self.rqmc_with_scrambles(N, batches, euler, CV=False, use_antithetic=True, use_bb=True) 
        # RQMC + BB + CV
        qmc_cv_bb_mean, qmc_cv_bb_se, qmc_cv_bb_ci, qmc_cv_bb_var = self.rqmc_with_scrambles(N, batches, euler, CV=True, use_antithetic=False, use_bb=True)
        # RQMC + BB + CV + Antithetic
        qmc_cv_anti_mean, qmc_cv_anti_se, qmc_cv_anti_ci, qmc_cv_anti_var = self.rqmc_with_scrambles(N, batches, euler, CV=True, use_antithetic=True, use_bb=True)

        results = {
            'Standard MC': {'mean': mc_mean, 'SE': mc_se, 'CI': mc_ci, 'var': mc_var},
            'Standard MC + Antithetic': {'mean': anti_mean, 'SE': anti_se, 'CI': anti_ci, 'var': anti_var},
            'Standard MC + CV': {'mean': mc_cv_mean, 'SE': mc_cv_se, 'CI': mc_cv_ci, 'var': mc_cv_var},
            'Standard MC + CV + Antithetic': {'mean': mc_cv_anti_mean, 'SE': mc_cv_anti_se, 'CI': mc_cv_anti_ci, 'var': mc_cv_anti_var},
            'RQMC + BB': {'mean': qmc_bb_mean, 'SE': qmc_bb_se, 'CI': qmc_bb_ci, 'var': qmc_bb_var},
            'RQMC + BB + Antithetic': {'mean': qmc_cv_mean, 'SE': qmc_cv_se, 'CI': qmc_cv_ci, 'var': qmc_cv_var},
            'RQMC + BB + CV': {'mean': qmc_cv_bb_mean, 'SE': qmc_cv_bb_se, 'CI': qmc_cv_bb_ci, 'var': qmc_cv_bb_var},
            'RQMC + BB + CV + Antithetic': {'mean': qmc_cv_anti_mean, 'SE': qmc_cv_anti_se, 'CI': qmc_cv_anti_ci, 'var': qmc_cv_anti_var},
        }
    
        return results

    def plot_convergence(self, N_values, method='mc', batches=8, euler=False):
        """Compute convergence metrics for a given method."""
        errors = []
        prices = []
        times = []
        cis = []
        
        # Map method name -> (callable, cv_flag[, extra_kwargs])
        method_map = {
            'mc': (self.mc_with_batches, False),
            'mc + antithetic': (self.mc_with_antithetic, False),
            'mc + cv': (self.mc_with_batches, True),
            'mc + cv + antithetic': (self.mc_with_antithetic, True),

            # plain QMC
            'qmc': (self.rqmc_with_scrambles, False),
            'qmc + cv': (self.rqmc_with_scrambles, True),
            'qmc + antithetic': (self.rqmc_with_scrambles, False, {'use_antithetic': True}),
            'qmc + cv + antithetic': (self.rqmc_with_scrambles, True, {'use_antithetic': True}),
            # Support Brownian-bridge variants
            'qmc + bb': (self.rqmc_with_scrambles, False, {'use_bb': True}),
            'qmc + bb + antithetic': (self.rqmc_with_scrambles, False, {'use_bb': True, 'use_antithetic': True}),
            'qmc + bb + cv': (self.rqmc_with_scrambles, True, {'use_bb': True}),
            'qmc + bb + cv + antithetic': (self.rqmc_with_scrambles, True, {'use_bb': True, 'use_antithetic': True}),
        }
        
        if method not in method_map:
            raise ValueError(f"Method must be one of {list(method_map.keys())}")

        entry = method_map[method]
        if len(entry) == 2:
            func, cv_flag = entry
            extra_kwargs = {}
        else:
            func, cv_flag, extra_kwargs = entry

        for N in N_values:
            t0 = time.perf_counter()
            mean, se, ci, _ = func(N, batches, euler, CV=cv_flag, **extra_kwargs)
            t1 = time.perf_counter()
            prices.append(mean)
            errors.append(se)
            cis.append(ci)
            times.append(float(t1 - t0))

        return prices, errors, cis, times
    

    def delta_from_Z_fd(self, Z, euler=False, rel_bump=0.001, scheme="central", beta=None):
        """Finite-difference Delta with common random numbers."""
        h = max(rel_bump*self.S0, 1e-12)
    
        if scheme == "central":
            up = self._payoffs_from_Z_with_params(Z, euler=euler, S0_override=self.S0 + h)
            dn = self._payoffs_from_Z_with_params(Z, euler=euler, S0_override=self.S0 - h)
            g = (up - dn) / (2.0 * h)
        elif scheme == "forward":
            up = self._payoffs_from_Z_with_params(Z, euler=euler, S0_override=self.S0 + h)
            base = self._payoffs_from_Z_with_params(Z, euler=euler, S0_override=self.S0)
            g = (up - base) / h
        else:
            raise ValueError("scheme must be 'central' or 'forward'")
        delta_mean = float(g.mean())
        delta_se = float(g.std(ddof=1) / np.sqrt(g.size))
        return delta_mean, delta_se, g
    
    def vega_from_Z_fd_cv(self, Z, euler=False, abs_bump=0.001, scheme="central", beta=None):
        """Finite-difference Vega with control variates and CRN."""
        h = max(abs_bump*self.sigma, 1e-12)
        if scheme == "central":
            up = self._payoffs_from_Z_with_params(Z, euler=euler, sigma_override=self.sigma + h)
            dn = self._payoffs_from_Z_with_params(Z, euler=euler, sigma_override=self.sigma - h)
            g = (up - dn) / (2.0 * h)
        elif scheme == "forward":
            up = self._payoffs_from_Z_with_params(Z, euler=euler, sigma_override=self.sigma + h)
            base = self._payoffs_from_Z_with_params(Z, euler=euler, sigma_override=self.sigma)
            g = (up - base) / h
        else:
            raise ValueError("scheme must be 'central' or 'forward'")

        N = Z.shape[0]
        drift = (self.r - 0.5 * self.sigma**2) * self.dt
        vol   = self.sigma * self.sqrt_dt

        logS = np.full(N, np.log(self.S0), dtype=float)
        sum_logS = np.zeros(N, dtype=float)

        for n in range(self.M):
            logS += drift + vol * Z[:, n]
            sum_logS += logS

        G = np.exp(sum_logS / self.M)
        dG_dsigma = (G * (sum_logS / self.M - np.log(self.S0) - (self.r - 0.5*self.sigma**2)*self.T) * self.T) / (self.sigma * self.M)
        geo_vega_paths = np.exp(-self.r * self.T) * (G > self.K) * dG_dsigma

        if beta is None:
            g_bar = g.mean()
            geo_bar = geo_vega_paths.mean()
            g_c = g - g_bar
            geo_c = geo_vega_paths - geo_bar
            var_geo = geo_c.var(ddof=1)
            beta = 0.0 if var_geo == 0 else ( (g_c*geo_c).mean() * N / max(N-1,1) ) / var_geo
        else:
            beta = float(beta)  
        adj = g + beta * (geo_vega_paths.mean() - geo_vega_paths)
        vega_mean = float(adj.mean())
        vega_se = float(adj.std(ddof=1) / np.sqrt(adj.size))
        return vega_mean, vega_se, adj


    def delta_from_Z_fd_cv(self, Z, euler=False, rel_bump=0.001, scheme="central", beta=None):
        """Finite-difference Delta with control variates and CRN."""
        h = max(rel_bump*self.S0, 1e-12)
        if h-self.sigma < 1e-12:
            h = 1e-12
        if scheme == "central":
            up = self._payoffs_from_Z_with_params(Z, euler=euler, S0_override=self.S0 + h)
            dn = self._payoffs_from_Z_with_params(Z, euler=euler, S0_override=self.S0 - h)
            g = (up - dn) / (2.0 * h)
        elif scheme == "forward":
            up = self._payoffs_from_Z_with_params(Z, euler=euler, S0_override=self.S0 + h)
            base = self._payoffs_from_Z_with_params(Z, euler=euler, S0_override=self.S0)
            g = (up - base) / h
        else:
            raise ValueError("scheme must be 'central' or 'forward'")

        N = Z.shape[0]
        drift = (self.r - 0.5 * self.sigma**2) * self.dt
        vol   = self.sigma * self.sqrt_dt

        logS = np.full(N, np.log(self.S0), dtype=float)
        sum_logS = np.zeros(N, dtype=float)

        for n in range(self.M):
            logS += drift + vol * Z[:, n]
            sum_logS += logS

        G = np.exp(sum_logS / self.M)
        dG_dS0 = (G / self.S0)  # pathwise derivative of geometric average w.r.t. S0
        geo_delta_paths = np.exp(-self.r * self.T) * (G > self.K) * dG_dS0

        if beta is None:
            g_bar = g.mean()
            geo_bar = geo_delta_paths.mean()
            g_c = g - g_bar
            geo_c = geo_delta_paths - geo_bar
            var_geo = geo_c.var(ddof=1)
            beta = 0.0 if var_geo == 0 else ( (g_c*geo_c).mean() * N / max(N-1,1) ) / var_geo
        else:
            beta = float(beta)  
        adj = g + beta * (geo_delta_paths.mean() - geo_delta_paths)
        delta_mean = float(adj.mean())
        delta_se = float(adj.std(ddof=1) / np.sqrt(adj.size))
        return delta_mean, delta_se, adj

            

    def vega_from_Z_fd(self, Z, euler=False, abs_bump=0.001, scheme="central"):
        """Finite-difference Vega with common random numbers."""
        h = max(abs_bump*self.sigma, 1e-12)

        if scheme == "central":
            up = self._payoffs_from_Z_with_params(Z, euler=euler, sigma_override=self.sigma + h)
            dn = self._payoffs_from_Z_with_params(Z, euler=euler, sigma_override=self.sigma - h)
            g = (up - dn) / (2.0 * h)
        elif scheme == "forward":
            up = self._payoffs_from_Z_with_params(Z, euler=euler, sigma_override=self.sigma + h)
            base = self._payoffs_from_Z_with_params(Z, euler=euler, sigma_override=self.sigma)
            g = (up - base) / h
        else:
            raise ValueError("scheme must be 'central' or 'forward'")
        vega_mean = float(g.mean())
        vega_se = float(g.std(ddof=1) / np.sqrt(g.size))
        return vega_mean, vega_se, g


    def mc_greeks_with_batches(self, N, batches=8, euler=False, use_antithetic=False, CV=False, rel_bump=0.001, abs_bump=0.001, scheme="central"):
        """Greeks via MC batches with CRN (Delta, Vega)."""
        delta_repl = []
        vega_repl = []
        for b in range(batches):
            if use_antithetic:
                Z = self.draw_pseudo_antithetic_numbers(self.base_seed + b, N, self.M)
            else:
                Z = self.draw_pseudo_random_numbers(self.base_seed + b, N, self.M)
            if CV:
                d_mean, _, _  = self.delta_from_Z_fd_cv(Z, euler=euler, rel_bump=rel_bump, scheme=scheme,beta=None)
                v_mean,  _, _  = self.vega_from_Z_fd_cv(Z, euler=euler, abs_bump=abs_bump, scheme=scheme,beta=None)
            else:
                d_mean, _, _  = self.delta_from_Z_fd(Z, euler=euler, rel_bump=rel_bump, scheme=scheme)
                v_mean,  _, _  = self.vega_from_Z_fd(Z, euler=euler, abs_bump=abs_bump, scheme=scheme)
            delta_repl.append(d_mean)
            vega_repl.append(v_mean)

        d_mean, d_se, d_ci, d_var_between = self.ci_from_replicates(delta_repl)
        v_mean, v_se, v_ci, v_var_between = self.ci_from_replicates(vega_repl)

        return (
            d_mean, d_se, d_ci, d_var_between, np.asarray(delta_repl, dtype=float)
        ), (
            v_mean, v_se, v_ci, v_var_between, np.asarray(vega_repl, dtype=float)
        )

    def rqmc_greeks_with_scrambles(self, N, scrambles=8, euler=False, use_antithetic=False, CV=False, use_bb=True,rel_bump=0.001, abs_bump=0.001,scheme="central"):
        """Greeks via RQMC scrambles (Delta, Vega)."""
        delta_repl = []
        vega_repl = []
        for s in range(scrambles):
            seed = self.base_seed + s
            if use_bb:
                if use_antithetic:
                    Z = self.draw_quasi_random_numbers_bb(seed, N, self.M, use_antithetic=True)
                else:
                    Z = self.draw_quasi_random_numbers_bb(seed, N, self.M)
            else:
                if use_antithetic:
                    Z = self.draw_quasi_antithetic_numbers(seed, N, self.M)
                else:
                    Z = self.draw_quasi_random_numbers(seed, N, self.M)
            if CV:
                d_mean, _, _  = self.delta_from_Z_fd_cv(Z, euler=euler, rel_bump=rel_bump, scheme=scheme, beta=None)
                v_mean, _, _  = self.vega_from_Z_fd_cv(Z, euler=euler, abs_bump=abs_bump, scheme=scheme, beta=None)
            else:
                d_mean, _,_  = self.delta_from_Z_fd(Z, euler=euler, rel_bump=rel_bump, scheme=scheme)
                v_mean, _, _  = self.vega_from_Z_fd(Z, euler=euler, abs_bump=abs_bump, scheme=scheme)
            delta_repl.append(d_mean)
            vega_repl.append(v_mean)

        d_mean, d_se, d_ci, d_var_between = self.ci_from_replicates(delta_repl)
        v_mean, v_se, v_ci, v_var_between = self.ci_from_replicates(vega_repl)
        return (
            d_mean, d_se, d_ci, d_var_between, np.asarray(delta_repl, dtype=float)
        ), (
            v_mean, v_se, v_ci, v_var_between, np.asarray(vega_repl, dtype=float)
        )
    
    def compare_greeks_methods(self, N_values, batches=8, euler=False, method='mc', Analysis='Delta', bump_size=0.001, scheme='central'):
        """Compare Greek estimators across methods."""

        errors_delta = []
        prices_delta = []
        times_delta = []
        cis_delta = []
        errors_vega = []
        prices_vega = []
        times_vega = []
        cis_vega = []

        method_map = {
            'mc': (self.mc_greeks_with_batches, False, {'use_antithetic': False, 'rel_bump': bump_size,'abs_bump': bump_size,'scheme':scheme}),
            'mc + antithetic': (self.mc_greeks_with_batches, False, {'use_antithetic': True, 'rel_bump': bump_size,'abs_bump': bump_size,'scheme':scheme}),
            'mc + cv': (self.mc_greeks_with_batches, True, {'use_antithetic': False, 'rel_bump': bump_size,'abs_bump': bump_size,'scheme':scheme}),
            'mc + cv + antithetic': (self.mc_greeks_with_batches, True, {'use_antithetic': True, 'rel_bump': bump_size,'abs_bump': bump_size,'scheme':scheme}),

            # plain QMC
            'qmc': (self.rqmc_greeks_with_scrambles, False, {'rel_bump': bump_size,'abs_bump': bump_size,'scheme':scheme}),
            'qmc + cv': (self.rqmc_greeks_with_scrambles, True, {'rel_bump': bump_size,'abs_bump': bump_size,'scheme':scheme}),
            'qmc + antithetic': (self.rqmc_greeks_with_scrambles, False, {'use_antithetic': True, 'rel_bump': bump_size,'abs_bump': bump_size,'scheme':scheme}),
            'qmc + cv + antithetic': (self.rqmc_greeks_with_scrambles, True, {'use_antithetic': True, 'rel_bump': bump_size,'abs_bump': bump_size,'scheme':scheme}),
            # Support Brownian-bridge variants
            'qmc + bb': (self.rqmc_greeks_with_scrambles, False, {'use_bb': True, 'rel_bump': bump_size,'abs_bump': bump_size,'scheme':scheme}),
            'qmc + bb + antithetic': (self.rqmc_greeks_with_scrambles, False, {'use_bb': True, 'use_antithetic': True, 'rel_bump': bump_size,'abs_bump': bump_size,'scheme':scheme}),
            'qmc + bb + cv': (self.rqmc_greeks_with_scrambles, True, {'use_bb': True, 'rel_bump': bump_size,'abs_bump': bump_size,'scheme':scheme}),
            'qmc + bb + cv + antithetic': (self.rqmc_greeks_with_scrambles, True, {'use_bb': True, 'use_antithetic': True, 'rel_bump': bump_size,'abs_bump': bump_size,'scheme':scheme}),
        }
        if method not in method_map:
            raise ValueError(f"Method must be one of {list(method_map.keys())}")

        entry = method_map[method]
        # unpack entry (support optional extra kwargs)
        if len(entry) == 2:
            func, cv_flag = entry
            extra_kwargs = {}
        else:
            func, cv_flag, extra_kwargs = entry

        for N in N_values:
            t0 = time.perf_counter()
            (mean_delta, se_delta, ci_delta, _, _),(mean_vega, se_vega, ci_vega, _, _) = func(N, batches, euler, CV=cv_flag, **extra_kwargs)
            t1 = time.perf_counter()

            prices_delta.append(mean_delta)
            errors_delta.append(se_delta)
            cis_delta.append(ci_delta)
            times_delta.append(float(t1 - t0))     
            prices_vega.append(mean_vega)
            errors_vega.append(se_vega)
            cis_vega.append(ci_vega)
            times_vega.append(float(t1 - t0))
        
        if Analysis=='Delta':
            return prices_delta, errors_delta, cis_delta, times_delta
        else:
            return prices_vega, errors_vega, cis_vega, times_vega

