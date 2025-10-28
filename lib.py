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

    def draw_quasi_random_numbers_bb(self, seed, N, M):
        """Generate quasi-random numbers using Sobol + Brownian bridge.

        This returns an (N, M) array of standard normals corresponding to
        Brownian increments in time order (so they can be used directly as Z
        in the existing simulation routines where dW = sqrt(dt) * Z[:, n]).

        The routine takes Sobol points -> normal variates -> Brownian bridge
        construction (dyadic mid-point order). The mapping is vectorized over
        the N paths.
        """
        # Generate standard normal variates from scrambled Sobol
        m = int(np.ceil(np.log2(N)))
        if 2**m != N:
            raise ValueError("For Sobol QMC, set N to 2**m.")
        sobol_engine = Sobol(d=M, scramble=True, seed=seed)
        U = sobol_engine.random_base2(m=m)
        Z_std = norm.ppf(np.clip(U, 1e-12, 1-1e-12))  # shape (N, M)

        # Brownian-bridge mapping: map independent normals Z_std to samples of
        # W(t1..tM) in dyadic (midpoint) order then convert to increments dW
        Np = N
        dt = self.dt
        sqrt_dt = self.sqrt_dt
        Mloc = M

        # times are at multiples of dt: t_i = i*dt for i=1..M
        # We'll work with indices 0..M (0 = time 0, M = time T). The sampled
        # values array has length M+1, with known W(0)=0 and W(T) from Z_std[:,0].

        # Build dyadic midpoint sampling order (excluding endpoints)
        order = []
        def rec(l, r):
            if r - l <= 1:
                return
            midx = (l + r) // 2
            order.append(midx)
            rec(l, midx)
            rec(midx, r)
        rec(0, Mloc)

        # Prepare sampled values container: shape (N, M+1)
        sampled = np.empty((Np, Mloc+1), dtype=float)
        sampled.fill(np.nan)
        sampled[:, 0] = 0.0

        # First normal (first column) -> W(T)
        sampled[:, Mloc] = Z_std[:, 0] * np.sqrt(self.T)

        # Fill remaining points in the dyadic order using subsequent columns of Z_std
        z_col = 1
        for mpos in order:
            # left and right boundaries are nearest integer indices with non-nan values
            # find l < mpos with sampled[:, l] not nan and r > mpos similarly
            # vectorized search using boolean masks per path would be heavy; instead
            # we use the fact that sampled values are the same pattern across paths
            # (we fill global positions), so we can find l and r scalars.
            # Because we fill in dyadic order, the nearest filled neighbors are the
            # previous split endpoints for all paths.
            # Find left boundary
            l = mpos - 1
            while l >= 0 and np.isnan(sampled[0, l]):
                l -= 1
            r = mpos + 1
            while r <= Mloc and np.isnan(sampled[0, r]):
                r += 1

            # Now l and r are scalar indices with sampled values present
            tl = l * dt
            tr = r * dt
            tm = mpos * dt

            # conditional mean and variance for Brownian bridge
            # mean = ( (r-m)*W(l) + (m-l)*W(r) ) / (r-l)
            # var  = (m-l)*(r-m)/(r-l) * dt
            denom = float(r - l)
            mean = (sampled[:, l] * (r - mpos) + sampled[:, r] * (mpos - l)) / denom
            var = ( (mpos - l) * (r - mpos) / denom ) * dt
            sd = np.sqrt(max(var, 0.0))

            sampled[:, mpos] = mean + Z_std[:, z_col] * sd
            z_col += 1

        # Now convert sampled W values (positions 1..M) into increments dW
        Wvals = sampled[:, 1:]
        Wprev = np.concatenate([np.zeros((Np, 1), dtype=float), Wvals[:, :-1]], axis=1)
        dW = Wvals - Wprev
        # convert to standard-normal increments Z such that dW = sqrt(dt) * Z
        Z_increments = dW / sqrt_dt
        return Z_increments

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
    

    def _payoffs_from_Z_with_params(self, Z, euler=True, S0_override=None, sigma_override=None):
        """
        Return discounted payoffs vector for CEV Asian call using Z and optional
        parameter overrides for S0 and sigma.
        """
        N = Z.shape[0]
        S0 = self.S0 if S0_override is None else float(S0_override)
        sigma = self.sigma if sigma_override is None else float(sigma_override)

        # Simulate paths with local params (copy of sim_CEV_paths using overrides)
        logS = np.empty((N, self.M+1), dtype=float)
        logS[:, 0] = np.log(S0)

        for n in range(self.M):
            logS_n = logS[:, n]
            S_n = np.exp(logS_n)
            dW = self.sqrt_dt * Z[:, n]
            drift = (self.r - 0.5 * sigma**2 * np.power(S_n, 2*self.gamma-2)) * self.dt
            diffusion = sigma * np.power(S_n, self.gamma-1) * dW

            if euler:
                logS[:, n+1] = logS_n + drift + diffusion
            else:
                correction = 0.5 * sigma**2 * (self.gamma - 1) * np.power(S_n, 2*self.gamma-2) * (dW**2 - self.dt)
                logS[:, n+1] = logS_n + drift + diffusion + correction

        S = np.exp(logS)
        arith_avg = S[:, 1:].mean(axis=1)
        return np.exp(-self.r * self.T) * np.maximum(arith_avg - self.K, 0.0)


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


    def estimate_beta_pilot_rqmc(self, Npilot, scrambles=4, euler=True, pilot_seed_offset=100000, use_bb=False):
        """Estimate a fixed beta using independent RQMC scrambles.

        This runs `scrambles` independent RQMC pilots (each with Npilot points) using
        seeds offset by `pilot_seed_offset` and returns the average beta. Use this
        beta as a fixed control-variate coefficient for production RQMC runs.

        Args:
            Npilot (int): number of pilot points per scramble (power of 2)
            scrambles (int): number of independent pilot scrambles to average over
            euler (bool): whether to use Euler scheme for CEV paths
            pilot_seed_offset (int): offset added to self.base_seed to ensure pilots
        Returns:
            float: average beta over pilot scrambles
        """
        betas = []
        for s in range(scrambles):
            seed = self.base_seed + pilot_seed_offset + s
            # choose QMC generator (plain or Brownian-bridge)
            if use_bb:
                Z = self.draw_quasi_random_numbers_bb(seed, Npilot, self.M)
            else:
                Z = self.draw_quasi_random_numbers(seed, Npilot, self.M)
            # Build X, Y on the SAME Z
            S_cev = self.sim_CEV_paths(Npilot, Z, euler=euler)
            X = np.exp(-self.r*self.T)*np.maximum(S_cev[:,1:].mean(1)-self.K, 0.0)
            Y = self.sim_GBM_paths(Npilot, Z)
            Xc, Yc = X - X.mean(), Y - Y.mean()
            denom = (Yc**2).sum()
            beta_s = (Xc*Yc).sum() / denom
            betas.append(beta_s)
        return float(np.mean(betas))


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

    def rqmc_with_scrambles(self, N, scrambles, euler=True, CV=False, beta_fixed=None,
                            pilot_N=16384, pilot_scrambles=4, pilot_seed_offset=100000,
                            use_bb=False):
        """Run RQMC with multiple scrambles.

        If CV=True and beta_fixed is None, this method will first run a separate
        pilot procedure (using `pilot_scrambles` independent scrambles of size
        `pilot_N`) to estimate a fixed beta via `estimate_beta_pilot_rqmc`. The
        pilot scrambles use a seed offset so they are independent from the
        production scrambles. The fixed beta is then applied to every production
        scramble, avoiding in-sample beta estimation that can increase variance
        under RQMC.

        Args:
            N (int): number of points per production scramble (power of 2)
            scrambles (int): number of production scrambles
            euler (bool): whether to use Euler scheme
            CV (bool): whether to apply control variates
            beta_fixed (float or None): if provided, use this beta directly
            pilot_N (int): number of pilot points per pilot scramble
            pilot_scrambles (int): number of pilot scrambles to average beta over
            pilot_seed_offset (int): offset added to base_seed for pilot scrambles
        Returns:
            tuple: (mean, se, ci, var_between, estimates_array)
        """

        # If CV requested but no fixed beta provided, estimate pilot beta once
        if CV and beta_fixed is None:
            beta_fixed = self.estimate_beta_pilot_rqmc(pilot_N, scrambles=pilot_scrambles,
                                                     euler=euler, pilot_seed_offset=pilot_seed_offset)
            print(f"Estimated fixed beta from pilot RQMC: {beta_fixed:.6f}")

        estimates = []
        for s in range(scrambles):
            seed = self.base_seed + s
            if use_bb:
                Z = self.draw_quasi_random_numbers_bb(seed, N, self.M)
            else:
                Z = self.draw_quasi_random_numbers(seed, N, self.M)
            if not CV:
                estimates.append(self.price_asian_option_from_Z(Z, euler))
            else:
                estimates.append(self.price_asian_option_cv_from_Z(Z, euler, beta_fixed=beta_fixed))

        mean, se, ci, var_between = self.ci_from_replicates(estimates)
        return mean, se, ci, var_between

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

        # RQMC + BB
        qmc_bb_mean, qmc_bb_se, qmc_bb_ci, qmc_bb_var = self.rqmc_with_scrambles(N, batches, euler, CV=False, use_bb=True)    

        # RQMC + CV
        qmc_cv_mean, qmc_cv_se, qmc_cv_ci, qmc_cv_var = self.rqmc_with_scrambles(N, batches, euler, CV=True)
        # RQMC + CB + CV
        qmc_cv_bb_mean, qmc_cv_bb_se, qmc_cv_bb_ci, qmc_cv_bb_var = self.rqmc_with_scrambles(N, batches, euler, CV=True, use_bb=True)


        results = {
            'Standard MC': {'mean': mc_mean, 'SE': mc_se, 'CI': mc_ci, 'var': mc_var},
            'Standard MC + Antithetic': {'mean': anti_mean, 'SE': anti_se, 'CI': anti_ci, 'var': anti_var},
            'Standard MC + CV': {'mean': mc_cv_mean, 'SE': mc_cv_se, 'CI': mc_cv_ci, 'var': mc_cv_var},
            'Standard MC + CV + Antithetic': {'mean': mc_cv_anti_mean, 'SE': mc_cv_anti_se, 'CI': mc_cv_anti_ci, 'var': mc_cv_anti_var},
            'RQMC': {'mean': qmc_mean, 'SE': qmc_se, 'CI': qmc_ci, 'var': qmc_var},
            'RQMC + CV': {'mean': qmc_cv_mean, 'SE': qmc_cv_se, 'CI': qmc_cv_ci, 'var': qmc_cv_var},
            'RQMC + BB': {'mean': qmc_bb_mean, 'SE': qmc_bb_se, 'CI': qmc_bb_ci, 'var': qmc_bb_var},
            'RQMC + BB + CV': {'mean': qmc_cv_bb_mean, 'SE': qmc_cv_bb_se, 'CI': qmc_cv_bb_ci, 'var': qmc_cv_bb_var}
        }
    
        return results

    def plot_convergence(self, N_values, method='mc', batches=8, euler=True):
        """Plot convergence of different methods."""
        errors = []
        prices = []
        
        method_map = {
            'mc': (self.mc_with_batches, False),
            'antithetic': (self.mc_with_antithetic, False),
            'mc + cv': (self.mc_with_batches, True),
            'mc + cv + antithetic': (self.mc_with_antithetic, True),
            'qmc': (self.rqmc_with_scrambles, False),
            'qmc + cv': (self.rqmc_with_scrambles, True)
        }

        if method not in method_map:
            raise ValueError(f"Method must be one of {list(method_map.keys())}")

        func, cv_flag = method_map[method]

        for N in N_values:
            mean, se, _, _ = func(N, batches, euler, CV=cv_flag)
            prices.append(mean)
            errors.append(se)

        return prices, errors
    

    def delta_from_Z_fd(self, Z, euler=True, rel_bump=0.001, scheme="central"):
        """
        Finite difference approximation of Delta (dPrice/dS0) using common random numbers.
        Returns (mean, SE, pathwise_estimates).
        """
        h = max(rel_bump * self.S0, 1e-12)
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

    def vega_from_Z_fd(self, Z, euler=True, abs_bump=0.001, scheme="central"):
        """
        Finite difference approximation of Vega (dPrice/dsigma) using common random numbers.
        Returns (mean, SE, pathwise_estimates).
        """
        h = max(abs_bump, 1e-12)
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

    def mc_greeks_with_batches(self, N, batches=8, euler=True, rel_bump=0.001, abs_bump=0.001):
        """
        Compute Greeks using MC with multiple batches, using CRN within each batch.
        Returns tuples for Delta and Vega: (mean, SE, CI, var_between, replicate_means)
        where replicate_means has length == batches.
        """
        delta_repl = []
        vega_repl = []
        for b in range(batches):
            Z = self.draw_pseudo_random_numbers(self.base_seed + b, N, self.M)
            d_mean, _, _  = self.delta_from_Z_fd(Z, euler=euler, rel_bump=rel_bump, scheme="central")
            v_mean,  _, _  = self.vega_from_Z_fd(Z, euler=euler, abs_bump=abs_bump, scheme="central")
            delta_repl.append(d_mean)
            vega_repl.append(v_mean)

        d_mean, d_se, d_ci, d_var_between = self.ci_from_replicates(delta_repl)
        v_mean, v_se, v_ci, v_var_between = self.ci_from_replicates(vega_repl)

        return (
            d_mean, d_se, d_ci, d_var_between, np.asarray(delta_repl, dtype=float)
        ), (
            v_mean, v_se, v_ci, v_var_between, np.asarray(vega_repl, dtype=float)
        )

    def rqmc_greeks_with_scrambles(self, N, scrambles=8, euler=True, rel_bump=0.001, abs_bump=0.001):
        """
        Compute Greeks using RQMC with multiple Owen-scrambled Sobol replications.
        Returns tuples for Delta and Vega: (mean, SE, CI, var_between, replicate_means)
        """
        delta_repl = []
        vega_repl = []
        for s in range(scrambles):
            seed = self.base_seed + s
            # draw using either plain Sobol->normal or Sobol+Brownian-bridge
            if getattr(self, '_rqmc_use_bb', False):
                Z = self.draw_quasi_random_numbers_bb(seed, N, self.M)
            else:
                Z = self.draw_quasi_random_numbers(seed, N, self.M)
            d_mean, _,_  = self.delta_from_Z_fd(Z, euler=euler, rel_bump=rel_bump, scheme="central")
            v_mean, _, _  = self.vega_from_Z_fd(Z, euler=euler, abs_bump=abs_bump, scheme="central")
            delta_repl.append(d_mean)
            vega_repl.append(v_mean)

        d_mean, d_se, d_ci, d_var_between = self.ci_from_replicates(delta_repl)
        v_mean, v_se, v_ci, v_var_between = self.ci_from_replicates(vega_repl)
        return (
            d_mean, d_se, d_ci, d_var_between, np.asarray(delta_repl, dtype=float)
        ), (
            v_mean, v_se, v_ci, v_var_between, np.asarray(vega_repl, dtype=float)
        )
    def compare_greeks_methods(self, N, batches=8, euler=True):
        """Compare different methods for estimating Greeks."""
        # Standard MC
        mc_delta, mc_vega = self.mc_greeks_with_batches(N, batches, euler)

        # RQMC
        rqmc_delta, rqmc_vega = self.rqmc_greeks_with_scrambles(N, batches, euler)

        results = {
            'Standard MC': {'Delta': mc_delta, 'Vega': mc_vega},
            'RQMC': {'Delta': rqmc_delta, 'Vega': rqmc_vega}
        }

        return results