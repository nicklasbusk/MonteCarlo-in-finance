# This is a Python library for Monte Carlo simulations in pricing an Asian option.
import numpy as np
import matplotlib.pyplot as plt
#import qmcpy as qp
from scipy.stats.qmc import Sobol
from scipy.stats import norm, t
import time


class MonteCarloSimulation:
    def __init__(self, S0, K, r, sigma, gamma, T, M, base_seed=42):
        """Initialize Monte Carlo simulation parameters.

        Args:
            S0 (float): Initial stock price
            K (float): Strike price
            r (float): Risk-free rate/ Mu
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
        # Clip to avoid extreme quantiles that produce inf/-inf in norm.ppf
        epsilon = 1e-10
        U = np.clip(U, epsilon, 1.0 - epsilon)
        return norm.ppf(U)

    @staticmethod
    def dyadic_midpoint_order(M):
        """Return dyadic midpoint sampling order (exclude endpoints 0 and M).

        This reproduces the order produced by the recursive midpoint splitting
        used for Brownian-bridge constructions: visit a segment, record its
        midpoint, then recurse left then right.
        """
        order = []
        stack = [(0, M)]
        while stack:
            l, r = stack.pop()
            if r - l <= 1:
                continue
            mid = (l + r) // 2
            order.append(mid)
            # push right then left so left is processed next (pre-order DFS)
            stack.append((mid, r))
            stack.append((l, mid))
        return order

    def draw_quasi_random_numbers_bb(self, seed, N, M, use_antithetic=False):
        """Generate quasi-random numbers using Sobol + Brownian bridge.

        This returns an (N, M) array of standard normals corresponding to
        Brownian increments in time order (so they can be used directly as Z
        in the existing simulation routines where dW = sqrt(dt) * Z[:, n]).

        The routine takes Sobol points -> normal variates -> Brownian bridge
        construction (dyadic mid-point order). The mapping is vectorized over
        the N paths.
        """
        # Clip epsilon to avoid extreme quantiles
        epsilon = 1e-10
        
        if use_antithetic:
            if N < 2:
                raise ValueError("For antithetic pairing, N must be at least 2.")

            # Generate half the Sobol points and stack their antithetic complements.
            m = int(np.ceil(np.log2(N)))
            half = N // 2
            # Use m-1 to generate half = 2**(m-1) Sobol points
            sobol_engine = Sobol(d=M, scramble=True, seed=seed)
            U_half = sobol_engine.random_base2(m=m-1)
            U = np.vstack([U_half, 1.0 - U_half])
            U = np.clip(U, epsilon, 1.0 - epsilon)
            Z_std = norm.ppf(U)
        else:
            # Generate standard normal variates from scrambled Sobol; require N=2**m
            m = int(np.ceil(np.log2(N)))
            if 2**m != N:
                raise ValueError("For Sobol QMC, set N to 2**m.")
            sobol_engine = Sobol(d=M, scramble=True, seed=seed)
            U = sobol_engine.random_base2(m=m)
            U = np.clip(U, epsilon, 1.0 - epsilon)
            Z_std = norm.ppf(U)  # shape (N, M)

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
        order = self.dyadic_midpoint_order(Mloc)

        # Prepare sampled values container: shape (N, M+1)
        sampled = np.empty((Np, Mloc+1), dtype=float)
        sampled.fill(np.nan)
        sampled[:, 0] = 0.0

        # First normal (first column) -> W(T)
        sampled[:, Mloc] = Z_std[:, 0] * np.sqrt(self.T)

        # Fill remaining points in the dyadic order using subsequent columns of Z_std
        z_col = 1
        for mpos in order:
            # Find nearest filled neighbors l < mpos < r (same for all paths
            # because we fill positions deterministically in dyadic order).
            l = mpos - 1
            while l >= 0 and np.isnan(sampled[0, l]):
                l -= 1
            r = mpos + 1
            while r <= Mloc and np.isnan(sampled[0, r]):
                r += 1

            # conditional mean and variance for Brownian bridge at index mpos
            denom = float(r - l)
            mean = (sampled[:, l] * (r - mpos) + sampled[:, r] * (mpos - l)) / denom
            var = ((mpos - l) * (r - mpos) / denom) * dt
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
    
    def draw_quasi_antithetic_numbers(self, seed, N, M):
        """Generate quasi-random numbers using Sobol sequence with antithetic pairs.

        Produces N samples where the first N/2 are Sobol points and the
        remaining N/2 are their antithetic complements (1 - U). Requires N
        to be a power of two and N >= 2.
        """
        if N < 2:
            raise ValueError("For antithetic pairing, N must be at least 2.")

        m = int(np.ceil(np.log2(N)))
        if 2**m != N:
            raise ValueError("For Sobol QMC, set N to 2**m.")

        half = N // 2
        # Use m-1 to generate half = 2**(m-1) Sobol points
        sobol_engine = Sobol(d=M, scramble=True, seed=seed)
        U_half = sobol_engine.random_base2(m=m-1)

        # Stack original and antithetic (1 - U) to form N samples
        U = np.vstack([U_half, 1.0 - U_half])
        
        # Clip to avoid extreme quantiles that produce inf/-inf in norm.ppf
        epsilon = 1e-10
        U = np.clip(U, epsilon, 1.0 - epsilon)

        # Convert to standard normals
        return norm.ppf(U)

    def _cev_arith_from_Z(self, Z, euler=False):
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

    def geometric_asian_closed_form_sigma(self, sigma_val):
        """Closed-form geometric Asian price for a given sigma (helper)."""
        m = np.log(self.S0) + (self.r - 0.5 * sigma_val**2) * self.T * (self.M + 1) / (2 * self.M)
        v = (sigma_val**2) * self.T * ((self.M + 1) * (2 * self.M + 1)) / (6 * self.M**2)
        s = np.sqrt(max(v, 0.0))
        if s == 0.0:
            G = np.exp(m)
            return float(np.exp(-self.r*self.T) * max(G - self.K, 0.0))
        d1 = (m - np.log(self.K) + v) / s
        d2 = d1 - s
        undiscounted = np.exp(m + 0.5*v) * norm.cdf(d1) - self.K * norm.cdf(d2)
        return float(np.exp(-self.r*self.T) * undiscounted)

    def geometric_asian_delta_closed_form(self):
        """Closed-form expected geometric-Asian Delta (dPrice/dS0) under GBM.

        Derived by differentiating the closed-form geometric Asian price w.r.t. S0.
        """
        # reuse same symbols as geometric_asian_closed_form
        m_const = (self.r - 0.5 * self.sigma**2) * self.T * (self.M + 1) / (2 * self.M)
        m = np.log(self.S0) + m_const
        v = (self.sigma**2) * self.T * ((self.M + 1) * (2 * self.M + 1)) / (6 * self.M**2)
        s = np.sqrt(max(v, 0.0))
        if s == 0.0:
            G = np.exp(m)
            return float(np.exp(-self.r*self.T) * (G > self.K) * (G / self.S0))

        A = np.exp(m + 0.5 * v)
        d1 = (m - np.log(self.K) + v) / s
        d2 = d1 - s

        # derivatives
        dm_dS0 = 1.0 / self.S0
        dA_dS0 = A * dm_dS0
        dd1_dS0 = dm_dS0 / s

        term1 = dA_dS0 * norm.cdf(d1)
        term2 = A * norm.pdf(d1) * dd1_dS0
        term3 = - self.K * norm.pdf(d2) * dd1_dS0

        undiscounted_deriv = term1 + term2 + term3
        return float(np.exp(-self.r * self.T) * undiscounted_deriv)

    def geometric_asian_vega_closed_form(self, rel_eps=1e-6):
        """Estimate closed-form geometric-Asian Vega by central finite-difference
        of the closed-form price with respect to sigma. This is fast and
        numerically stable for the small eps used here.
        """
        eps = max(rel_eps * max(1.0, abs(self.sigma)), 1e-8)
        p_plus = self.geometric_asian_closed_form_sigma(self.sigma + eps)
        p_minus = self.geometric_asian_closed_form_sigma(self.sigma - eps)
        return float((p_plus - p_minus) / (2.0 * eps))

    def sim_CEV_paths(self, N, Z, euler=False):
        """Simulate CEV model paths."""
        # Recompute dt/sqrt_dt in case self.M was changed externally
        dt = self.T / self.M
        sqrt_dt = np.sqrt(dt)

        logS = np.empty((N, self.M+1), dtype=float)
        logS[:, 0] = np.log(self.S0)

        # Small floor to avoid taking powers of zero which can produce inf/nan
        S_floor = 1e-16

        for n in range(self.M):
            logS_n = logS[:, n]
            S_n = np.exp(logS_n)
            # apply floor to S_n for numerical stability in power operations
            S_n_safe = np.maximum(S_n, S_floor)
            dW = sqrt_dt * Z[:, n]

            # compute powers in log-space
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
        """Price Asian option using provided random numbers."""
        N = Z.shape[0]
        S = self.sim_CEV_paths(N, Z, euler=euler)
        avg = S[:, 1:].mean(axis=1)
        disc_payoff = np.exp(-self.r * self.T) * np.maximum(avg - self.K, 0.0)
        return float(disc_payoff.mean())

    def price_asian_option_from_Z_raw(self, Z, euler=False): #Used in euler/milstein comparison
        """Price Asian option using provided random numbers."""
        N = Z.shape[0]
        S = self.sim_CEV_paths(N, Z, euler=euler)
        avg = S[:, 1:].mean(axis=1)
        disc_payoff = np.exp(-self.r * self.T) * np.maximum(avg - self.K, 0.0)
        return disc_payoff

    def _payoffs_from_Z_with_params(self, Z, euler=False, S0_override=None, sigma_override=None):
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

        # Recompute dt/sqrt_dt in case self.M was changed externally
        dt = self.T / self.M
        sqrt_dt = np.sqrt(dt)

        # Small floor to avoid taking powers of zero
        S_floor = 1e-16

        for n in range(self.M):
            logS_n = logS[:, n]
            S_n = np.exp(logS_n)
            S_n_safe = np.maximum(S_n, S_floor)
            dW = sqrt_dt * Z[:, n]

            # compute powers in log-space
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


    def estimate_beta_pilot_rqmc(self, Npilot, scrambles=4, euler=False, pilot_seed_offset=100000, use_bb=True):
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
            # Guard against zero or non-finite denominator which would produce inf/nan
            if denom == 0 or not np.isfinite(denom):
                beta_s = 0.0
            else:
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
            ci = ((mean - tcrit * se_mean).round(4), (mean + tcrit * se_mean).round(4))
        else:
            ci = (np.nan, np.nan)
        return mean, se_mean, ci, var_between

    def rqmc_with_scrambles(self, N, scrambles, euler=False, CV=False, beta_fixed=None,
                            pilot_N=16384, pilot_scrambles=4, pilot_seed_offset=100000,
                            use_bb=False, use_antithetic=False):
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
    
    def rqmc_with_antithetic(self, N, batches, euler=False, CV=False):
        """Run MC with antithetic variates."""
        estimates = []
        for b in range(batches):
            Z = self.draw_quasi_antithetic_numbers(self.base_seed + b, N, self.M)
            if not CV:    
                estimates.append(self.price_asian_option_from_Z(Z, euler))
            else:
                estimates.append(self.price_asian_option_cv_from_Z(Z, euler, beta_fixed=None))
        return self.ci_from_replicates(estimates)

    def mc_with_antithetic_cv_pairavg(self, N, batches, euler=False): #Dont get used?
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

    def compare_methods(self, N, batches=8, euler=False):
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
        """Plot convergence of different methods."""
        errors = []
        prices = []
        times = []
        cis = []
        
        # method_map values are tuples of the form:
        #   (callable_func, cv_flag) or
        #   (callable_func, cv_flag, extra_kwargs_dict)
        # extra_kwargs_dict are forwarded to the callable (useful to enable BB)
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
        # unpack entry (support optional extra kwargs)
        if len(entry) == 2:
            func, cv_flag = entry
            extra_kwargs = {}
        else:
            func, cv_flag, extra_kwargs = entry

        for N in N_values:
            t0 = time.perf_counter()
            # forward CV flag and any extra kwargs (e.g. use_bb=True)
            mean, se, ci, _ = func(N, batches, euler, CV=cv_flag, **extra_kwargs)
            t1 = time.perf_counter()
            prices.append(mean)
            errors.append(se)
            cis.append(ci)
            times.append(float(t1 - t0))

        return prices, errors, cis, times
    

    def delta_from_Z_fd(self, Z, euler=False, rel_bump=0.001, scheme="central", beta=None):
        """
        Finite difference approximation of Delta (dPrice/dS0) using common random numbers.
        Returns (mean, SE, pathwise_estimates).
        """
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
        """
        Finite difference approximation of Vega (dPrice/dsigma) using common random numbers
        and control variates.
        Returns (mean, SE, pathwise_estimates).
        """
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

        # Control variate: geometric Asian Vega under GBM
        N = Z.shape[0]
        # Compute geometric Asian Vega pathwise estimates
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
            # sample-optimal beta: Cov(g, geo_vega_paths)/Var(geo_vega_paths)
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
        """
        Finite difference approximation of Delta (dPrice/dS0) using common random numbers
        and control variates.
        Returns (mean, SE, pathwise_estimates).
        """
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

        # Control variate: geometric Asian Delta under GBM
        N = Z.shape[0]
        # Compute geometric Asian Delta pathwise estimates
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
            # sample-optimal beta: Cov(g, geo_delta_paths)/Var(geo_delta_paths)
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
        """
        Finite difference approximation of Vega (dPrice/dsigma) using common random numbers.
        Returns (mean, SE, pathwise_estimates).
        """
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
        """
        Compute Greeks using MC with multiple batches, using CRN within each batch.
        Returns tuples for Delta and Vega: (mean, SE, CI, var_between, replicate_means)
        where replicate_means has length == batches.
        """
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
        """
        Compute Greeks using RQMC with multiple Owen-scrambled Sobol replications.
        Returns tuples for Delta and Vega: (mean, SE, CI, var_between, replicate_means)
        """
        delta_repl = []
        vega_repl = []
        # RQMC uses the same in-sample CV approach as MC when CV=True: pass
        # beta=None to the per-scramble estimators so they compute sample-optimal
        # beta on that scramble. This mirrors the behavior of MC Greek routines.
        for s in range(scrambles):
            seed = self.base_seed + s
            # draw using either plain Sobol->normal or Sobol+Brownian-bridge
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
                # per-scramble in-sample beta estimation (same as MC)
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
        """Compare different methods for estimating Greeks.
        N_values:lst
        """

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


def vrf_greek_vs_param(
    base_params,
    param_name: str,
    param_values: list,
    N: int,
    n_batches: int,
    greek: str = "delta",
):
    """
    Compute variance reduction factor (VRF) for a Greek as a function of a single model parameter.

    Args:
        base_params: instance of MonteCarloSimulation (or dict-like with keys S0,K,r,sigma,gamma,T,M,base_seed)
        param_name: one of 'sigma', 'gamma', 'T'
        param_values: iterable of values to test
        N: number of paths per batch / per scramble (for QMC must be power of 2 when used with Sobol)
        n_batches: number of batches (MC) or scrambles (RQMC)
        greek: 'delta' or 'vega'

    Returns:
        dict containing keys: 'param_values','var_cmc','var_rqmc','vrf','se_cmc','se_rqmc','greek','param_name'
    """
    allowed = {"sigma", "gamma", "T"}
    if param_name not in allowed:
        raise ValueError(f"param_name must be one of {allowed}")

    # helper to construct a MonteCarloSimulation from base params
    def make_sim_with_param(v):
        if isinstance(base_params, MonteCarloSimulation):
            b = base_params
            S0 = b.S0
            K = b.K
            r = b.r
            sigma = b.sigma
            gamma = b.gamma
            T = b.T
            M = b.M
            base_seed = getattr(b, "base_seed", 42)
        else:
            # assume dict-like
            S0 = base_params.get("S0")
            K = base_params.get("K")
            r = base_params.get("r")
            sigma = base_params.get("sigma")
            gamma = base_params.get("gamma")
            T = base_params.get("T")
            M = base_params.get("M")
            base_seed = base_params.get("base_seed", 42)

        # replace the requested parameter
        if param_name == "sigma":
            sigma = float(v)
        elif param_name == "gamma":
            gamma = float(v)
        elif param_name == "T":
            T = float(v)

        sim = MonteCarloSimulation(S0, K, r, sigma, gamma, T, M, base_seed=base_seed)
        return sim

    param_vals = list(param_values)
    var_cmc_list = []
    var_rqmc_list = []
    vrf_list = []
    se_cmc_list = []
    se_rqmc_list = []

    kind = greek.strip().lower()
    if kind not in ("delta", "vega"):
        raise ValueError("greek must be 'delta' or 'vega'")

    for v in param_vals:
        sim = make_sim_with_param(v)

        # Run MC batches (standard MC) to obtain between-batch variance and SE
        (d_mean_mc, d_se_mc, d_ci_mc, d_var_between_mc, d_repl_mc), (v_mean_mc, v_se_mc, v_ci_mc, v_var_between_mc, v_repl_mc) = sim.mc_greeks_with_batches(N, batches=n_batches, use_antithetic=True,CV=True)
        
        # Run RQMC scrambles (use Brownian bridge by default for good performance)
        (d_mean_q, d_se_q, d_ci_q, d_var_between_q, d_repl_q), (v_mean_q, v_se_q, v_ci_q, v_var_between_q, v_repl_q) = sim.rqmc_greeks_with_scrambles(N, scrambles=n_batches, use_bb=True, CV=False)

        if kind == "delta":
            var_cmc = float(d_var_between_mc)
            var_rqmc = float(d_var_between_q)
            se_cmc = float(d_se_mc)
            se_rqmc = float(d_se_q)
        else:
            var_cmc = float(v_var_between_mc)
            var_rqmc = float(v_var_between_q)
            se_cmc = float(v_se_mc)
            se_rqmc = float(v_se_q)

        vrf = float(var_cmc / var_rqmc) if (var_rqmc is not None and var_rqmc != 0 and np.isfinite(var_rqmc)) else float('nan')

        var_cmc_list.append(var_cmc)
        var_rqmc_list.append(var_rqmc)
        vrf_list.append(vrf)
        se_cmc_list.append(se_cmc)
        se_rqmc_list.append(se_rqmc)

    return {
        "param_values": param_vals,
        "var_cmc": var_cmc_list,
        "var_rqmc": var_rqmc_list,
        "vrf": vrf_list,
        "se_cmc": se_cmc_list,
        "se_rqmc": se_rqmc_list,
        "greek": kind,
        "param_name": param_name,
    }


def plot_vrf_vs_param(results: dict, title_prefix: str = "") -> None:
    """
    Plot VRF = var_cmc / var_rqmc versus parameter values.

    Creates two vertical subplots:
      - Top: VRF vs parameter
      - Bottom: var_cmc and var_rqmc vs parameter (log-scale y)

    Args:
        results: dict returned by `vrf_greek_vs_param`.
        title_prefix: optional prefix for the plot title.
    """
    # defensive extraction
    x = np.asarray(results.get("param_values", []), dtype=float)
    vrf = np.asarray(results.get("vrf", []), dtype=float)
    var_cmc = np.asarray(results.get("var_cmc", []), dtype=float)
    var_rqmc = np.asarray(results.get("var_rqmc", []), dtype=float)

    param_name = results.get("param_name", "param")
    greek = results.get("greek", "greek").capitalize()

    # friendly x-label
    x_label = param_name

    plt.figure(figsize=(8, 8))

    # Top: VRF
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(x, vrf, marker="o", linestyle="-", color="#1f77b4")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("VRF (var_cmc / var_rqmc)")
    ax1.set_title(f"{title_prefix} {greek} - VRF vs {param_name}".strip())
    ax1.grid(True)

    # Bottom: variances
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(x, var_cmc, marker="s", linestyle="-", label="var_cmc", color="#ff7f0e")
    ax2.plot(x, var_rqmc, marker="d", linestyle="--", label="var_rqmc", color="#2ca02c")
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Between-replicate variance")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, which="both", ls="--", lw=0.5)

    plt.tight_layout()
    plt.show()


def vrf_greek_vs_param_all(
    base_params,
    param_name: str,
    param_values: list,
    N: int,
    n_batches: int,
    greek: str = "delta",
    methods: list = None,
    rel_bump: float = 0.001,
    abs_bump: float = 0.001,
    scheme: str = "central",
):
    """
    Compute VRF (variance reduction factor) for multiple Monte Carlo/RQMC setups
    relative to standard CMC across a parameter sweep.

    Args:
        base_params: MonteCarloSimulation instance or dict-like params
        param_name: one of 'sigma','gamma','T'
        param_values: iterable of parameter values to test
        N: number of points per batch/scramble
        n_batches: number of batches (MC) or scrambles (RQMC)
        greek: 'delta' or 'vega'
        methods: list of method names to evaluate. If None, a sensible default is used.
        rel_bump/abs_bump/scheme: forwarded to Greek estimators

    Returns:
        dict with keys: 'param_values','param_name','greek','methods','vars','ses','vrf_vs_cmc'
    """
    default_methods = [
        'mc',
        'mc + antithetic',
        'mc + cv',
        'mc + cv + antithetic',
        'qmc + bb',
        'qmc + bb + cv'
    ]
    if methods is None:
        methods = default_methods

    kind = greek.strip().lower()
    if kind not in ("delta", "vega"):
        raise ValueError("greek must be 'delta' or 'vega'")

    # factory to make MonteCarloSimulation for each param value
    def make_sim(v):
        if isinstance(base_params, MonteCarloSimulation):
            b = base_params
            S0, K, r, sigma, gamma, T, M, base_seed = (
                b.S0, b.K, b.r, b.sigma, b.gamma, b.T, b.M, getattr(b, 'base_seed', 42)
            )
        else:
            S0 = base_params.get('S0')
            K = base_params.get('K')
            r = base_params.get('r')
            sigma = base_params.get('sigma')
            gamma = base_params.get('gamma')
            T = base_params.get('T')
            M = base_params.get('M')
            base_seed = base_params.get('base_seed', 42)

        if param_name == 'sigma':
            sigma = float(v)
        elif param_name == 'gamma':
            gamma = float(v)
        elif param_name == 'T':
            T = float(v)
        else:
            raise ValueError("param_name must be one of 'sigma','gamma','T'")

        return MonteCarloSimulation(S0, K, r, sigma, gamma, T, M, base_seed=base_seed)

    param_vals = list(param_values)
    methods_list = list(methods)

    # prepare storage
    vars_dict = {m: [] for m in methods_list}
    ses_dict = {m: [] for m in methods_list}
    vrf_dict = {m: [] for m in methods_list}

    for v in param_vals:
        sim = make_sim(v)

        # compute baseline MC variance for this param value
        # use mc_greeks_with_batches with CV/antithetic flags as appropriate for 'mc'
        # but baseline is plain 'mc' (no CV, no antithetic)
        (d_mean_mc, d_se_mc, d_ci_mc, d_var_between_mc, d_repl_mc), (v_mean_mc, v_se_mc, v_ci_mc, v_var_between_mc, v_repl_mc) = sim.mc_greeks_with_batches(N, batches=n_batches, use_antithetic=False, CV=False, rel_bump=rel_bump, abs_bump=abs_bump, scheme=scheme)

        if kind == 'delta':
            var_cmc = float(d_var_between_mc)
            se_cmc = float(d_se_mc)
        else:
            var_cmc = float(v_var_between_mc)
            se_cmc = float(v_se_mc)

        # store baseline in dict under 'mc' if present
        if 'mc' in vars_dict:
            vars_dict['mc'].append(var_cmc)
            ses_dict['mc'].append(se_cmc)
            vrf_dict['mc'].append(1.0)

        # now compute for each method
        for m in methods_list:
            # skip baseline, already recorded
            if m == 'mc':
                continue

            # Map method string to call
            m_low = m.lower()
            try:
                if m_low.startswith('mc'):
                    use_ant = 'antithetic' in m_low
                    use_cv = 'cv' in m_low
                    (d_mean, d_se, d_ci, d_var_between, d_repl), (v_mean, v_se, v_ci, v_var_between, v_repl) = sim.mc_greeks_with_batches(N, batches=n_batches, use_antithetic=use_ant, CV=use_cv, rel_bump=rel_bump, abs_bump=abs_bump, scheme=scheme)
                else:
                    # treat as rqmc variant
                    use_ant = 'antithetic' in m_low
                    use_cv = 'cv' in m_low
                    use_bb = 'bb' in m_low
                    (d_mean, d_se, d_ci, d_var_between, d_repl), (v_mean, v_se, v_ci, v_var_between, v_repl) = sim.rqmc_greeks_with_scrambles(N, scrambles=n_batches, use_antithetic=use_ant, CV=use_cv, use_bb=use_bb, rel_bump=rel_bump, abs_bump=abs_bump, scheme=scheme)

                if kind == 'delta':
                    var_m = float(d_var_between)
                    se_m = float(d_se)
                else:
                    var_m = float(v_var_between)
                    se_m = float(v_se)

            except Exception as exc:
                # on error, append nan and continue
                var_m = float('nan')
                se_m = float('nan')

            vars_dict[m].append(var_m)
            ses_dict[m].append(se_m)

            # VRF relative to CMC: var_cmc / var_m
            if var_m == 0 or (not np.isfinite(var_m)):
                vrf_val = float('nan')
            else:
                vrf_val = float(var_cmc / var_m)
            vrf_dict[m].append(vrf_val)

    return {
        'param_values': param_vals,
        'param_name': param_name,
        'greek': kind,
        'methods': methods_list,
        'vars': vars_dict,
        'ses': ses_dict,
        'vrf_vs_cmc': vrf_dict,
    }


def plot_vrf_vs_param_methods(results: dict, title_prefix: str = "") -> None:
    """
    Plot VRF vs parameter for multiple methods (lines per method).

    Args:
        results: output from `vrf_greek_vs_param_all`
        title_prefix: optional title prefix
    """
    x = np.asarray(results.get('param_values', []), dtype=float)
    methods = results.get('methods', [])
    vrf = results.get('vrf_vs_cmc', {})
    param_name = results.get('param_name', 'param')
    greek = results.get('greek', '').capitalize()

    plt.figure(figsize=(8, 5))
    for m in methods:
        y = np.asarray(vrf.get(m, []), dtype=float)
        plt.plot(x, y, marker='o', linestyle='-', label=m)

    plt.xlabel(param_name)
    plt.ylabel('VRF (var_cmc / var_method)')
    plt.title(f"{title_prefix} {greek} - VRF vs {param_name}".strip())
    plt.grid(True)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()