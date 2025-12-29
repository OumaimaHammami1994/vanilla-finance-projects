import numpy as np

class Autocall:
    """Autocallable pricer using Monte Carlo on GBM.

    Payoff rules implemented:
    - At each observation time t_i:
        * If S_t >= C*S0 -> auto-call: pay `notional * (1 + coupon)` at t_i and stop.
        * Else if S_t >= B*S0 -> pay coupon `notional * coupon` at t_i and continue.
        * Else -> no coupon and continue.
    - If never auto-called, at maturity T:
        * If S_T >= A*S0 -> return `notional` (capital returned) at T.
        * Else -> return `notional * (S_T / S0)` (downside exposure) at T.

    All cashflows are discounted to present value using the constant risk-free rate `r`.
    """

    def __init__(self, S0, notional, coupon, Coupon_barrier, Autocall_barrier, Protection_barrier, obs_times, T, r, sigma):
        self.S0 = float(S0)
        self.notional = float(notional)
        self.coupon = float(coupon)
        # A: protection threshold (fraction of S0), B: coupon threshold, C: autocall threshold
        self.Coupon_barrier = float(Coupon_barrier)
        self.Autocall_barrier = float(Autocall_barrier)
        self.Protection_barrier = float(Protection_barrier)
        self.obs_times = sorted(float(t) for t in obs_times)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)

    def _simulate_gbm_at_times(self, n_paths, times, seed=None):
        """Simulate GBM values at the specified times (array-like, increasing),
        returns array shape (n_paths, len(times)) including time 0 as first column.
        """
        times = np.asarray(times, dtype=float)
        if times[0] != 0.0:
            times = np.concatenate(([0.0], times))
        rng = np.random.default_rng(seed)
        n_intervals = len(times) - 1
        dt = np.diff(times)
        # Normal increments: shape (n_paths, n_intervals)
        Z = rng.standard_normal(size=(n_paths, n_intervals))
        S = np.empty((n_paths, len(times)), dtype=float)
        S[:, 0] = self.S0
        for i in range(n_intervals):
            drift = (self.r - 0.5 * self.sigma ** 2) * dt[i]
            vol = self.sigma * np.sqrt(dt[i])
            S[:, i + 1] = S[:, i] * np.exp(drift + vol * Z[:, i])
        return S, times

    def price_monte_carlo(self, n_paths=100_000, seed=None):
        """Price the autocall using Monte Carlo simulation.

        Returns:
        - price (float): estimated present value
        - stderr (float): standard error of the estimate
        """
        # Simulate at observation times and maturity
        sim_times = list(self.obs_times)
        if self.T not in sim_times:
            sim_times = sim_times + [self.T]
        S_paths, times = self._simulate_gbm_at_times(n_paths, sim_times, seed=seed)
        # Observation columns correspond to indices 1..len(obs_times)
        n_obs = len(self.obs_times)
        obs_idx = np.arange(1, 1 + n_obs) if n_obs > 0 else np.array([], dtype=int)
        A_level = self.Coupon_barrier * self.S0  # protection
        B_level = self.B * self.S0  # coupon
        C_level = self.C * self.S0  # autocall

        payoffs = np.zeros(n_paths, dtype=float)

        # Track which paths have been autocalled
        called = np.zeros(n_paths, dtype=bool)

        # Process observation dates
        if n_obs > 0:
            obs_times_arr = np.array(times[1:1 + n_obs])
            for j, t in enumerate(obs_times_arr):
                Sj = S_paths[:, 1 + j]
                # Autocall condition
                is_called = (~called) & (Sj >= C_level)
                if is_called.any():
                    payoffs[is_called] = self.notional * (1.0 + self.coupon) * np.exp(-self.r * t)
                    called[is_called] = True
                # Coupon-only condition (not autocalled at this obs)
                pays_coupon = (~called) & (Sj >= B_level) & (Sj < C_level)
                if pays_coupon.any():
                    payoffs[pays_coupon] += self.notional * self.coupon * np.exp(-self.r * t)

        # For paths never autocalled, handle maturity payout
        not_called_idx = ~called
        if not_called_idx.any():
            ST = S_paths[not_called_idx, -1]
            # If ST >= A_level -> capital returned
            above_A = ST >= A_level
            if above_A.any():
                payoffs[not_called_idx][above_A] += self.notional * np.exp(-self.r * self.T)
            if (~above_A).any():
                payoffs[not_called_idx][~above_A] += self.notional * (ST[~above_A] / self.S0) * np.exp(-self.r * self.T)

        price = payoffs.mean()
        stderr = payoffs.std(ddof=1) / np.sqrt(len(payoffs))
        return float(price), float(stderr)


if __name__ == '__main__':
    # Quick local demo
    ac = Autocall(S0=100, notional=1000, coupon=0.05, A=0.7, B=0.9, C=1.0, obs_times=[0.25,0.5,0.75], T=1.0, r=0.01, sigma=0.2)
    price, se = ac.price_monte_carlo(n_paths=20000, seed=42)
    print(f"Autocall price: {price:.2f} ± {1.96*se:.2f} (95% CI)")
    if show_paths > 0:
        # plot a small sample of simulated paths
        sim_n = min(int(show_paths), 1000)
        sim_times = obs_times if obs_times[-1] != T else obs_times
        S_paths, times = ac._simulate_gbm_at_times(sim_n, sim_times, seed=seed_arg)
        fig, ax = plt.subplots(figsize=(8, 4))
        for i in range(min(sim_n, len(S_paths))):
            ax.plot(times, S_paths[i, :], color='C0', alpha=0.6)
        ax.set_title('Sample simulated price paths')
        ax.set_xlabel('Time (years)')
        ax.set_ylabel('Price')
        st.pyplot(fig)