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

    def __init__(self, S0, notional, coupon, coupon_barrier, autocall_barrier, protection_barrier, n_obs, T, r, sigma):
        self.S0 = float(S0)
        self.notional = float(notional)
        self.coupon = float(coupon)
        self.coupon_barrier = float(coupon_barrier)
        self.autocall_barrier = float(autocall_barrier)
        self.protection_barrier = float(protection_barrier)
        self.n_obs=n_obs
        #self.n_obs = len(obs_times)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.dt = self.T / self.n_obs if self.n_obs > 0 else self.T



    def _calculate_autocall_price(self, n_paths=100_000):
        discounted_payoffs = []
        for _ in range(n_paths):
            S = self.S0
            cashflow = 0.0
            called = False

            for k in range(1, self.n_obs + 1):
                # ---- Black-Scholes ----
                Z = np.random.normal()
                S *= np.exp((self.r - 0.5 * self.sigma**2) * self.dt + self.sigma * np.sqrt(self.dt) * Z)

                # ---- Autocall ----
                if S >= self.autocall_barrier * self.S0:
                    cashflow += self.notional * self.coupon * np.exp(-self.r * self.dt)
                    cashflow += self.notional * np.exp(-self.r * self.dt)
                    called = True
                    break

                # ---- Coupon ----
                if S >= self.coupon_barrier * self.S0:
                    cashflow += self.notional * self.coupon * np.exp(-self.r * self.dt)

            # ---- Maturité ----
            if not called:
                if S >= self.protection_barrier * self.S0:
                    cashflow += self.notional * np.exp(-self.r * self.T)
                else:
                    cashflow += self.notional * (S / self.S0) * np.exp(-self.r * self.T)

            discounted_payoffs.append(cashflow)
            
        price=np.mean(discounted_payoffs)
        stderr = np.std(discounted_payoffs, ddof=1) / np.sqrt(len(discounted_payoffs))
        return float(price), float(stderr)
       

    def price_monte_carlo(self, n_paths=100_000, seed=None):
        """Price the autocall using Monte Carlo simulation.

        Returns:
        - price (float): estimated present value
        - stderr (float): standard error of the estimate
        """
        if seed is not None:
            np.random.seed(seed)
        return self._calculate_autocall_price(n_paths)


