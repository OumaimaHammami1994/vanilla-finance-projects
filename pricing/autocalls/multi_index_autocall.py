import numpy as np

class MultiIndexAutocall:
    """Multi-index autocall pricer using Monte Carlo on GBM for multiple stocks.

    The basket is defined as the arithmetic average of the stock prices.
    Payoff rules implemented:
    - At each observation time t_i:
        * If basket_t >= autocall_barrier * basket_0 -> auto-call: pay `notional * (1 + coupon)` at t_i and stop.
        * Else if basket_t >= coupon_barrier * basket_0 -> pay coupon `notional * coupon` at t_i and continue.
        * Else -> no coupon and continue.
    - If never auto-called, at maturity T:
        * If basket_T >= protection_barrier * basket_0 -> return `notional` (capital returned) at T.
        * Else -> return `notional * (basket_T / basket_0)` (downside exposure) at T.

    All cashflows are discounted to present value using the constant risk-free rate `r`.
    """

    def __init__(self, S0_list, notional, coupon, coupon_barrier, autocall_barrier, protection_barrier, n_obs, T, r, sigma_list, basket_type='mean', rho=0.0):
        self.S0_list = [float(s) for s in S0_list]
        self.n_assets = len(self.S0_list)
        if basket_type == 'mean':
            self.basket_0 = np.mean(self.S0_list)
        elif basket_type == 'min':
            self.basket_0 = np.min(self.S0_list)
        elif basket_type == 'max':
            self.basket_0 = np.max(self.S0_list)
        else:
            raise ValueError("basket_type must be 'mean', 'min', or 'max'")
        self.basket_type = basket_type
        self.notional = float(notional)
        self.coupon = float(coupon)
        self.coupon_barrier = float(coupon_barrier)
        self.autocall_barrier = float(autocall_barrier)
        self.protection_barrier = float(protection_barrier)
        self.n_obs = n_obs
        self.T = float(T)
        self.r = float(r)
        self.sigma_list = [float(s) for s in sigma_list]
        self.dt = self.T / self.n_obs if self.n_obs > 0 else self.T
        # Correlation matrix
        self.corr_matrix = np.full((self.n_assets, self.n_assets), rho) + np.eye(self.n_assets) * (1 - rho)
        self.L = np.linalg.cholesky(self.corr_matrix)

    def _calculate_autocall_price(self, n_paths=100_000):
        discounted_payoffs = []
        for _ in range(n_paths):
            S_list = self.S0_list.copy()
            cashflow = 0.0
            called = False

            for k in range(1, self.n_obs + 1):
                # Simulate each stock with correlation
                Z = np.random.normal(size=self.n_assets)
                Z_corr = self.L @ Z
                for i in range(self.n_assets):
                    S_list[i] *= np.exp((self.r - 0.5 * self.sigma_list[i]**2) * self.dt + self.sigma_list[i] * np.sqrt(self.dt) * Z_corr[i])

                # Compute basket
                if self.basket_type == 'mean':
                    basket = np.mean(S_list)
                elif self.basket_type == 'min':
                    basket = np.min(S_list)
                elif self.basket_type == 'max':
                    basket = np.max(S_list)

                # ---- Autocall ----
                if basket >= self.autocall_barrier * self.basket_0:
                    cashflow += self.notional * self.coupon * np.exp(-self.r * k * self.dt)
                    cashflow += self.notional * np.exp(-self.r * k * self.dt)
                    called = True
                    break

                # ---- Coupon ----
                if basket >= self.coupon_barrier * self.basket_0:
                    cashflow += self.notional * self.coupon * np.exp(-self.r * k * self.dt)

            # ---- Maturity ----
            if not called:
                if self.basket_type == 'mean':
                    basket_T = np.mean(S_list)
                elif self.basket_type == 'min':
                    basket_T = np.min(S_list)
                elif self.basket_type == 'max':
                    basket_T = np.max(S_list)
                if basket_T >= self.protection_barrier * self.basket_0:
                    cashflow += self.notional * np.exp(-self.r * self.T)
                else:
                    cashflow += self.notional * (basket_T / self.basket_0) * np.exp(-self.r * self.T)

            discounted_payoffs.append(cashflow)

        price = np.mean(discounted_payoffs)
        stderr = np.std(discounted_payoffs, ddof=1) / np.sqrt(len(discounted_payoffs))
        return float(price), float(stderr)

    def price_monte_carlo(self, n_paths=100_000, seed=None):
        """Price the multi-index autocall using Monte Carlo simulation.

        Returns:
        - price (float): estimated present value
        - stderr (float): standard error of the estimate
        """
        if seed is not None:
            np.random.seed(seed)
        return self._calculate_autocall_price(n_paths)