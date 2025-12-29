import unittest
from pricing.autocalls.vanilla_autocall import Autocall


class TestAutocall(unittest.TestCase):
    def test_price_bounds_and_repeatability(self):
        # Use protection_barrier=0.7, coupon_barrier=0.9, autocall_barrier=1.0
        ac = Autocall(S0=100, notional=1000, coupon=0.05, coupon_barrier=0.9, autocall_barrier=1.0, protection_barrier=0.7, n_obs=3, T=1.0, r=0.01, sigma=0.2)
        p1, se1 = ac.price_monte_carlo(n_paths=20000, seed=42)
        p2, se2 = ac.price_monte_carlo(n_paths=20000, seed=42)
        # Reproducible with same seed
        self.assertAlmostEqual(p1, p2, places=8)
        # Price should be between 0 and notional*(1+coupon)
        self.assertGreaterEqual(p1, 0.0)
        self.assertLessEqual(p1, ac.notional * (1.0 + ac.coupon))

    def test_edge_cases(self):
        # Barrier very low -> likely autocalled immediately -> price approx immediate autocalled payoff
        ac_low = Autocall(S0=100, notional=1000, coupon=0.05, coupon_barrier=0.01, autocall_barrier=0.01, protection_barrier=0.0, n_obs=1, T=1.0, r=0.0, sigma=0.01)
        p_low, _ = ac_low.price_monte_carlo(n_paths=10000, seed=1)
        self.assertAlmostEqual(p_low, ac_low.notional * (1.0 + ac_low.coupon), delta=ac_low.notional*0.01)

    def test_coupons_accumulate_and_maturity_return(self):
        # Construct a scenario with very low volatility so paths stay near S0.
        # Set autocall_barrier very high so autocall never happens, coupon_barrier low so coupons are always paid, protection_barrier=0.5 so capital returned at maturity.
        ac = Autocall(S0=100, notional=1000, coupon=0.02, coupon_barrier=0.8, autocall_barrier=10.0, protection_barrier=0.5, n_obs=3, T=1.0, r=0.0, sigma=0.001)
        p, _ = ac.price_monte_carlo(n_paths=20000, seed=7)
        # Expected approx: sum of three coupons + notional at maturity (no discount)
        expected = ac.notional * (1.0 + ac.coupon * ac.n_obs)
        self.assertAlmostEqual(p, expected, delta=ac.notional*0.02)


if __name__ == '__main__':
    unittest.main()
