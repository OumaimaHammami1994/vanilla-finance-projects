"# Autocall Pricing Project

This project implements a Monte Carlo-based pricer for autocallable structured products, featuring a Streamlit web interface for interactive pricing with real market data.

## Features

- **Autocall Payoff Logic**: Implements  payoff rules with multiple barriers (protection, coupon, autocall).
- **Streamlit Web App**: User-friendly interface for pricing autocalls with inputs for tickers, barriers, and parameters. Fetches live market prices using yfinance.
- **Modular Architecture**: Organized package structure under `pricing/autocalls/` for maintainability.
- **Unit Tests**: Comprehensive tests for validation of pricing logic and edge cases.

## Autocall Payoff Rules

The autocall pricer implements the following payoff structure:

- At each observation time `t_i`:
  - If `S_t >= autocall_barrier * S0` → Auto-call: Pay `notional * (1 + coupon)` at `t_i` and terminate.
  - Else if `S_t >= coupon_barrier * S0` → Pay coupon: `notional * coupon` at `t_i` and continue.
  - Else → No payment and continue.
- At maturity `T` (if never auto-called):
  - If `S_T >= protection_barrier * S0` → Return capital: `notional` at `T`.
  - Else → Downside exposure: `notional * (S_T / S0)` at `T`.

All cashflows are discounted to present value using the constant risk-free rate `r`.

## Installation

1. Clone the repository and navigate to the project directory.
2. Create a virtual environment (optional but recommended):
   ```
   python -m venv pricer_env
   ```
3. Activate the environment:
   - Windows: `pricer_env\Scripts\activate`
   - macOS/Linux: `source pricer_env/bin/activate`
4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Web Application
Launch the interactive Streamlit app:
```
streamlit run streamlit_app.py
```
This opens a web interface where you can input ticker symbols, barrier levels, and other parameters to price autocalls in real-time.


### Running Tests
Execute the unit tests to validate the implementation:
```
python -m pytest tests/test_autocall.py -v
```


## Project Structure

```
vanilla pricer/
├── pricing/
│   └── autocalls/
│       ├── __init__.py
│       └── vanilla_autocall.py  # Main Autocall class
├── tests/
│   └── test_autocall.py         # Unit tests
├── streamlit_app.py             # Web interface
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

