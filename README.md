"# Vanilla Option Pricing Project

This project implements pricing for vanilla European options using multiple models: Black-Scholes, Monte Carlo simulation, and Binomial model.

## Features

- Black-Scholes pricing for call and put options
- Monte Carlo simulation for option pricing
- Binomial model (Cox-Ross-Rubinstein) for option pricing
- Calculation of option Greeks: Delta, Gamma, Vega
- Unit tests for validation

## Installation

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script to see an example:
```
python main.py
```

Run tests:
```
python -m unittest test_option.py
```

## Files

- `option.py`: Contains the `VanillaOption` class with pricing methods
- `main.py`: Example usage script
- `test_option.py`: Unit tests
- `requirements.txt`: Python dependencies
- `explore.ipynb`: Jupyter notebook for stock data exploration (not directly related to option pricing)" 
 - `autocall.py`: Autocallable product Monte Carlo pricer
 - `main_autocall.py`: Example runner for the autocall pricer
 - `test_autocall.py`: Unit tests for autocall Monte Carlo pricing

### Autocall pricer

The `autocall.py` module implements a simple Monte Carlo pricer for an autocallable product. See `main_autocall.py` for an example of usage and `test_autocall.py` for tests.
