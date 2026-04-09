# Ethanimine Rate Coefficients from MQCT

This repository contains Python scripts for processing collision cross-sections to calculate rate coefficients for Ethanimine (E-isomer) and Helium (He) interactions. It processes data derived from Mixed Quantum/Classical Theory (MQCT).

## Features
- Calculates state-to-state rate coefficients at various temperatures (5K - 300K).
- Maps exact rotational quantum states `(J, Ka, Kc)` by parsing MQCT log outputs.
- Extrapolates low-energy cross-sections using robust analytical fitting and piecewise integration.
- Automatically generates diagnostic visualizations (2x3 interactive grids) for transition integrals.
- Compares calculated MQCT rate coefficients against standard theoretical scaling laws.

## Required Libraries
To run the analysis, you will need the following Python libraries installed:
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `tqdm`

## Usage
Update the file paths in the `1. CONSTANTS & CONFIGURATION` section of `E_He_MQCT_Rate_Coefficient_15.py` to point to your local `USER_INPUT_CHECK.out` and `E_he_Database.dat` files before running the script.
