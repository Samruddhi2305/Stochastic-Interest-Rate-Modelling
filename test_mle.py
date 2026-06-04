import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import iv # Modified Bessel function of the first kind
from sklearn.metrics import r2_score

# Load data
train = pd.read_csv('data/train_data.csv')
train.columns = train.columns.str.strip()
r_train = train['ZC025YR'].values
dt = 1/252

# Log-likelihood of CIR transition density
def cir_log_likelihood(params):
    kappa, theta, sigma = params
    # Impose constraints
    if kappa <= 0 or theta <= 0 or sigma <= 0 or 2*kappa*theta < sigma**2:
        return 1e10 # Return large value if constraints (including Feller) are violated
        
    r_t = r_train[:-1]
    r_t1 = r_train[1:]
    
    c = 2 * kappa / (sigma**2 * (1 - np.exp(-kappa * dt)))
    u = c * r_t * np.exp(-kappa * dt)
    v = c * r_t1
    q = 2 * kappa * theta / sigma**2 - 1
    
    # We want to calculate: ln(c) - u - v + (q/2)*ln(v/u) + ln(I_q(2*sqrt(u*v)))
    # For large arguments, iv(q, x) can overflow, so we use exponentially scaled bessel ivase
    # iv(q, x) = ive(q, x) * exp(x)
    # So ln(I_q(x)) = ln(I_q_scaled(x)) + x
    from scipy.special import i0e, i1e
    # Since q is not necessarily 0 or 1, we use scipy.special.ive(q, x)
    from scipy.special import ive
    
    x = 2 * np.sqrt(u * v)
    
    # Avoid log of zero or negative numbers in bessel
    with np.errstate(divide='ignore', invalid='ignore'):
        bessel_scaled = ive(q, x)
        # Handle cases where bessel_scaled is 0 or NaN
        bessel_scaled = np.maximum(bessel_scaled, 1e-30)
        log_bessel = np.log(bessel_scaled) + x
        
        log_pdf = np.log(c) - u - v + 0.5 * q * np.log(v / u) + log_bessel
        
    # Return negative log-likelihood
    # Filter out NaNs or infs
    log_pdf = log_pdf[np.isfinite(log_pdf)]
    if len(log_pdf) < len(r_t) - 10:
        return 1e10
    return -np.sum(log_pdf)

# Run optimization for MLE
initial_guess = [0.1, 0.02, 0.04]
bounds = [(1e-5, 5.0), (1e-5, 0.1), (1e-5, 0.1)]
res = minimize(cir_log_likelihood, initial_guess, bounds=bounds, method='L-BFGS-B')

print("--- MLE Calibrated Parameters ---")
if res.success:
    kappa_mle, theta_mle, sigma_mle = res.x
    print(f"kappa: {kappa_mle:.6f}")
    print(f"theta: {theta_mle:.6f}")
    print(f"sigma: {sigma_mle:.6f}")
    feller_mle = 2 * kappa_mle * theta_mle - sigma_mle**2
    print(f"Feller condition: {feller_mle:.6f} ({'Satisfied' if feller_mle >= 0 else 'Violated'})")
else:
    print("MLE Optimization failed:", res.message)
