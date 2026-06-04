import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import r2_score

# 1. Load and clean data
train = pd.read_csv('data/train_data.csv')
test = pd.read_csv('data/test_data.csv')

# Clean columns
train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()

# Define maturities
# 3M is 0.25Y. The others are:
# 6M=0.5, 9M=0.75, 1Y=1.0, 2Y=2.0, 5Y=5.0, 10Y=10.0, 20Y=20.0, 30Y=30.0
maturities_train = {
    'ZC025YR': 0.25,
    'ZC050YR': 0.50,
    'ZC075YR': 0.75,
    'ZC100YR': 1.00,
    'ZC200YR': 2.00,
    'ZC500YR': 5.00,
    'ZC1000YR': 10.00,
    'ZC2000YR': 20.00,
    'ZC3000YR': 30.00
}

maturities_test = {
    'ZC025YR': 0.25,
    'ZC050YR': 0.50,
    'ZC075YR': 0.75,
    'ZC100YR': 1.00,
    'ZC200YR': 2.00
}

# Target maturities to predict (6M through 30Y for train, 6M through 2Y for test)
targets_train_cols = ['ZC050YR', 'ZC075YR', 'ZC100YR', 'ZC200YR', 'ZC500YR', 'ZC1000YR', 'ZC2000YR', 'ZC3000YR']
targets_test_cols = ['ZC050YR', 'ZC075YR', 'ZC100YR', 'ZC200YR']

# Short rate is ZC025YR (3M)
r_train = train['ZC025YR'].values
r_test = test['ZC025YR'].values

# Actual curves
Y_train_actual = train[targets_train_cols].values
Y_test_actual = test[targets_test_cols].values

# 2. CIR Yield formula
def cir_yield(r, tau, kappa, theta, sigma):
    # Ensure parameter constraints
    if kappa <= 0 or theta <= 0 or sigma <= 0:
        return np.ones_like(r) * 1e10
    
    h = np.sqrt(kappa**2 + 2 * sigma**2)
    
    # Avoid division by zero
    num_A = 2 * h * np.exp((kappa + h) * tau / 2)
    den_A = 2 * h + (kappa + h) * (np.exp(h * tau) - 1)
    
    # Calculate A and B
    # Handle overflow or invalid values by capping
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        A = (num_A / den_A) ** (2 * kappa * theta / sigma**2)
        B = 2 * (np.exp(h * tau) - 1) / den_A
        
        # Calculate yield
        y = (B * r - np.log(A)) / tau
    
    return y

# 3. Model performance evaluation
def evaluate_model(kappa, theta, sigma, df, target_cols, r_val, maturities_dict):
    preds = []
    actuals = []
    for col in target_cols:
        tau = maturities_dict[col]
        y_pred = cir_yield(r_val, tau, kappa, theta, sigma)
        y_act = df[col].values
        preds.append(y_pred)
        actuals.append(y_act)
        
    preds = np.array(preds).T # Shape: (N, num_targets)
    actuals = np.array(actuals).T
    
    # Calculate R2 pooled
    r2 = r2_score(actuals.flatten(), preds.flatten())
    
    # Also calculate R2 per maturity
    r2s = {}
    for i, col in enumerate(target_cols):
        r2s[col] = r2_score(actuals[:, i], preds[:, i])
        
    return r2, r2s, preds

# --- Method 1: OLS on Short Rate ---
# We estimate physical parameters kappa, theta, sigma
dt = 1/252 # daily data
r_diff = np.diff(r_train)
r_t = r_train[:-1]

# Divide by sqrt(r_t)
y_ols = r_diff / np.sqrt(r_t)
x1_ols = 1.0 / np.sqrt(r_t)
x2_ols = np.sqrt(r_t)

# Design matrix for OLS without intercept: y = beta0 * x1 + beta1 * x2
X_ols = np.column_stack((x1_ols, x2_ols))
beta, residuals, rank, s = np.linalg.lstsq(X_ols, y_ols, rcond=None)

beta0, beta1 = beta
kappa_ols = -beta1 / dt
theta_ols = -beta0 / beta1 / dt # Wait! beta0 = kappa * theta * dt, beta1 = -kappa * dt => beta0/beta1 = -theta => theta = -beta0/beta1. No dt needed!
# Let's double check: beta0 = kappa*theta*dt. beta1 = -kappa*dt. So beta0 / beta1 = -theta.
# Thus theta_ols = -beta0 / beta1.
# Let's check residuals to get sigma
resid = y_ols - (beta0 * x1_ols + beta1 * x2_ols)
sigma_ols = np.sqrt(np.var(resid) / dt)

print("--- OLS Physical Parameters ---")
print(f"kappa: {kappa_ols:.6f}")
print(f"theta: {theta_ols:.6f}")
print(f"sigma: {sigma_ols:.6f}")
feller_ols = 2 * kappa_ols * theta_ols - sigma_ols**2
print(f"Feller condition (2*kappa*theta >= sigma^2): {feller_ols:.6f} ({'Satisfied' if feller_ols >= 0 else 'Violated'})")

# Evaluate OLS on Train and Test
r2_train_ols, r2s_train_ols, _ = evaluate_model(kappa_ols, theta_ols, sigma_ols, train, targets_train_cols, r_train, maturities_train)
r2_test_ols, r2s_test_ols, _ = evaluate_model(kappa_ols, theta_ols, sigma_ols, test, targets_test_cols, r_test, maturities_test)
print(f"OLS Train R2: {r2_train_ols:.6f}")
print(f"OLS Test R2: {r2_test_ols:.6f}")
print("OLS Test R2 per maturity:", r2s_test_ols)


# --- Method 2: Cross-Sectional Yield Curve Calibration ---
# Find parameters that minimize yield prediction error on the train set
def loss_func(params):
    kappa, theta, sigma = params
    if kappa <= 0 or theta <= 0 or sigma <= 0:
        return 1e10
    
    total_sse = 0
    for col in targets_train_cols:
        tau = maturities_train[col]
        y_pred = cir_yield(r_train, tau, kappa, theta, sigma)
        y_act = train[col].values
        total_sse += np.sum((y_act - y_pred)**2)
    return total_sse

# Run optimization
initial_guess = [0.1, 0.02, 0.05]
bounds = [(1e-5, 10.0), (1e-5, 0.5), (1e-5, 0.5)]
res = minimize(loss_func, initial_guess, bounds=bounds, method='L-BFGS-B')

kappa_cs, theta_cs, sigma_cs = res.x
print("\n--- Cross-Sectional Risk-Neutral Parameters ---")
print(f"kappa: {kappa_cs:.6f}")
print(f"theta: {theta_cs:.6f}")
print(f"sigma: {sigma_cs:.6f}")
feller_cs = 2 * kappa_cs * theta_cs - sigma_cs**2
print(f"Feller condition (2*kappa*theta >= sigma^2): {feller_cs:.6f} ({'Satisfied' if feller_cs >= 0 else 'Violated'})")

# Evaluate CS on Train and Test
r2_train_cs, r2s_train_cs, _ = evaluate_model(kappa_cs, theta_cs, sigma_cs, train, targets_train_cols, r_train, maturities_train)
r2_test_cs, r2s_test_cs, _ = evaluate_model(kappa_cs, theta_cs, sigma_cs, test, targets_test_cols, r_test, maturities_test)
print(f"CS Train R2: {r2_train_cs:.6f}")
print(f"CS Test R2: {r2_test_cs:.6f}")
print("CS Test R2 per maturity:", r2s_test_cs)
