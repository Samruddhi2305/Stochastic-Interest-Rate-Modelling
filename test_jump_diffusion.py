import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import quad
from sklearn.metrics import r2_score

# Load and clean data
train = pd.read_csv('data/train_data.csv')
test = pd.read_csv('data/test_data.csv')
train.columns = train.columns.str.strip()
test.columns = test.columns.str.strip()

targets_train_cols = ['ZC050YR', 'ZC075YR', 'ZC100YR', 'ZC200YR', 'ZC500YR', 'ZC1000YR', 'ZC2000YR', 'ZC3000YR']
targets_test_cols = ['ZC050YR', 'ZC075YR', 'ZC100YR', 'ZC200YR']

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

r_train = train['ZC025YR'].values
r_test = test['ZC025YR'].values

# B(tau) is standard CIR B(tau)
def get_B(tau, kappa, sigma):
    h = np.sqrt(kappa**2 + 2 * sigma**2)
    den = 2 * h + (kappa + h) * (np.exp(h * tau) - 1)
    B = 2 * (np.exp(h * tau) - 1) / den
    return B

# A_CIR(tau) is standard CIR A(tau)
def get_log_A_CIR(tau, kappa, theta, sigma):
    h = np.sqrt(kappa**2 + 2 * sigma**2)
    num = 2 * h * np.exp((kappa + h) * tau / 2)
    den = 2 * h + (kappa + h) * (np.exp(h * tau) - 1)
    log_A = (2 * kappa * theta / sigma**2) * np.log(num / den)
    return log_A

# Jump diffusion A(tau)
def get_log_A_JD(tau, kappa, theta, sigma, lambda_J, mu_J):
    log_A_cir = get_log_A_CIR(tau, kappa, theta, sigma)
    if lambda_J == 0 or mu_J == 0:
        return log_A_cir
    
    # We need to integrate: - lambda_J * mu_J * B(s) / (1 + mu_J * B(s)) from 0 to tau
    # Since B(s) is very smooth, we can use quad
    def integrand(s):
        B_s = get_B(s, kappa, sigma)
        return - (lambda_J * mu_J * B_s) / (1.0 + mu_J * B_s)
    
    val, _ = quad(integrand, 0.0, tau)
    return log_A_cir + val

def cir_jd_yield(r, tau, kappa, theta, sigma, lambda_J, mu_J):
    B = get_B(tau, kappa, sigma)
    log_A = get_log_A_JD(tau, kappa, theta, sigma, lambda_J, mu_J)
    y = (B * r - log_A) / tau
    return y

# Evaluate Jump Diffusion
def evaluate_jd(kappa, theta, sigma, lambda_J, mu_J, df, target_cols, r_val, maturities_dict):
    preds = []
    actuals = []
    for col in target_cols:
        tau = maturities_dict[col]
        # Predict yield for this maturity
        y_pred = cir_jd_yield(r_val, tau, kappa, theta, sigma, lambda_J, mu_J)
        y_act = df[col].values
        preds.append(y_pred)
        actuals.append(y_act)
        
    preds = np.array(preds).T
    actuals = np.array(actuals).T
    
    r2 = r2_score(actuals.flatten(), preds.flatten())
    r2s = {}
    for i, col in enumerate(target_cols):
        r2s[col] = r2_score(actuals[:, i], preds[:, i])
    return r2, r2s, preds

# Objective function for calibration
def loss_func_jd(params):
    kappa, theta, sigma, lambda_J, mu_J = params
    if kappa <= 0 or theta <= 0 or sigma <= 0 or lambda_J < 0 or mu_J < 0:
        return 1e10
    
    total_sse = 0
    # Vectorized check: pre-calculate log_A and B for each maturity
    for col in targets_train_cols:
        tau = maturities_train[col]
        y_pred = cir_jd_yield(r_train, tau, kappa, theta, sigma, lambda_J, mu_J)
        y_act = train[col].values
        total_sse += np.sum((y_act - y_pred)**2)
    return total_sse

# Run optimization
initial_guess = [0.16, 0.024, 0.0001, 0.05, 0.01]
bounds = [(1e-5, 10.0), (1e-5, 0.5), (1e-6, 0.5), (0.0, 5.0), (0.0, 0.5)]
res = minimize(loss_func_jd, initial_guess, bounds=bounds, method='L-BFGS-B')

kappa_jd, theta_jd, sigma_jd, lambda_jd, mu_jd = res.x
print("--- Jump-Diffusion Calibrated Parameters ---")
print(f"kappa: {kappa_jd:.6f}")
print(f"theta: {theta_jd:.6f}")
print(f"sigma: {sigma_jd:.6f}")
print(f"lambda_J: {lambda_jd:.6f}")
print(f"mu_J: {mu_jd:.6f}")
feller_jd = 2 * kappa_jd * theta_jd - sigma_jd**2
print(f"Feller condition (2*kappa*theta >= sigma^2): {feller_jd:.6f} ({'Satisfied' if feller_jd >= 0 else 'Violated'})")

# Evaluate JD on Train and Test
r2_train, r2s_train, _ = evaluate_jd(kappa_jd, theta_jd, sigma_jd, lambda_jd, mu_jd, train, targets_train_cols, r_train, maturities_train)
r2_test, r2s_test, _ = evaluate_jd(kappa_jd, theta_jd, sigma_jd, lambda_jd, mu_jd, test, targets_test_cols, r_test, maturities_test)
print(f"JD Train R2: {r2_train:.6f}")
print(f"JD Test R2: {r2_test:.6f}")
print("JD Test R2 per maturity:", r2s_test)
