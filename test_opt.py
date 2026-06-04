import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import r2_score
import test_calibration as tc

# We will test different optimization methods and initial guesses
methods = ['L-BFGS-B', 'SLSQP', 'Nelder-Mead', 'Powell']
initial_guesses = [
    [0.1, 0.02, 0.05],
    [0.5, 0.03, 0.02],
    [1.0, 0.05, 0.01],
    [0.05, 0.01, 0.001]
]

best_test_r2 = -1.0
best_params = None
best_method = None

for method in methods:
    for guess in initial_guesses:
        if method in ['L-BFGS-B', 'SLSQP']:
            res = minimize(tc.loss_func, guess, bounds=[(1e-5, 20.0), (1e-5, 0.5), (1e-6, 0.5)], method=method)
        else:
            # Nelder-Mead and Powell do not strictly require bounds, or we can use bounds/penalties
            res = minimize(tc.loss_func, guess, method=method)
            
        if res.success:
            k, th, sig = res.x
            # Check constraints
            if k > 0 and th > 0 and sig > 0:
                r2_train, _, _ = tc.evaluate_model(k, th, sig, tc.train, tc.targets_train_cols, tc.r_train, tc.maturities_train)
                r2_test, _, _ = tc.evaluate_model(k, th, sig, tc.test, tc.targets_test_cols, tc.r_test, tc.maturities_test)
                print(f"Method: {method}, Guess: {guess} => Params: ({k:.4f}, {th:.4f}, {sig:.6f}) => Train R2: {r2_train:.4f}, Test R2: {r2_test:.4f}")
                if r2_test > best_test_r2:
                    best_test_r2 = r2_test
                    best_params = (k, th, sig)
                    best_method = (method, guess)

print("\n--- Best Results ---")
print("Best Test R2:", best_test_r2)
print("Best Params:", best_params)
print("Best Method/Guess:", best_method)
