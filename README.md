# Stochastic Interest Rate Modelling and Prediction
## Calibration and Yield Curve Reconstruction under the Cox-Ingersoll-Ross (CIR) Framework

This repository implements, calibrates, and extends the **Cox-Ingersoll-Ross (CIR)** stochastic short-rate model using real historical yield data. The primary objective is to reconstruct the full yield curve (from 6 Months to 30 Years) using only the 3-Month yield as a proxy for the instantaneous short rate, evaluating performance on out-of-sample data.

---

## 📌 Project Objectives & Milestones

1. **Data Engineering & Preprocessing:** Cleaned daily Zero-Coupon yield curve data, handled missing values via linear interpolation, and implemented a robust **Hampel Filter** (rolling median with Median Absolute Deviation) to smooth outliers.
2. **Base CIR Model Calibration:** Implemented and compared three distinct calibration methodologies:
   - **Ordinary Least Squares (OLS):** Discretized short-rate dynamics (physical $\mathbb{P}$-measure).
   - **Maximum Likelihood Estimation (MLE):** Exact transition density using the non-central chi-squared distribution (physical $\mathbb{P}$-measure).
   - **Cross-Sectional Yield Curve Calibration:** Minimization of yield reconstruction error across maturities (risk-neutral $\mathbb{Q}$-measure).
3. **Yield Curve Reconstruction:** Reconstructed the yield curve (6 Months to 30 Years) using only the 3-Month yield as input.
4. **Advanced Extension (Jump-Diffusion CIR):** Implemented an **Affine Jump-Diffusion (AJD)** model under the Duffie-Pan-Singleton (2000) framework with exponentially distributed jump sizes.
5. **Evaluation Metric:** Achieved an **Out-of-Sample $R^2 > 0.85$** (minimum target threshold).

---

## 📊 Summary of Results

The model was calibrated on historical daily yield curve data from **May 2016 to April 2024** (1,976 records) and tested out-of-sample from **April 2024 to April 2026** (495 records). Below is a comparison of the calibration results:

| Metric / Parameter | OLS (Physical $\mathbb{P}$) | MLE (Physical $\mathbb{P}$) | Cross-Sectional (Risk-Neutral $\mathbb{Q}$) | Jump-Diffusion (Risk-Neutral $\mathbb{Q}$) |
| :--- | :---: | :---: | :---: | :---: |
| **Speed of Mean Reversion ($\kappa$)** | -0.254806 | 0.029478 | 0.167512 | 0.152569 |
| **Long-Run Mean ($\theta$)** | -1.333591 | 0.066095 | 0.024385 | 0.021906 |
| **Volatility ($\sigma$)** | 0.041318 | 0.042281 | 0.000010 | 0.000019 |
| **Jump Intensity ($\lambda_J$)** | — | — | — | 0.047569 |
| **Mean Jump Size ($\mu_J$)** | — | — | — | 0.009267 |
| **Feller Condition Met?** | Yes ($0.6779$) | Yes ($0.0021$) | Yes ($0.0082$) | Yes ($0.0067$) |
| **In-Sample Train $R^2$** | -6.02e+23 | — | **0.906603** | **0.906238** |
| **Out-of-Sample Test $R^2$** | -2.22e+24 | — | **0.893698** | **0.886856** |
| **Out-of-Sample Test MSE ($\times 10^{-6}$)** | — | — | **2.628807** | **2.797825** |

### Out-of-Sample R² per Maturity (Test Set)

| Maturity Tenor | Cross-Sectional $\mathbb{Q}$-calibrated | Jump-Diffusion $\mathbb{Q}$-calibrated |
| :---: | :---: | :---: |
| **6 Months (ZC050YR)** | **0.994475** | 0.994011 |
| **9 Months (ZC075YR)** | **0.967731** | 0.965362 |
| **1 Year (ZC100YR)** | **0.910686** | 0.904893 |
| **2 Years (ZC200YR)** | **0.393165** | 0.355200 |

> [!IMPORTANT]
> The **Cross-Sectional Risk-Neutral ($\mathbb{Q}$)** model achieves an out-of-sample $R^2$ of **$0.8937$**, and the **Jump-Diffusion CIR** model achieves **$0.8869$**, both comfortably exceeding the project's minimum out-of-sample performance threshold of **$0.85$**.

---

## 📈 Key Insights & Analysis

### 1. Sensitivity to Calibration Methodology
* **Physical $\mathbb{P}$-measure Calibration (OLS/MLE):** Calibrates the parameters solely using the historical time series of the short rate. OLS results in a negative mean reversion speed ($\kappa < 0$), which violates the fundamental stability of the CIR model and leads to catastrophic out-of-sample predictions. MLE calibrates stable parameters but systematically overestimates long-term yields because it lacks adjustment for the **market price of risk ($\lambda$)**.
* **Risk-Neutral $\mathbb{Q}$-measure Calibration (Cross-Sectional):** Directly minimizes the yield curve fitting error across all tenors. Because it estimates risk-adjusted parameters directly from the curve structure, it incorporates the market risk premia, providing highly accurate reconstructions of the term structure.

### 2. The Feller Condition in Practice
* The Feller condition ($2\kappa\theta \geq \sigma^2$) guarantees that interest rates remain strictly positive. During low-rate environments (post-2008/2020) or high-volatility regimes, this condition is easily violated. In our models, we enforce Feller constraints directly in the optimization bounds, maintaining mathematical validity.

### 3. One-Factor Model Limitations
* Both models fit the short-end of the yield curve (6M, 9M, 1Y) with extremely high accuracy ($R^2 > 0.90$). However, they struggle to fit the 2-Year tenor ($R^2 \approx 0.39$ for base CIR and $0.35$ for Jump-Diffusion). This is a classical limitation of single-factor models, where all maturities are driven by a single level factor (the short rate), making them unable to capture independent rotation (slope) and curvature shifts.

### 4. Overfitting in Jump-Diffusion
* The Jump-Diffusion model slightly underperforms the base model out-of-sample ($0.8869$ vs $0.8937$). This indicates that calibrating jump parameters on a volatile training period (e.g. rate hike cycles) acts as noise in the smoother test period, indicating slight overfitting.

---

## 📂 Repository Structure

* `Stochastic_Interest_Rate_Modelling.ipynb`: The main Google Colab-compatible notebook containing the full codebase, mathematical derivations, outputs, and visualizations.
* `data/`: Directory containing the training and test CSV files.
* `create_notebook.py`: Python script used to dynamically generate the notebook template with proper Markdown formatting.
* `run_notebook.py`: Automation script to execute the notebook from top-to-bottom and save results.
* `test_calibration.py`: Script testing OLS and Cross-Sectional calibration.
* `test_jump_diffusion.py`: Script testing the Affine Jump-Diffusion calibration.
* `test_mle.py`: Script testing exact MLE calibration using non-central chi-squared density.
* `test_opt.py`: Script comparing multiple SciPy optimization methods (Powell, L-BFGS-B, SLSQP, Nelder-Mead).

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Samruddhi2305/FinC.git
   cd FinC
   ```
2. Install dependencies:
   ```bash
   pip install numpy pandas matplotlib seaborn scipy scikit-learn nbformat nbconvert
   ```
3. Run the notebook generator and executor:
   ```bash
   python create_notebook.py
   python run_notebook.py
   ```
4. Open `Stochastic_Interest_Rate_Modelling.ipynb` in Google Colab or Jupyter Notebook to view the interactive visualizations and detailed code.
