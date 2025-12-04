# Brent Forecasting – Rolling Backtest

Rolling monthly forecasts for Brent Crude using:
- LSTM + Attention
- TCN (heteroskedastic head)
- XGBoost
- Linear Ensemble

Daily data:
- Brent (close target)
- WTI, DXY, USD/INR, VIX
- Yahoo Finance primary
- FRED secondary

Target:
- Next-day log return
- Predictions reconverted to price path

Output:
- Monthly Excel with:
  - lstm
  - tcn
  - xgboost
  - ensemble
  - metrics
- Monte-Carlo simulation plots under:
outputs/simulations/


# Brent Crude Rolling Backtester
A quantitative research framework for recursively forecasting Brent crude prices using deep learning and structured ensemble methods.

The system performs **monthly rolling backtests**:
- Train on historical market data up to end of Month `M`
- Predict log-returns recursively for Month `M+1`
- Convert log-returns → price path
- Evaluate prediction quality
- Ensemble the models using historical NNLS weights (no look-ahead)
- Save monthly prediction reports as Excel files
- (Optional) Monte-Carlo simulation using TCN stochastic head

This engine is designed for **robust, leakage-free forecasting** suitable for production quant research.

---

## ✔ Current Models

### 1. LSTM + Attention
- Recursive forecasting of daily log-returns
- Multi-step prediction via sequence roll-forward

### 2. Temporal Convolutional Network (TCN)
- Causal convolutions
- Heteroskedastic prediction head:  
  predicts mean and log-variance of log returns

### 3. Ridge Regression (coming next)
Planned addition:
- Monthly horizon drift predictor
- Low-variance baseline component for ensemble stability

---

## ✔ Ensemble Logic

### Current Method:
- NNLS non-negative weighting
- Fit using **previous month actuals**
- Apply weights to next month predictions

### Upcoming Enhancements:
- 12-month exponentially-decayed weight memory
- Ridge-constrained non-negative regression
- Monthly regime-based prior weighting

**No future data is ever used.**

---






### Workflow
1. `download_data.py`
2. `prepare_features.py`
3. `dataset_builder.py`
optional(run optuna for optimization)
4. Train models
5. Monthly recursive prediction
6. Save excel & MC plots

Sequence length: 120 days  
Models run on GPU (PyTorch Lightning)

Config stored in `config/configs.py`

commands:
python -m scripts.download_data

python -m scripts.prepare_features

###optuna hyperparameter tuning

python -m optimize.tune_lstm
python -m optimize.tune_tcn
python -m optimize.tune_ridge


python -m scripts.rolling_backtest 
custom args:
    --feature_path data/feature_matrix.csv \
    --excel_dir outputs/backtest_excels \
    --sim_dir outputs/simulations \
    --train_cutoff 2023-12-31 \
    --final_month 2025-11 \
    --seq_len 120 \
    --mc_sims 300




