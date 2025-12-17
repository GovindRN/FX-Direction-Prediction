# FX Direction Prediction with Keras DNN Model Builder (EUR/USD)

This repo contains:
1) a **TensorFlow/Keras model builder** module for binary classification, plus helpers for **reproducibility** and **class weighting**  
2) an **end-to-end EUR/USD trading research script** that engineers technical-indicator features, trains a base ML model, then trains a DNN meta-model and runs a simple backtest.

---

## Contents

### 1) `DNNModelWithML` (Model Utilities)
- `set_seeds(seed=100)`: sets Python/NumPy/TensorFlow seeds
- `cw(df)`: balanced class weights from `df["dir"]` (expects binary 0/1)
- `create_model(...)`: Functional API MLP (ReLU hidden, sigmoid output)
- `create_sequential_model(...)`: same architecture using Sequential API

Models compile with:
- loss: `binary_crossentropy`
- optimizer: `Adam(lr=1e-4)`
- metric: `accuracy`

### 2) EUR/USD Pipeline Script
End-to-end script that:
- loads tab-delimited EUR/USD OHLCV data (`<DATE>`, `<TIME>` → datetime index)
- builds target: `dir = 1` if log return > 0 else `0`
- adds indicators via `ta.add_all_ta_features`
- creates **lagged features** (default 5 lags) for selected indicators
- trains base ML model(s) (currently `LogisticRegression`) and adds `model_i_pred` as feature(s)
- trains a TensorFlow DNN meta-model using `create_model(...)` + `cw(...)`
- converts predicted probabilities into positions and plots strategy equity curves
- saves:
  - `DNN_model.keras`
  - `params.pkl` containing `{"mu": mu, "std": std}`

---

## Requirements
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib ta
````

---

## Quick Start

### Model utilities

```python
from DNNModelWithML import set_seeds, cw, create_model

set_seeds(100)

model = create_model(
    input_dim=X_train.shape[1],
    hl=5, hu=64,
    dropout=True, rate=0.2,
    regularize=True
)
```

### Run the EUR/USD script

1. Place your dataset under `datasets/` and set the path in the script:

   ```python
   df = preprocessing("datasets/EURUSD_M20_202001020600_202312292340.csv")
   ```
2. Run:

   ```bash
   python your_script.py
   ```

---

## Backtest Logic (Script)

* Buy if `proba > 0.53`, sell if `proba < 0.47`, else hold previous position
* Trades only during NY hours **02:00–12:00** (otherwise flat)
* Computes:

  * `creturns` (buy-and-hold)
  * `cstrategy` (gross strategy)
  * `cstrategy_net` (net of trading cost via `ptc`)

---

## Notes

* `cw(df)` expects a DataFrame column named `dir` with binary labels (0/1).
* Train/test split is time-based (first 66% train, last 34% test).
* The script uses two normalizations:

  * `StandardScaler` for the base ML model features
  * z-score normalization `(df - mu) / std` for the DNN (saved to `params.pkl`)
* GPU configuration is attempted (uses first GPU if available).

---

## Outputs

* Interactive plots (probability histogram + equity curves)
* `DNN_model.keras`
* `params.pkl` (`mu`, `std`)

---

## Disclaimer

Research/backtesting code only — not financial advice. Add robust validation (walk-forward, slippage, sensitivity testing) before any real-world use.
