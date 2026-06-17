"""
features.py
-----------
Cac ham Feature Engineering nang cao va tao Multi-Horizon Tensor.

Triet ly: Feature Engineering quyet dinh hon kien truc mo hinh.
Lag features + Crack Spread + Cyclical Encoding = inputs manh.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


# ------------------------------------------------------------------ #
#  ADF Stationarity Test
# ------------------------------------------------------------------ #
def adf_test(series: pd.Series, name: str = "Series") -> dict:
    """
    Kiem dinh Augmented Dickey-Fuller.
    Chung minh chuoi khong dung -> can dung LSTM/DL thay model tuyen tinh.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    pval = result[1]
    is_stationary = pval < 0.05
    print(f"\n{'='*50}")
    print(f"ADF Test - {name}")
    print(f"{'='*50}")
    print(f"  ADF Statistic : {result[0]:.4f}")
    print(f"  p-value       : {pval:.6f}")
    for key, val in result[4].items():
        print(f"  Critical ({key}) : {val:.4f}")
    print(f"  => {'DUNG (Stationary)' if is_stationary else 'KHONG DUNG (Non-stationary)'}")
    return {"statistic": result[0], "pvalue": pval, "stationary": is_stationary}


# ------------------------------------------------------------------ #
#  Cross-Correlation Analysis
# ------------------------------------------------------------------ #
def cross_correlation(series_x: pd.Series, series_y: pd.Series,
                       max_lag: int = 20, name_x: str = "X",
                       name_y: str = "Y") -> pd.Series:
    """
    Tinh tuong quan tre (Cross-Correlation): X(t-k) vs Y(t).
    Xac dinh do tre tac dong cua USD_Index / GPR len MG95.
    """
    x_norm = (series_x - series_x.mean()) / (series_x.std() + 1e-8)
    y_norm = (series_y - series_y.mean()) / (series_y.std() + 1e-8)
    lags   = range(0, max_lag + 1)
    xcorr  = pd.Series(
        [y_norm.corr(x_norm.shift(lag)) for lag in lags],
        index=lags,
        name=f"corr({name_x}(t-k), {name_y}(t))"
    )
    best_lag = xcorr.abs().idxmax()
    print(f"Cross-Corr {name_x} -> {name_y}: best lag = {best_lag} ngay "
          f"(corr = {xcorr[best_lag]:.4f})")
    return xcorr


# ------------------------------------------------------------------ #
#  Multi-Horizon Window Builder
# ------------------------------------------------------------------ #
def make_multihorizon_windows(X_scaled: np.ndarray,
                               y_series: np.ndarray,
                               time_steps: int,
                               horizons: list = [1, 5, 10]) -> dict:
    """
    Tao windows cho nhieu horizon cuong do cung tap du lieu.

    Tra ve dict {H: (X_windows, y_windows)} cho moi horizon.
    Dung cho Multi-Horizon evaluation (chuon chuan GUMNet-WF).
    """
    result = {}
    for h in horizons:
        Xo, yo = [], []
        max_i = len(X_scaled) - h
        for i in range(time_steps, max_i + 1):
            Xo.append(X_scaled[i - time_steps:i, :])
            yo.append(y_series[i + h - 1])
        result[h] = (
            np.array(Xo, dtype=np.float32),
            np.array(yo, dtype=np.float32)
        )
        print(f"  H={h:2d}: {result[h][0].shape[0]} mau, "
              f"shape={result[h][0].shape}")
    return result


# ------------------------------------------------------------------ #
#  Chronological Split helper for Walk-Forward
# ------------------------------------------------------------------ #
def walkforward_folds(n_total: int,
                       n_pretrain: int,
                       n_folds: int = 8,
                       fold_size: int = None) -> list:
    """
    Sinh danh sach (train_end, val_start, val_end) cho Walk-Forward.

    Expanding Window Strategy: tap train ngay cang lon.
    Mo phong dung quy trinh giao dich thuc te (Concept Drift).
    """
    if fold_size is None:
        fold_size = max(30, (n_total - n_pretrain) // (n_folds + 1))

    folds = []
    for f in range(n_folds):
        val_start = n_pretrain + f * fold_size
        val_end   = val_start + fold_size
        if val_end > n_total:
            break
        folds.append({
            "fold":       f + 1,
            "train_end":  val_start,     # Expanding: train tu dau den val_start
            "val_start":  val_start,
            "val_end":    val_end,
        })
    return folds


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from data_loader import load_and_engineer, TARGET
    import warnings; warnings.filterwarnings("ignore")

    df = load_and_engineer()
    adf_test(df[TARGET], "MG95 (chuoi goc)")
    adf_test(df[TARGET].diff().dropna(), "MG95 (sai phan bac 1)")
    cross_correlation(df["USD_Index"], df[TARGET], max_lag=15,
                      name_x="USD_Index", name_y="MG95")
    print("\n[features.py] OK")
