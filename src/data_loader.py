"""
data_loader.py
--------------
Load du lieu, RobustScaler, chronological split va Sliding Window.

Triet ly (theo GUMNet-WF): RobustScaler chong nhieu outlier tu cac cu
soc thi truong (chien tranh, dai dich). Chia du lieu theo truc thoi
gian tuyet doi - khong random - de tranh Data Leakage.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from pathlib import Path

# ------------------------------------------------------------------ #
#  Constants
# ------------------------------------------------------------------ #
DATA_PATH    = Path(__file__).parent.parent / "data" / "processed" / "clean_data_exo_ver1.csv"
TARGET       = "MG95"
TIME_STEPS   = 30           # Sequence length mac dinh (cho H=1,5)
TRAIN_RATIO  = 0.80
VAL_RATIO    = 0.10
# TEST_RATIO  = 0.10 (remaining)

# SEQ_LEN dong (theo GUMNet-WF Implementation Details)
SEQ_LEN_BY_HORIZON = {1: 15, 5: 20, 10: 45, 30: 60}


# ------------------------------------------------------------------ #
#  Load & Feature Engineering
# ------------------------------------------------------------------ #
def load_and_engineer(data_path: str = None, target_col: str = TARGET) -> pd.DataFrame:
    """
    Doc file CSV, phan tich datetime index, tien xu ly & feature engineering.

    Tra ve DataFrame da co day du features.
    """
    path = data_path or DATA_PATH
    df = pd.read_csv(path, parse_dates=["Ngày"], index_col="Ngày")
    df.columns = df.columns.str.strip()

    # Forward-fill gia tri thieu (phu hop time series - khong dung mean/median)
    df = df.ffill().bfill()

    # ---- Rolling Statistics ----
    for col in [target_col, "WTI"]:
        for w in [7, 30]:
            df[f"{col}_MA{w}"]  = df[col].rolling(w, min_periods=1).mean()
            df[f"{col}_Vol{w}"] = df[col].rolling(w, min_periods=1).std().fillna(0)

    # ---- Lag Features (cam hung tu PACF analysis) ----
    for lag in [1, 2, 3, 5, 7, 14, 30]:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
        df[f"WTI_lag{lag}"]  = df["WTI"].shift(lag)

    # ---- Rate of Change ----
    df[f"{target_col}_ROC7"]  = df[target_col].pct_change(7)
    df[f"{target_col}_ROC30"] = df[target_col].pct_change(30)
    df["WTI_ROC7"]   = df["WTI"].pct_change(7)

    # ---- Crack Spread (theo GUMNet-WF: bien dan dat gia xang) ----
    df["Crack_Spread"]     = df[target_col] - df["WTI"]      # Bien loi nhuan tinh che
    df["Brent_WTI_Spread"] = df["BRT DTD"] - df["WTI"]

    # ---- Cyclical Encoding (theo GUMNet-WF: tinh tuan hoan tuan giao dich) ----
    df["Month"]      = df.index.month
    df["Quarter"]    = df.index.quarter
    df["DayOfWeek"]  = df.index.dayofweek
    df["Year"]       = df.index.year
    df["Month_sin"]  = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"]  = np.cos(2 * np.pi * df["Month"] / 12)
    df["DOW_sin"]    = np.sin(2 * np.pi * df["DayOfWeek"] / 5)
    df["DOW_cos"]    = np.cos(2 * np.pi * df["DayOfWeek"] / 5)

    df = df.dropna()
    return df


# ------------------------------------------------------------------ #
#  Multi-Horizon Target Creation
# ------------------------------------------------------------------ #
def make_multihorizon_targets(df: pd.DataFrame,
                               horizons: list = [1, 5, 10]) -> pd.DataFrame:
    """
    Tao cac cot target MG95_H1, MG95_H5, MG95_H10 (gia tuong lai).
    Su dung cho Multi-Horizon evaluation.
    """
    for h in horizons:
        df[f"MG95_H{h}"] = df["MG95"].shift(-h)
    return df


# ------------------------------------------------------------------ #
#  Scaling
# ------------------------------------------------------------------ #
def fit_scalers(df: pd.DataFrame, feature_cols: list, target_col: str = TARGET):
    """
    Fit RobustScaler tren tap train.
    Ly do RobustScaler: Gia dau co nhieu outlier cu soc (Nga-Ukraine 2022,
    Covid 2020). RobustScaler dung median & IQR, khong bi lech boi outlier.
    """
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()
    scaler_X.fit(df[feature_cols].values)
    scaler_y.fit(df[[target_col]].values)
    return scaler_X, scaler_y


def transform_data(df: pd.DataFrame, scaler_X, scaler_y,
                   feature_cols: list, target_col: str = TARGET):
    X_scaled = scaler_X.transform(df[feature_cols].values)
    y_scaled = scaler_y.transform(df[[target_col]].values).flatten()
    return X_scaled, y_scaled


def inverse_transform_y(arr: np.ndarray, scaler_y) -> np.ndarray:
    return scaler_y.inverse_transform(arr.reshape(-1, 1)).flatten()


# ------------------------------------------------------------------ #
#  Sliding Window
# ------------------------------------------------------------------ #
def make_windows(X: np.ndarray, y: np.ndarray,
                 time_steps: int = TIME_STEPS,
                 horizon: int = 1):
    """
    Chuyen doi du lieu 2D [N, F] sang tensor 3D [samples, time_steps, features].

    Moi mau: cua so [t-time_steps : t] -> du bao gia tai t + horizon - 1.
    Day la dinh dang bat buoc cho LSTM / iTransformer / GUMNet.
    """
    Xo, yo = [], []
    max_i = len(X) - horizon
    for i in range(time_steps, max_i + 1):
        Xo.append(X[i - time_steps:i, :])
        yo.append(y[i + horizon - 1])
    return np.array(Xo, dtype=np.float32), np.array(yo, dtype=np.float32)


# ------------------------------------------------------------------ #
#  Chronological Split
# ------------------------------------------------------------------ #
def chronological_split(X_win: np.ndarray, y_win: np.ndarray,
                         train_ratio: float = TRAIN_RATIO,
                         val_ratio: float = VAL_RATIO):
    """
    Chia du lieu theo truc thoi gian (Chronological Split).
    TUYET DOI khong dung random split - tranh Data Leakage.
    """
    n  = len(X_win)
    nt = int(n * train_ratio)
    nv = int(n * val_ratio)

    splits = {
        "X_train": X_win[:nt],          "y_train": y_win[:nt],
        "X_val":   X_win[nt:nt + nv],   "y_val":   y_win[nt:nt + nv],
        "X_test":  X_win[nt + nv:],     "y_test":  y_win[nt + nv:],
        "n_train": nt, "n_val": nv,
    }
    return splits


# ------------------------------------------------------------------ #
#  Full Pipeline (convenience)
# ------------------------------------------------------------------ #
def build_pipeline(data_path: str = None,
                   horizon: int = 1,
                   time_steps: int = None,
                   target_col: str = TARGET):
    """
    Chay toan bo pipeline: load -> engineer -> scale -> window -> split.
    Tra ve dict chua tat ca splits + scaler + metadata.
    """
    ts = time_steps or SEQ_LEN_BY_HORIZON.get(horizon, TIME_STEPS)

    df = load_and_engineer(data_path, target_col=target_col)

    feature_cols = [c for c in df.columns if c != target_col]
    n_train_end  = int(len(df) * TRAIN_RATIO)

    # Fit scalers CHI TREN TRAIN SET
    scaler_X, scaler_y = fit_scalers(
        df.iloc[:n_train_end], feature_cols, target_col
    )

    X_sc, y_sc = transform_data(df, scaler_X, scaler_y, feature_cols, target_col)
    X_win, y_win = make_windows(X_sc, y_sc, time_steps=ts, horizon=horizon)

    splits = chronological_split(X_win, y_win)
    splits.update({
        "scaler_X":     scaler_X,
        "scaler_y":     scaler_y,
        "feature_cols": feature_cols,
        "n_features":   len(feature_cols),
        "time_steps":   ts,
        "horizon":      horizon,
        "df":           df,
        "date_index":   df.index[ts:],
    })
    return splits


if __name__ == "__main__":
    splits = build_pipeline(horizon=1)
    print(f"[data_loader] Pipeline H=1 OK")
    print(f"  Train: {splits['X_train'].shape}")
    print(f"  Val  : {splits['X_val'].shape}")
    print(f"  Test : {splits['X_test'].shape}")
    print(f"  Features: {splits['n_features']}")
