"""
jump_gated_arimax_catboost.py
-----------------------------
Hybrid model for one-step fuel-price forecasting.

Flow:
1. ARIMAX predicts the base price level.
2. CatBoost learns the ARIMAX residual from shock-aware features.
3. A CatBoost classifier estimates whether tomorrow is a large price-jump day.
4. A CatBoost regressor estimates tomorrow's price delta.
5. The final prediction softly blends the base forecast and the delta forecast.

The model only creates features in memory. It does not modify source data files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from statsmodels.tsa.statespace.sarimax import SARIMAX


DEFAULT_EXOG_COLS = ["WTI", "USD_Index", "GPR", "BRT DTD", "Brent_EU_Daily"]
CHAMPION_EXOG_COLS = ["WTI", "USD_Index", "GPR", "BRT DTD", "BRT KH", "Brent_EU_Daily", "NAPHTHA"]
NEWS_SIGNAL_COLS = [
    "all_n",
    "all_sent_mean",
    "all_sent_sum",
    "all_intensity",
    "war_n",
    "war_sent_sum",
    "war_intensity",
    "political_economy_n",
    "political_economy_sent_sum",
    "political_economy_intensity",
    "natural_disaster_n",
    "natural_disaster_sent_sum",
    "natural_disaster_intensity",
]


@dataclass
class JumpGatedConfig:
    target: str = "MG95"
    horizon: int = 1
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    exog_cols: list[str] = field(default_factory=lambda: DEFAULT_EXOG_COLS.copy())
    arimax_order: tuple[int, int, int] = (2, 1, 2)
    oof_folds: int = 5
    oof_min_train_frac: float = 0.55
    jump_thresholds: list[float] = field(default_factory=lambda: [1.5, 2, 2.5, 3, 4, 5])
    soft_gammas: list[float] = field(default_factory=lambda: [0.5, 1, 2])
    hard_cutoffs: list[float] = field(default_factory=lambda: [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6])
    random_seed: int = 42


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float], name: str) -> dict:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(yt, yp)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mape = float(np.mean(np.abs((yt - yp) / (np.abs(yt) + 1e-8))) * 100)
    smape = float(np.mean(2.0 * np.abs(yp - yt) / (np.abs(yt) + np.abs(yp) + 1e-8)) * 100)
    return {
        "Model": name,
        "MAE": round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "MAPE(%)": round(float(mape), 4),
        "SMAPE(%)": round(float(smape), 4),
        "R2": round(float(r2_score(yt, yp)), 4),
    }


def add_news_lag_features(
    df: pd.DataFrame,
    *,
    news_path: str | Path | None,
    lag_days: Iterable[int] = (1, 3, 7, 14),
    rolling_windows: Iterable[int] = (3, 7, 14),
    news_cols: Iterable[str] = NEWS_SIGNAL_COLS,
) -> tuple[pd.DataFrame, list[str]]:
    """Join daily news and create lag/rolling-lag features used by the champion trial."""
    out = df.copy()
    if news_path is None or not Path(news_path).exists():
        return out, []

    news = pd.read_csv(news_path, parse_dates=["date"]).set_index("date")
    news = news[~news.index.duplicated(keep="last")].sort_index()
    news = news.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    base_cols = [c for c in news_cols if c in news.columns]
    if not base_cols:
        return out, []

    engineered = pd.DataFrame(index=news.index)
    for c in base_cols:
        s = pd.to_numeric(news[c], errors="coerce").fillna(0.0)
        for lag in lag_days:
            engineered[f"{c}_lag{lag}"] = s.shift(lag).fillna(0.0)
        for window in rolling_windows:
            col = f"{c}_roll{window}_lag1"
            engineered[col] = s.shift(1).rolling(window, min_periods=1).sum().fillna(0.0)

    missing_cols = [c for c in engineered.columns if c not in out.columns]
    if missing_cols:
        out = out.join(engineered[missing_cols], how="left")
        out[missing_cols] = out[missing_cols].fillna(0.0)
    return out, missing_cols


def load_champion_frame(
    *,
    root: str | Path,
    data_path: str | Path | None = None,
    news_path: str | Path | None = None,
    date_col: str = "Ngày",
    drop_initial_rows: int = 30,
) -> pd.DataFrame:
    """
    Load raw price/exogenous data and add only the news lag/rolling features
    that improved the champion multi-horizon experiment.

    The first rows are dropped to align with the original notebooks after their
    lag/rolling feature engineering.
    """
    root = Path(root)
    data_path = Path(data_path) if data_path is not None else root / "data" / "processed" / "clean_data_exo_ver1.csv"
    news_path = Path(news_path) if news_path is not None else root / "news-crawler" / "data" / "daily_features.csv"

    df = pd.read_csv(data_path)
    df.columns = df.columns.astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    if date_col not in df.columns:
        raise ValueError(f"Missing date column {date_col!r}. Columns: {list(df.columns)}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    df, _ = add_news_lag_features(df, news_path=news_path)
    if drop_initial_rows > 0:
        df = df.iloc[int(drop_initial_rows):].copy()
    return df


def add_shock_features(
    df: pd.DataFrame,
    *,
    target: str = "MG95",
    news_path: str | Path | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Add shock-aware features in memory and optionally join daily news features."""
    out = df.copy()

    doy = out.index.dayofyear.values.astype(float)
    out["DOY_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["DOY_cos"] = np.cos(2 * np.pi * doy / 365.25)

    news_cols: list[str] = []
    if news_path is not None and Path(news_path).exists():
        news = pd.read_csv(news_path, parse_dates=["date"]).set_index("date")
        news = news[~news.index.duplicated(keep="last")].sort_index()
        news_cols = list(news.columns)

        missing_news_cols = [c for c in news_cols if c not in out.columns]
        if missing_news_cols:
            out = out.join(news[missing_news_cols], how="left")
        available_news_cols = [c for c in news_cols if c in out.columns]
        out[available_news_cols] = out[available_news_cols].fillna(0.0)

        for c in [
            "all_n",
            "all_sent_sum",
            "all_intensity",
            "political_economy_n",
            "political_economy_sent_sum",
            "political_economy_intensity",
            "war_n",
            "war_sent_sum",
            "war_intensity",
            "natural_disaster_n",
            "natural_disaster_sent_sum",
            "natural_disaster_intensity",
        ]:
            if c in out.columns:
                for w in [3, 7, 14]:
                    out[f"{c}_roll{w}"] = out[c].rolling(w, min_periods=1).sum()

        out, _ = add_news_lag_features(out, news_path=news_path)

    price_delta = out[target].diff().fillna(0.0)
    changed = price_delta.abs() > 1e-9

    days_since = []
    count = 0
    for is_changed in changed.values:
        count = 0 if is_changed else count + 1
        days_since.append(count)

    out["days_since_price_change"] = days_since
    out["last_price_change"] = price_delta.where(changed).replace(0, np.nan).ffill().fillna(0.0)
    out["last_abs_price_change"] = out["last_price_change"].abs()
    out["price_change_flag"] = changed.astype(int)

    group_id = changed.cumsum()
    for col in ["WTI", "USD_Index", "GPR", "BRT DTD", "BRT KH", "NAPHTHA", "Brent_EU_Daily"]:
        if col not in out.columns:
            continue
        delta = out[col].diff().fillna(0.0)
        out[f"{col}_delta1"] = delta
        out[f"{col}_cum_since_price_change"] = delta.groupby(group_id).cumsum()
        for w in [3, 7, 14, 30]:
            out[f"{col}_delta_sum{w}"] = delta.rolling(w, min_periods=1).sum()
            out[f"{col}_vol{w}"] = delta.rolling(w, min_periods=2).std().fillna(0.0)

    out[f"{target}_delta1"] = price_delta
    for w in [3, 7, 14, 30]:
        out[f"{target}_abs_delta_roll{w}"] = price_delta.abs().rolling(w, min_periods=1).sum()
        out[f"{target}_vol_delta{w}"] = price_delta.rolling(w, min_periods=2).std().fillna(0.0)

    feature_cols = [c for c in out.columns if c != target and not c.startswith("__")]
    out[feature_cols] = out[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return out, news_cols


def make_supervised_frame(df: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    work = df.copy()
    work["__y"] = work[target].shift(-horizon)
    work["__feature_pos"] = np.arange(len(work))
    work["__target_pos"] = work["__feature_pos"] + horizon
    work = work.dropna(subset=["__y"]).copy()
    work["__feature_pos"] = work["__feature_pos"].astype(int)
    work["__target_pos"] = work["__target_pos"].astype(int)
    work["__target_date"] = df.index[work["__target_pos"].values]
    work["__delta_next"] = work["__y"] - work[target]
    return work


def _period_errors(test_dates, actual, pred_arimax, pred_base, pred_jump) -> pd.DataFrame:
    err = pd.DataFrame({
        "date": pd.Index(test_dates),
        "actual": actual,
        "arimax": pred_arimax,
        "shock_aware": pred_base,
        "jump_gated": pred_jump,
    })
    for c in ["arimax", "shock_aware", "jump_gated"]:
        err[f"abs_err_{c}"] = np.abs(err["actual"] - err[c])

    periods = [
        ("Normal before shock", "2024-07-01", "2026-02-28"),
        ("Shock window", "2026-03-01", "2026-05-31"),
        ("Year 2025", "2025-01-01", "2025-12-31"),
        ("Full test", str(err["date"].min().date()), str(err["date"].max().date())),
    ]

    rows = []
    for name, start, end in periods:
        mask = (err["date"] >= pd.Timestamp(start)) & (err["date"] <= pd.Timestamp(end))
        if not mask.any():
            continue
        rows.append({
            "Period": name,
            "Start": start,
            "End": end,
            "N": int(mask.sum()),
            "ARIMAX_MAE": round(float(err.loc[mask, "abs_err_arimax"].mean()), 4),
            "ShockAware_MAE": round(float(err.loc[mask, "abs_err_shock_aware"].mean()), 4),
            "JumpGated_MAE": round(float(err.loc[mask, "abs_err_jump_gated"].mean()), 4),
        })
    return pd.DataFrame(rows)


def run_jump_gated_arimax_catboost(
    df: pd.DataFrame,
    *,
    root: str | Path | None = None,
    news_path: str | Path | None = None,
    config: JumpGatedConfig | None = None,
    progress: bool = True,
) -> dict:
    """Train and evaluate Jump-Gated ARIMAX-CatBoost on the chronological test set."""
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    except ImportError as exc:
        raise ImportError("CatBoost is required. Install with: pip install catboost") from exc

    cfg = config or JumpGatedConfig()
    if cfg.horizon != 1:
        raise NotImplementedError("Jump-Gated ARIMAX-CatBoost currently supports horizon=1 only.")

    if news_path is None and root is not None:
        news_path = Path(root) / "news-crawler" / "data" / "daily_features.csv"

    data, news_cols = add_shock_features(df, target=cfg.target, news_path=news_path)
    feature_cols = [c for c in data.columns if c != cfg.target and not c.startswith("__")]
    work = make_supervised_frame(data, cfg.target, cfg.horizon)

    n = len(work)
    ntr = int(n * cfg.train_ratio)
    nvl = int(n * cfg.val_ratio)
    pretest_end_row = ntr + nvl
    tr = work.iloc[:ntr]
    vl = work.iloc[ntr:pretest_end_row]
    te = work.iloc[pretest_end_row:]

    exog_cols = [c for c in cfg.exog_cols if c in data.columns]
    y_full = data[cfg.target].astype(float).reset_index(drop=True)
    ex_full = data[exog_cols].astype(float).reset_index(drop=True)

    def arimax_one_step_predict(train_end_pos, pred_start_pos, pred_end_pos, maxiter=80):
        model = SARIMAX(
            y_full.iloc[:train_end_pos],
            exog=ex_full.iloc[:train_end_pos],
            order=cfg.arimax_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False, maxiter=maxiter)
        append_y = y_full.iloc[pred_start_pos:pred_end_pos + 1]
        append_ex = ex_full.iloc[pred_start_pos:pred_end_pos + 1]
        res2 = res.append(append_y, exog=append_ex, refit=False)
        pred = res2.predict(start=pred_start_pos, end=pred_end_pos, exog=append_ex).values
        return np.arange(pred_start_pos, pred_end_pos + 1), pred

    def build_oof_residuals():
        min_train_row = int(pretest_end_row * cfg.oof_min_train_frac)
        fold_edges = np.linspace(min_train_row, pretest_end_row, cfg.oof_folds + 1, dtype=int)
        chunks = []

        for fold_id in range(cfg.oof_folds):
            row_start = int(fold_edges[fold_id])
            row_end = int(fold_edges[fold_id + 1])
            fold_rows = work.iloc[row_start:row_end].copy()
            pred_start = int(fold_rows["__target_pos"].iloc[0])
            pred_end = int(fold_rows["__target_pos"].iloc[-1])

            pos, pred = arimax_one_step_predict(pred_start, pred_start, pred_end, maxiter=60)
            actual = y_full.iloc[pos].values
            residual = actual - pred

            out = fold_rows[feature_cols].copy()
            out["__residual"] = residual
            out["__target_pos"] = pos
            chunks.append(out)

            if progress:
                start_date = data.index[pred_start].date()
                end_date = data.index[pred_end].date()
                mae = np.mean(np.abs(residual))
                print(f"OOF fold {fold_id + 1}: {start_date} -> {end_date} | MAE={mae:.4f}")

        return pd.concat(chunks, axis=0).sort_index()

    test_start = int(te["__target_pos"].iloc[0])
    test_end = int(te["__target_pos"].iloc[-1])
    test_pos, pred_arimax = arimax_one_step_predict(test_start, test_start, test_end)
    y_test = y_full.iloc[test_pos].values

    val_start = int(vl["__target_pos"].iloc[0])
    val_end = int(vl["__target_pos"].iloc[-1])
    val_pos, pred_arimax_val = arimax_one_step_predict(val_start, val_start, val_end, maxiter=60)
    y_val = y_full.iloc[val_pos].values

    oof = build_oof_residuals()
    oof_cut = int(len(oof) * 0.80)
    oof_train = oof.iloc[:oof_cut]
    oof_valid = oof.iloc[oof_cut:]

    res_model = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=900,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=8,
        random_seed=cfg.random_seed,
        od_type="Iter",
        od_wait=80,
        allow_writing_files=False,
        verbose=False,
    )
    res_model.fit(
        Pool(oof_train[feature_cols], oof_train["__residual"].values),
        eval_set=Pool(oof_valid[feature_cols], oof_valid["__residual"].values),
        use_best_model=True,
    )

    pred_base_test = pred_arimax + res_model.predict(te[feature_cols])
    pred_base_val = pred_arimax_val + res_model.predict(vl[feature_cols])

    xtr = tr[feature_cols]
    xvl = vl[feature_cols]
    xte = te[feature_cols]

    variant_rows = []
    variant_payload = {}

    for threshold in cfg.jump_thresholds:
        y_jump_tr = (tr["__delta_next"].abs().values >= threshold).astype(int)
        y_jump_vl = (vl["__delta_next"].abs().values >= threshold).astype(int)
        y_jump_te = (te["__delta_next"].abs().values >= threshold).astype(int)

        if y_jump_tr.sum() < 10 or y_jump_vl.sum() < 3:
            continue

        pos_weight = float((len(y_jump_tr) - y_jump_tr.sum()) / (y_jump_tr.sum() + 1e-8))
        class_weights = {0: 1.0, 1: pos_weight}

        jump_clf = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=600,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=8,
            random_seed=cfg.random_seed,
            od_type="Iter",
            od_wait=60,
            allow_writing_files=False,
            verbose=False,
            class_weights=class_weights,
        )
        jump_clf.fit(Pool(xtr, y_jump_tr), eval_set=Pool(xvl, y_jump_vl), use_best_model=True)

        p_jump_val = jump_clf.predict_proba(xvl)[:, 1]
        p_jump_test = jump_clf.predict_proba(xte)[:, 1]
        auc_val = roc_auc_score(y_jump_vl, p_jump_val) if len(np.unique(y_jump_vl)) > 1 else np.nan

        abs_delta_tr = np.abs(tr["__delta_next"].values)
        median_abs_delta = np.median(abs_delta_tr) + 1e-8
        delta_weights = 1.0 + 3.0 * np.minimum(abs_delta_tr / median_abs_delta, 10.0)

        delta_model = CatBoostRegressor(
            loss_function="MAE",
            eval_metric="MAE",
            iterations=1000,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=8,
            random_seed=cfg.random_seed,
            od_type="Iter",
            od_wait=80,
            allow_writing_files=False,
            verbose=False,
        )
        delta_model.fit(
            Pool(xtr, tr["__delta_next"].values, weight=delta_weights),
            eval_set=Pool(xvl, vl["__delta_next"].values),
            use_best_model=True,
        )

        pred_delta_val = delta_model.predict(xvl)
        pred_delta_test = delta_model.predict(xte)
        pred_delta_price_val = vl[cfg.target].values + pred_delta_val
        pred_delta_price_test = te[cfg.target].values + pred_delta_test

        for gamma in cfg.soft_gammas:
            w_val = np.clip(p_jump_val, 0, 1) ** gamma
            w_test = np.clip(p_jump_test, 0, 1) ** gamma
            pred_val = (1 - w_val) * pred_base_val + w_val * pred_delta_price_val
            pred_test = (1 - w_test) * pred_base_test + w_test * pred_delta_price_test

            row = regression_metrics(y_test, pred_test, f"Jump-Gated soft thr={threshold} gamma={gamma}")
            row.update({
                "threshold": threshold,
                "gate": "soft",
                "gate_param": gamma,
                "valid_MAE": round(float(mean_absolute_error(y_val, pred_val)), 4),
                "valid_AUC": round(float(auc_val), 4) if not np.isnan(auc_val) else np.nan,
                "test_mean_gate": round(float(np.mean(w_test)), 4),
                "test_true_jump_rate": round(float(np.mean(y_jump_te)), 4),
            })
            variant_rows.append(row)
            variant_payload[(float(threshold), "soft", float(gamma))] = pred_test

        for cutoff in cfg.hard_cutoffs:
            pred_val = np.where(p_jump_val >= cutoff, pred_delta_price_val, pred_base_val)
            pred_test = np.where(p_jump_test >= cutoff, pred_delta_price_test, pred_base_test)

            row = regression_metrics(y_test, pred_test, f"Jump-Gated hard thr={threshold} cutoff={cutoff}")
            row.update({
                "threshold": threshold,
                "gate": "hard",
                "gate_param": cutoff,
                "valid_MAE": round(float(mean_absolute_error(y_val, pred_val)), 4),
                "valid_AUC": round(float(auc_val), 4) if not np.isnan(auc_val) else np.nan,
                "test_mean_gate": round(float(np.mean(p_jump_test >= cutoff)), 4),
                "test_true_jump_rate": round(float(np.mean(y_jump_te)), 4),
            })
            variant_rows.append(row)
            variant_payload[(float(threshold), "hard", float(cutoff))] = pred_test

    variants = pd.DataFrame(variant_rows)
    if variants.empty:
        raise RuntimeError("No Jump-Gated variant could be trained. Check jump thresholds and validation data.")

    variants = variants.sort_values(["valid_MAE", "MAE"]).reset_index(drop=True)
    best = variants.iloc[0]
    best_key = (float(best["threshold"]), str(best["gate"]), float(best["gate_param"]))
    pred_jump = variant_payload[best_key]

    rows = [
        regression_metrics(y_test, pred_arimax, "ARIMAX"),
        regression_metrics(y_test, pred_base_test, "Shock-Aware ARIMAX-CatBoost"),
        regression_metrics(y_test, pred_jump, "Jump-Gated ARIMAX-CatBoost"),
    ]
    metrics = pd.DataFrame(rows)
    base_mae = float(metrics.loc[metrics["Model"] == "ARIMAX", "MAE"].iloc[0])
    metrics["MAE_vs_ARIMAX(%)"] = ((metrics["MAE"].astype(float) - base_mae) / base_mae * 100).round(2)
    metrics = metrics.sort_values("MAE").reset_index(drop=True)

    test_dates = pd.Index(te["__target_date"])
    predictions = pd.DataFrame({
        "date": test_dates,
        "actual": y_test,
        "arimax": pred_arimax,
        "shock_aware": pred_base_test,
        "jump_gated": pred_jump,
    })
    period_errors = _period_errors(test_dates, y_test, pred_arimax, pred_base_test, pred_jump)

    if progress:
        best_mae = metrics.loc[metrics["Model"] == "Jump-Gated ARIMAX-CatBoost", "MAE"].iloc[0]
        print(f"Jump-Gated ARIMAX-CatBoost MAE={best_mae:.4f}")

    return {
        "metrics": metrics,
        "variants": variants,
        "best_variant": best.to_dict(),
        "period_errors": period_errors,
        "predictions": predictions,
        "feature_cols": feature_cols,
        "news_cols": news_cols,
        "test_dates": test_dates,
        "y_test": y_test,
        "pred_jump_gated": pred_jump,
    }


def _rolling_arimax_forecast(
    y_full: pd.Series,
    ex_full: pd.DataFrame | None,
    *,
    first_origin: int,
    last_origin: int,
    horizon: int,
    order: tuple[int, int, int],
    maxiter: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast y(origin + horizon) for each origin in a contiguous range."""
    train_end = first_origin + 1
    model = SARIMAX(
        y_full.iloc[:train_end],
        exog=None if ex_full is None else ex_full.iloc[:train_end],
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    cur = model.fit(disp=False, maxiter=maxiter)

    positions = []
    preds = []
    for origin in range(first_origin, last_origin + 1):
        target_pos = origin + horizon
        if target_pos >= len(y_full):
            break

        if ex_full is None:
            fc = cur.forecast(steps=horizon)
        else:
            future_ex = ex_full.iloc[origin + 1:origin + horizon + 1]
            fc = cur.forecast(steps=horizon, exog=future_ex)

        positions.append(target_pos)
        preds.append(float(np.asarray(fc)[-1]))

        next_pos = origin + 1
        if next_pos < len(y_full):
            if ex_full is None:
                cur = cur.extend(y_full.iloc[next_pos:next_pos + 1])
            else:
                cur = cur.extend(
                    y_full.iloc[next_pos:next_pos + 1],
                    exog=ex_full.iloc[next_pos:next_pos + 1],
                )

    return np.asarray(positions, dtype=int), np.asarray(preds, dtype=float)


def _candidate_jump_thresholds(delta: pd.Series, cfg: JumpGatedConfig) -> list[float]:
    """Build a small threshold set that adapts to each horizon's delta scale."""
    abs_delta = np.asarray(delta.abs(), dtype=float)
    qs = np.quantile(abs_delta, [0.55, 0.70, 0.85])
    raw = list(cfg.jump_thresholds) + [float(x) for x in qs]
    thresholds = sorted({round(float(x), 4) for x in raw if np.isfinite(x) and x > 0})
    if len(thresholds) <= 6:
        return thresholds
    keep_idx = np.linspace(0, len(thresholds) - 1, 6, dtype=int)
    return [thresholds[i] for i in keep_idx]


def run_jump_gated_arimax_catboost_horizon(
    df: pd.DataFrame,
    *,
    horizon: int,
    root: str | Path | None = None,
    news_path: str | Path | None = None,
    config: JumpGatedConfig | None = None,
    exact_h1: bool = True,
    progress: bool = True,
) -> dict:
    """
    Train and evaluate the same ARIMAX-CatBoost framework for any horizon.

    For H=1, exact_h1=True reuses the original one-step implementation so the
    notebook final score remains identical. For H>1, ARIMAX forecasts H steps
    ahead from each origin, CatBoost learns residuals, and the gate detects
    large H-day moves.
    """
    if exact_h1 and horizon == 1:
        cfg = config or JumpGatedConfig()
        h1_cfg = JumpGatedConfig(**{**cfg.__dict__, "horizon": 1})
        result = run_jump_gated_arimax_catboost(
            df,
            root=root,
            news_path=news_path,
            config=h1_cfg,
            progress=progress,
        )
        result["metrics"]["Horizon"] = 1
        cols = ["Model", "Horizon", "MAE", "RMSE", "MAPE(%)", "SMAPE(%)", "R2"]
        result["metrics"] = result["metrics"][cols + [c for c in result["metrics"].columns if c not in cols]]
        return result

    try:
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    except ImportError as exc:
        raise ImportError("CatBoost is required. Install with: pip install catboost") from exc

    cfg = config or JumpGatedConfig()
    cfg = JumpGatedConfig(**{**cfg.__dict__, "horizon": horizon})

    if news_path is None and root is not None:
        news_path = Path(root) / "news-crawler" / "data" / "daily_features.csv"

    data, news_cols = add_shock_features(df, target=cfg.target, news_path=news_path)
    feature_cols = [c for c in data.columns if c != cfg.target and not c.startswith("__")]
    work = make_supervised_frame(data, cfg.target, horizon)

    n = len(work)
    ntr = int(n * cfg.train_ratio)
    nvl = int(n * cfg.val_ratio)
    pretest_end_row = ntr + nvl
    tr = work.iloc[:ntr]
    vl = work.iloc[ntr:pretest_end_row]
    te = work.iloc[pretest_end_row:]

    exog_cols = [c for c in cfg.exog_cols if c in data.columns]
    y_full = data[cfg.target].astype(float).reset_index(drop=True)
    ex_full = data[exog_cols].astype(float).reset_index(drop=True) if exog_cols else None

    def rolling_for_rows(rows: pd.DataFrame, maxiter: int = 60):
        first_origin = int(rows["__feature_pos"].iloc[0])
        last_origin = int(rows["__feature_pos"].iloc[-1])
        return _rolling_arimax_forecast(
            y_full,
            ex_full,
            first_origin=first_origin,
            last_origin=last_origin,
            horizon=horizon,
            order=cfg.arimax_order,
            maxiter=maxiter,
        )

    def build_oof_residuals():
        min_train_row = int(pretest_end_row * cfg.oof_min_train_frac)
        fold_edges = np.linspace(min_train_row, pretest_end_row, cfg.oof_folds + 1, dtype=int)
        chunks = []

        for fold_id in range(cfg.oof_folds):
            row_start = int(fold_edges[fold_id])
            row_end = int(fold_edges[fold_id + 1])
            fold_rows = work.iloc[row_start:row_end].copy()
            pos, pred = rolling_for_rows(fold_rows, maxiter=50)
            actual = y_full.iloc[pos].values

            out = fold_rows.iloc[:len(pos)][feature_cols].copy()
            out["__residual"] = actual - pred
            chunks.append(out)

            if progress:
                start_date = data.index[pos[0]].date()
                end_date = data.index[pos[-1]].date()
                mae = np.mean(np.abs(out["__residual"].values))
                print(f"H={horizon} OOF fold {fold_id + 1}: {start_date} -> {end_date} | MAE={mae:.4f}")

        return pd.concat(chunks, axis=0).sort_index()

    test_pos, pred_arimax = rolling_for_rows(te)
    y_test = y_full.iloc[test_pos].values
    val_pos, pred_arimax_val = rolling_for_rows(vl, maxiter=50)
    y_val = y_full.iloc[val_pos].values

    oof = build_oof_residuals()
    oof_cut = int(len(oof) * 0.80)
    oof_train = oof.iloc[:oof_cut]
    oof_valid = oof.iloc[oof_cut:]

    res_model = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=700,
        depth=4,
        learning_rate=0.035,
        l2_leaf_reg=8,
        random_seed=cfg.random_seed,
        od_type="Iter",
        od_wait=60,
        allow_writing_files=False,
        verbose=False,
    )
    res_model.fit(
        Pool(oof_train[feature_cols], oof_train["__residual"].values),
        eval_set=Pool(oof_valid[feature_cols], oof_valid["__residual"].values),
        use_best_model=True,
    )

    pred_base_test = pred_arimax + res_model.predict(te[feature_cols].iloc[:len(test_pos)])
    pred_base_val = pred_arimax_val + res_model.predict(vl[feature_cols].iloc[:len(val_pos)])

    xtr = tr[feature_cols]
    xvl = vl[feature_cols].iloc[:len(val_pos)]
    xte = te[feature_cols].iloc[:len(test_pos)]
    vl_eval = vl.iloc[:len(val_pos)]
    te_eval = te.iloc[:len(test_pos)]

    variant_rows = []
    variant_payload = {}
    thresholds = _candidate_jump_thresholds(tr["__delta_next"], cfg)

    for threshold in thresholds:
        y_jump_tr = (tr["__delta_next"].abs().values >= threshold).astype(int)
        y_jump_vl = (vl_eval["__delta_next"].abs().values >= threshold).astype(int)
        y_jump_te = (te_eval["__delta_next"].abs().values >= threshold).astype(int)

        if y_jump_tr.sum() < 10 or len(np.unique(y_jump_tr)) < 2:
            continue

        pos_weight = float((len(y_jump_tr) - y_jump_tr.sum()) / (y_jump_tr.sum() + 1e-8))
        class_weights = {0: 1.0, 1: pos_weight}

        jump_clf = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            iterations=400,
            depth=4,
            learning_rate=0.04,
            l2_leaf_reg=8,
            random_seed=cfg.random_seed,
            od_type="Iter",
            od_wait=50,
            allow_writing_files=False,
            verbose=False,
            class_weights=class_weights,
        )
        jump_clf.fit(Pool(xtr, y_jump_tr), eval_set=Pool(xvl, y_jump_vl), use_best_model=True)

        p_jump_val = jump_clf.predict_proba(xvl)[:, 1]
        p_jump_test = jump_clf.predict_proba(xte)[:, 1]
        auc_val = roc_auc_score(y_jump_vl, p_jump_val) if len(np.unique(y_jump_vl)) > 1 else np.nan

        abs_delta_tr = np.abs(tr["__delta_next"].values)
        median_abs_delta = np.median(abs_delta_tr) + 1e-8
        delta_weights = 1.0 + 3.0 * np.minimum(abs_delta_tr / median_abs_delta, 10.0)

        delta_model = CatBoostRegressor(
            loss_function="MAE",
            eval_metric="MAE",
            iterations=700,
            depth=4,
            learning_rate=0.035,
            l2_leaf_reg=8,
            random_seed=cfg.random_seed,
            od_type="Iter",
            od_wait=60,
            allow_writing_files=False,
            verbose=False,
        )
        delta_model.fit(
            Pool(xtr, tr["__delta_next"].values, weight=delta_weights),
            eval_set=Pool(xvl, vl_eval["__delta_next"].values),
            use_best_model=True,
        )

        pred_delta_val = delta_model.predict(xvl)
        pred_delta_test = delta_model.predict(xte)
        pred_delta_price_val = vl_eval[cfg.target].values + pred_delta_val
        pred_delta_price_test = te_eval[cfg.target].values + pred_delta_test

        for gamma in cfg.soft_gammas:
            w_val = np.clip(p_jump_val, 0, 1) ** gamma
            w_test = np.clip(p_jump_test, 0, 1) ** gamma
            pred_val = (1 - w_val) * pred_base_val + w_val * pred_delta_price_val
            pred_test = (1 - w_test) * pred_base_test + w_test * pred_delta_price_test

            row = regression_metrics(y_test, pred_test, f"Jump-Gated soft thr={threshold} gamma={gamma}")
            row.update({
                "Horizon": horizon,
                "threshold": threshold,
                "gate": "soft",
                "gate_param": gamma,
                "valid_MAE": round(float(mean_absolute_error(y_val, pred_val)), 4),
                "valid_AUC": round(float(auc_val), 4) if not np.isnan(auc_val) else np.nan,
                "test_mean_gate": round(float(np.mean(w_test)), 4),
                "test_true_jump_rate": round(float(np.mean(y_jump_te)), 4),
            })
            variant_rows.append(row)
            variant_payload[(float(threshold), "soft", float(gamma))] = pred_test

        for cutoff in cfg.hard_cutoffs:
            pred_val = np.where(p_jump_val >= cutoff, pred_delta_price_val, pred_base_val)
            pred_test = np.where(p_jump_test >= cutoff, pred_delta_price_test, pred_base_test)

            row = regression_metrics(y_test, pred_test, f"Jump-Gated hard thr={threshold} cutoff={cutoff}")
            row.update({
                "Horizon": horizon,
                "threshold": threshold,
                "gate": "hard",
                "gate_param": cutoff,
                "valid_MAE": round(float(mean_absolute_error(y_val, pred_val)), 4),
                "valid_AUC": round(float(auc_val), 4) if not np.isnan(auc_val) else np.nan,
                "test_mean_gate": round(float(np.mean(p_jump_test >= cutoff)), 4),
                "test_true_jump_rate": round(float(np.mean(y_jump_te)), 4),
            })
            variant_rows.append(row)
            variant_payload[(float(threshold), "hard", float(cutoff))] = pred_test

    fallback_rows = [
        {
            "Model": "ARIMAX fallback",
            "Horizon": horizon,
            "threshold": np.nan,
            "gate": "arimax",
            "gate_param": np.nan,
            "valid_MAE": round(float(mean_absolute_error(y_val, pred_arimax_val)), 4),
            "valid_AUC": np.nan,
            "test_mean_gate": 0.0,
            "test_true_jump_rate": np.nan,
        },
        {
            "Model": "Shock-Aware fallback",
            "Horizon": horizon,
            "threshold": np.nan,
            "gate": "shock_aware",
            "gate_param": np.nan,
            "valid_MAE": round(float(mean_absolute_error(y_val, pred_base_val)), 4),
            "valid_AUC": np.nan,
            "test_mean_gate": 0.0,
            "test_true_jump_rate": np.nan,
        },
    ]

    variants = pd.concat([pd.DataFrame(fallback_rows), pd.DataFrame(variant_rows)], ignore_index=True)
    variants = variants.sort_values(["valid_MAE"], na_position="last").reset_index(drop=True)
    best = variants.iloc[0].to_dict()

    if best["gate"] == "arimax":
        pred_jump = pred_arimax
    elif best["gate"] == "shock_aware":
        pred_jump = pred_base_test
    else:
        best_key = (float(best["threshold"]), str(best["gate"]), float(best["gate_param"]))
        pred_jump = variant_payload[best_key]

    rows = [
        regression_metrics(y_test, pred_arimax, "ARIMAX"),
        regression_metrics(y_test, pred_base_test, "Shock-Aware ARIMAX-CatBoost"),
        regression_metrics(y_test, pred_jump, "Jump-Gated ARIMAX-CatBoost"),
    ]
    metrics = pd.DataFrame(rows)
    metrics["Horizon"] = horizon
    cols = ["Model", "Horizon", "MAE", "RMSE", "MAPE(%)", "SMAPE(%)", "R2"]
    metrics = metrics[cols]

    test_dates = pd.Index(data.index[test_pos])
    predictions = pd.DataFrame({
        "date": test_dates,
        "actual": y_test,
        "arimax": pred_arimax,
        "shock_aware": pred_base_test,
        "jump_gated": pred_jump,
    })

    if progress:
        row = metrics.loc[metrics["Model"] == "Jump-Gated ARIMAX-CatBoost"].iloc[0]
        print(f"H={horizon} Jump-Gated ARIMAX-CatBoost MAPE={row['MAPE(%)']:.4f}% | MAE={row['MAE']:.4f}")

    return {
        "metrics": metrics,
        "variants": variants,
        "best_variant": best,
        "predictions": predictions,
        "feature_cols": feature_cols,
        "news_cols": news_cols,
        "test_dates": test_dates,
        "y_test": y_test,
        "pred_jump_gated": pred_jump,
    }


def run_multihorizon_jump_gated_arimax_catboost(
    df: pd.DataFrame,
    *,
    horizons: Iterable[int] = (1, 5, 10, 30, 60),
    root: str | Path | None = None,
    news_path: str | Path | None = None,
    config: JumpGatedConfig | None = None,
    exact_h1: bool = True,
    progress: bool = True,
) -> dict:
    """Run Jump-Gated ARIMAX-CatBoost for multiple horizons and combine results."""
    metrics_rows = []
    variant_rows = []
    details = {}

    for horizon in horizons:
        if progress:
            print("=" * 70)
            print(f"Jump-Gated ARIMAX-CatBoost | H = {horizon}")
            print("=" * 70)

        result = run_jump_gated_arimax_catboost_horizon(
            df,
            horizon=int(horizon),
            root=root,
            news_path=news_path,
            config=config,
            exact_h1=exact_h1,
            progress=progress,
        )
        details[int(horizon)] = result

        jump_metric = result["metrics"].loc[
            result["metrics"]["Model"] == "Jump-Gated ARIMAX-CatBoost"
        ].copy()
        metrics_rows.append(jump_metric)

        variants = result["variants"].copy()
        if "Horizon" not in variants.columns:
            variants["Horizon"] = int(horizon)
        variant_rows.append(variants)

    metrics = pd.concat(metrics_rows, ignore_index=True)
    variants = pd.concat(variant_rows, ignore_index=True)
    return {"metrics": metrics, "variants": variants, "details": details}
