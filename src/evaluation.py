"""
evaluation.py
-------------
Walk-Forward Validation, tinh chi so loi, Multi-Horizon evaluation,
va tat ca cac ham ve bieu do ket qua.

Triet ly (theo GUMNet-WF):
- Walk-Forward Validation mo phong chinh xac moi truong giao dich thuc
  te: mo hinh chi duoc biet qua khu, du bao tuong lai, roi cap nhat.
- Multi-Horizon (H=1,5,10): dinh luong ro rang su phan ra tin hieu
  khi du bao cang xa.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CHARTS_DIR = "results/charts"


# ------------------------------------------------------------------ #
#  Metric helpers
# ------------------------------------------------------------------ #
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    name: str = "Model") -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    # SMAPE: symmetric MAPE, bounded [0,200], robust khi y_true gan 0
    smape = np.mean(2.0 * np.abs(y_pred - y_true) /
                    (np.abs(y_true) + np.abs(y_pred) + 1e-8)) * 100
    r2   = r2_score(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    return {
        "Model":    name,
        "MAE":      round(mae,   4),
        "RMSE":     round(rmse,  4),
        "MSE":      round(mse,   4),
        "MAPE(%)":  round(mape,  4),
        "SMAPE(%)": round(smape, 4),
        "R2":       round(r2,    4),
    }


def print_metrics(m: dict):
    print(f"\n{'='*52}")
    print(f"  {m['Model']}")
    print(f"{'='*52}")
    print(f"  MAE     : {m['MAE']:.4f}  nghin dong/lit")
    print(f"  RMSE    : {m['RMSE']:.4f}")
    print(f"  MSE     : {m['MSE']:.4f}")
    print(f"  MAPE    : {m['MAPE(%)']:.4f} %")
    print(f"  SMAPE   : {m['SMAPE(%)']:.4f} %")
    print(f"  R2      : {m['R2']:.4f}")


# ------------------------------------------------------------------ #
#  Walk-Forward Validation
# ------------------------------------------------------------------ #
def walk_forward_lgbm(X_win: np.ndarray, y_win: np.ndarray,
                       scaler_y, folds: list,
                       best_params: dict,
                       date_index: pd.DatetimeIndex) -> dict:
    """
    Walk-Forward Validation cho LightGBM (Expanding Window).

    Moi fold:
      - Train tren [0 : fold.train_end]
      - Evaluate tren [fold.val_start : fold.val_end]
      => Mo phong: mo hinh hoc tu lich su, du bao giai doan tiep theo.
    """
    import lightgbm as lgb
    from .models.baseline_lgbm import train_lgbm

    fold_results = []
    for fold in folds:
        f      = fold["fold"]
        te     = fold["train_end"]
        vs, ve = fold["val_start"], fold["val_end"]

        Xtr = X_win[:te].reshape(te, -1)
        ytr = y_win[:te]
        Xvl = X_win[vs:ve].reshape(ve - vs, -1)
        yvl = y_win[vs:ve]

        model = train_lgbm(Xtr, ytr, Xvl, yvl, best_params)
        pred  = model.predict(Xvl)

        yt_inv = scaler_y.inverse_transform(yvl.reshape(-1,1)).flatten()
        yp_inv = scaler_y.inverse_transform(pred.reshape(-1,1)).flatten()
        m      = compute_metrics(yt_inv, yp_inv,
                                  name=f"Fold {f} (LightGBM WF)")

        fold_dates = date_index[vs:ve]
        fold_results.append({
            **m,
            "fold":       f,
            "train_size": te,
            "val_start":  fold_dates[0].strftime("%Y-%m"),
            "val_end":    fold_dates[-1].strftime("%Y-%m"),
            "y_true":     yt_inv,
            "y_pred":     yp_inv,
            "dates":      fold_dates,
        })

        print(f"  Fold {f:2d} | train={te:4d} | "
              f"{fold_dates[0].date()} -> {fold_dates[-1].date()} "
              f"| MAE={m['MAE']:.3f} | MAPE={m['MAPE(%)']:.2f}% | R2={m['R2']:.4f}")

    df_wf = pd.DataFrame([{k: v for k, v in r.items()
                            if k not in ["y_true","y_pred","dates"]}
                           for r in fold_results])
    print(f"\n  [Walk-Forward Summary]")
    print(f"  MAE  mean±std : {df_wf['MAE'].mean():.4f} ± {df_wf['MAE'].std():.4f}")
    print(f"  MAPE mean±std : {df_wf['MAPE(%)'].mean():.4f} ± {df_wf['MAPE(%)'].std():.4f} %")
    print(f"  R2   mean±std : {df_wf['R2'].mean():.4f} ± {df_wf['R2'].std():.4f}")
    return {"fold_results": fold_results, "summary": df_wf}


# ------------------------------------------------------------------ #
#  Multi-Horizon Evaluation
# ------------------------------------------------------------------ #
def evaluate_multihorizon(model_fn,        # callable(X_test) -> y_pred (scaled)
                           horizon_data: dict,  # {H: (X_win, y_win)}
                           scaler_y,
                           n_train: int,
                           n_val: int,
                           model_name: str = "Model") -> pd.DataFrame:
    """
    Danh gia mo hinh tren nhieu horizons [H=1, H=5, H=10].
    Dinh luong su phan ra tin hieu: do chinh xac giam dan khi H tang.
    """
    rows = []
    for h, (Xw, yw) in sorted(horizon_data.items()):
        n   = len(Xw)
        te  = min(int(n * 0.8), n - 1)
        nv  = min(int(n * 0.10), n - te - 1)
        Xte = Xw[te + nv:]
        yte = yw[te + nv:]

        pred_sc = model_fn(Xte)
        yt_inv  = scaler_y.inverse_transform(yte.reshape(-1,1)).flatten()
        yp_inv  = scaler_y.inverse_transform(
                      np.array(pred_sc).reshape(-1,1)).flatten()

        m = compute_metrics(yt_inv, yp_inv, name=f"H={h}")
        m["Horizon"] = h
        rows.append(m)
        print(f"  H={h:2d} | MAE={m['MAE']:.4f} | MAPE={m['MAPE(%)']:.4f}% | R2={m['R2']:.4f}")

    df = pd.DataFrame(rows).set_index("Horizon")
    return df


# ------------------------------------------------------------------ #
#  Charts
# ------------------------------------------------------------------ #
def plot_actual_vs_predicted(dates, y_true, y_pred,
                              model_name: str,
                              metrics: dict,
                              save_path: str = None,
                              target: str = "MG95"):
    fig, axes = plt.subplots(2, 1, figsize=(18, 11))

    axes[0].plot(dates, y_true, color="#2C3E50", lw=1.8,
                 label="Actual (Thuc te)", alpha=0.9)
    axes[0].plot(dates, y_pred, color="#E74C3C", lw=1.2,
                 linestyle="--", label="Predicted (Du bao)", alpha=0.85)
    axes[0].fill_between(dates, y_true, y_pred,
                         alpha=0.10, color="#E74C3C")
    axes[0].set_title(
        f"Gia {target} - Thuc te vs Du bao | {model_name}\n"
        f"MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f} | "
        f"MAPE={metrics['MAPE(%)']:.4f}% | R2={metrics['R2']:.4f}",
        fontsize=13, fontweight="bold")
    axes[0].set_ylabel(f"Gia {target} (nghin dong/lit)", fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=30)

    resid = y_true - y_pred
    std_r = resid.std()
    axes[1].bar(range(len(resid)), resid,
                color=["#E74C3C" if r > 0 else "#3498DB" for r in resid],
                alpha=0.55, width=1.0)
    axes[1].axhline(0, color="black", lw=1.0)
    axes[1].axhline( std_r, color="orange", ls="--", lw=1.5,
                     label=f"+1σ = {std_r:.3f}")
    axes[1].axhline(-std_r, color="orange", ls="--", lw=1.5,
                     label=f"-1σ = {-std_r:.3f}")
    pct = np.mean(np.abs(resid) < std_r) * 100
    axes[1].text(0.98, 0.06, f"{pct:.1f}% nam trong ±1σ",
                 transform=axes[1].transAxes, ha="right", fontsize=10,
                 bbox=dict(boxstyle="round", fc="white", ec="orange"))
    axes[1].set_title("Phan tich Sai so (Residuals)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Sai so (nghin dong/lit)")
    axes[1].set_xlabel("Mau test (thu tu thoi gian)")
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    path = save_path or f"{CHARTS_DIR}/actual_vs_predicted_{model_name.replace(' ','_')}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Chart] Da luu: {path}")


def plot_multihorizon_bar(df_mh: pd.DataFrame,
                           model_name: str,
                           save_path: str = None,
                           target: str = "MG95"):
    """
    Bar chart the hien su phan ra tin hieu khi H tang.
    Chuon chuan danh gia theo GUMNet-WF.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["MAE", "MAPE(%)", "R2"]
    colors  = {1: "#27AE60", 5: "#F39C12", 10: "#E74C3C"}

    for ax, met in zip(axes, metrics):
        vals = df_mh[met].values
        bars = ax.bar(
            [f"H={h}" for h in df_mh.index],
            vals,
            color=[colors.get(h, "#95A5A6") for h in df_mh.index],
            edgecolor="white", lw=1.5,
        )
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(vals)*0.01,
                    f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_title(met, fontsize=12, fontweight="bold")
        ax.set_xlabel("Horizon (ngay)")
        ax.set_ylabel(met)

    plt.suptitle(
        f"Su phan ra tin hieu theo Horizon - {target} - {model_name}\n"
        f"(Do chinh xac giam dan khi du bao cang xa)",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = save_path or f"{CHARTS_DIR}/multihorizon_{model_name.replace(' ','_')}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Chart] Da luu: {path}")


def plot_walkforward(wf_results: list, df_summary: pd.DataFrame,
                     save_path: str = None,
                     target: str = "MG95"):
    """
    Ve bieu do Walk-Forward: du bao tren tung fold + MAE/R2 theo fold.
    """
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(18, 11))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])
    cmap = plt.cm.tab10(np.linspace(0, 0.9, len(wf_results)))
    for i, r in enumerate(wf_results):
        ax1.plot(r["dates"], r["y_true"], color="#2C3E50", lw=1.0, alpha=0.4,
                 label="Actual" if i == 0 else "_")
        ax1.plot(r["dates"], r["y_pred"], color=cmap[i], lw=1.5, ls="--",
                 label=f"Fold {r['fold']} (MAPE={r['MAPE(%)']:.2f}%)")
    ax1.set_title("Walk-Forward Validation - Du bao tung Fold (Expanding Window)",
                  fontsize=13, fontweight="bold")
    ax1.set_ylabel(f"{target} (nghin dong/lit)")
    ax1.legend(fontsize=8, ncol=4, loc="upper left")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(df_summary["fold"], df_summary["MAE"],
            color=cmap[:len(df_summary)], edgecolor="white")
    ax2.axhline(df_summary["MAE"].mean(), color="red", ls="--", lw=2,
                label=f"Mean={df_summary['MAE'].mean():.3f}")
    ax2.set_title("MAE theo Fold", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Fold #"); ax2.set_ylabel("MAE"); ax2.legend(fontsize=9)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df_summary["fold"], df_summary["R2"], "o-",
             color="#8E44AD", lw=2, ms=8)
    ax3.axhline(df_summary["R2"].mean(), color="#8E44AD", ls="--", lw=2,
                label=f"Mean R2={df_summary['R2'].mean():.4f}")
    ax3.set_ylim(0, 1.05)
    ax3.set_title("R2 theo Fold", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Fold #"); ax3.set_ylabel("R2"); ax3.legend(fontsize=9)

    plt.suptitle(f"Walk-Forward Validation - {target} - LightGBM (Optuna Tuned)",
                 fontsize=15, fontweight="bold")
    path = save_path or f"{CHARTS_DIR}/walk_forward.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Chart] Da luu: {path}")


def plot_model_comparison_bar(results: list, save_path: str = None,
                              target: str = "MG95"):
    """Bar chart so sanh MAE / RMSE / MAPE / R2 giua cac mo hinh."""
    df = pd.DataFrame([{k: v for k, v in r.items()
                         if k not in ["y_true", "y_pred"]}
                        for r in results]).set_index("Model")
    metrics = ["MAE", "RMSE", "MAPE(%)", "R2"]
    colors  = ["#27AE60", "#3498DB", "#8E44AD"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    for ax, met in zip(axes, metrics):
        vals = df[met].values
        bars = ax.bar(df.index, vals,
                      color=colors[:len(df)], edgecolor="white", lw=1.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(abs(vals))*0.01,
                    f"{v:.4f}", ha="center", fontsize=9, fontweight="bold")
        ax.set_title(met, fontsize=12, fontweight="bold")
        ax.set_xticklabels(df.index, rotation=15, ha="right", fontsize=9)
        if met == "R2":
            ax.set_ylim(0, 1.15)

    plt.suptitle(f"So sanh hieu nang cac mo hinh tren tap Test - {target}",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_path or f"{CHARTS_DIR}/model_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [Chart] Da luu: {path}")
