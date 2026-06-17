"""
baseline_lgbm.py
----------------
LightGBM Regression voi Optuna Hyperparameter Tuning.

Triet ly: LightGBM + lag features = baseline rat manh cho time series
ngan han. Optuna (TPE Sampler) tim bo tham so toi uu hon Grid Search.
"""

import numpy as np
import optuna
import lightgbm as lgb
from sklearn.metrics import mean_squared_error

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ------------------------------------------------------------------ #
#  Optuna Objective
# ------------------------------------------------------------------ #
def _lgbm_objective(trial, X_tr, y_tr, X_vl, y_vl):
    params = {
        "n_estimators"      : trial.suggest_int("n_estimators", 300, 2000),
        "learning_rate"     : trial.suggest_float("learning_rate", 0.003, 0.1, log=True),
        "max_depth"         : trial.suggest_int("max_depth", 3, 9),
        "num_leaves"        : trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples" : trial.suggest_int("min_child_samples", 10, 60),
        "subsample"         : trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree"  : trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha"         : trial.suggest_float("reg_alpha", 1e-5, 2.0, log=True),
        "reg_lambda"        : trial.suggest_float("reg_lambda", 1e-5, 2.0, log=True),
        "min_split_gain"    : trial.suggest_float("min_split_gain", 0.0, 0.5),
        "random_state"      : 42,
        "n_jobs"            : -1,
        "verbose"           : -1,
    }
    m = lgb.LGBMRegressor(**params)
    m.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        callbacks=[
            lgb.early_stopping(40, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    return mean_squared_error(y_vl, m.predict(X_vl))


# ------------------------------------------------------------------ #
#  Public API
# ------------------------------------------------------------------ #
def tune_lgbm(X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray,   y_val: np.ndarray,
              n_trials: int = 100) -> dict:
    """
    Chay Optuna de tim best_params cho LightGBM.
    Tra ve dict best_params + study object.
    """
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    study.optimize(
        lambda trial: _lgbm_objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    print(f"\nOptuna LightGBM hoan tat! Best val MSE: {study.best_value:.6f}")
    return {"best_params": study.best_params, "study": study}


def train_lgbm(X_train: np.ndarray, y_train: np.ndarray,
               X_val: np.ndarray,   y_val: np.ndarray,
               best_params: dict) -> lgb.LGBMRegressor:
    """Train LightGBM voi best_params tu Optuna."""
    params = {**best_params, "random_state": 42, "n_jobs": -1, "verbose": -1}
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )
    return model
