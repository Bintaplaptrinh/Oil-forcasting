"""
main.py
-------
Entry point: Chay toan bo pipeline du bao cho 4 mat hang:
MG95, MG92, DO 0.001%, DO 0.05%.
"""

import argparse
import warnings
import sys
import os
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader  import (build_pipeline, inverse_transform_y,
                                SEQ_LEN_BY_HORIZON)
from src.features     import walkforward_folds, make_multihorizon_windows
from src.models       import (tune_lgbm, train_lgbm,
                               train_itransformer, tune_itransformer,
                               train_gumnet_lite,
                               build_gumnet_ultra, train_gumnet_ultra, tune_gumnet_ultra)
from src.evaluation   import (compute_metrics, print_metrics,
                               walk_forward_lgbm, evaluate_multihorizon,
                               plot_actual_vs_predicted,
                               plot_multihorizon_bar,
                               plot_walkforward,
                               plot_model_comparison_bar,
                               CHARTS_DIR)

TARGETS = ["DO 0.001%", "DO 0.05%", "MG95", "MG92"]

def main(optuna_trials: int = 100,
         horizons: list = None,
         wf_folds: int = 8,
         fast_tune: bool = False):

    HORIZONS = horizons or [1, 5, 10]
    
    trials_lgbm = 20 if fast_tune else optuna_trials
    trials_dl   = 10 if fast_tune else min(30, optuna_trials)
    
    print("\n" + "="*65)
    print("  DU BAO THI TRUONG XANG DAU - MULTI-TARGET PIPELINE")
    print("="*65)
    print(f"Targets: {TARGETS}")
    print(f"Optuna Trials: LGBM={trials_lgbm}, DeepLearning={trials_dl}")

    all_targets_results = []

    for target in TARGETS:
        print("\n" + "#"*65)
        print(f"  BAT DAU XU LY TARGET: {target}")
        print("#"*65)
        
        target_dir = os.path.join(CHARTS_DIR, target.replace("%", "").replace(" ", "_"))
        os.makedirs(target_dir, exist_ok=True)

        splits = build_pipeline(horizon=1, time_steps=SEQ_LEN_BY_HORIZON[1], target_col=target)

        X_train = splits["X_train"];  y_train = splits["y_train"]
        X_val   = splits["X_val"];    y_val   = splits["y_val"]
        X_test  = splits["X_test"];   y_test  = splits["y_test"]
        scaler_y    = splits["scaler_y"]
        scaler_X    = splits["scaler_X"]
        n_features  = splits["n_features"]
        time_steps  = splits["time_steps"]
        date_index  = splits["date_index"]
        nt, nv      = splits["n_train"], splits["n_val"]

        X_tr2 = X_train.reshape(len(X_train), -1)
        X_v2  = X_val.reshape(len(X_val), -1)
        X_te2 = X_test.reshape(len(X_test), -1)

        test_dates = date_index[nt + nv:]

        # --- 1. LightGBM ---
        print(f"\n[Optuna] Tuning LightGBM ({trials_lgbm} trials)...")
        tune_out  = tune_lgbm(X_tr2, y_train, X_v2, y_val, n_trials=trials_lgbm)
        best_lgbm = tune_out["best_params"]
        model_lgbm= train_lgbm(X_tr2, y_train, X_v2, y_val, best_lgbm)
        pred_lgbm = model_lgbm.predict(X_te2)
        yt_inv    = inverse_transform_y(y_test, scaler_y)
        yp_lgbm_inv = inverse_transform_y(pred_lgbm, scaler_y)
        res_lgbm  = compute_metrics(yt_inv, yp_lgbm_inv, name="LightGBM")
        plot_actual_vs_predicted(test_dates, yt_inv, yp_lgbm_inv, "LightGBM", res_lgbm, save_path=f"{target_dir}/actual_vs_pred_LGBM.png")

        # --- 2. iTransformer ---
        print(f"\n[Optuna] Tuning iTransformer ({trials_dl} trials)...")
        tune_itf = tune_itransformer(X_train, y_train, X_val, y_val, time_steps, n_features, n_trials=trials_dl)
        best_itf = tune_itf["best_params"]
        model_itf, _ = train_itransformer(
            X_train, y_train, X_val, y_val, time_steps=time_steps, n_features=n_features, horizon=1, spike_lambda=3.0,
            epochs=150, batch_size=64, d_model=best_itf["d_model"], n_heads=best_itf["n_heads"], n_layers=best_itf["n_layers"]
        )
        pred_itf    = model_itf.predict(X_test, verbose=0).flatten()
        yp_itf_inv  = inverse_transform_y(pred_itf, scaler_y)
        res_itf     = compute_metrics(yt_inv, yp_itf_inv, name="iTransformer")
        plot_actual_vs_predicted(test_dates, yt_inv, yp_itf_inv, "iTransformer", res_itf, save_path=f"{target_dir}/actual_vs_pred_iTF.png")

        # --- 3. GUMNet-Ultra ---
        print(f"\n[Optuna] Tuning GUMNet-Ultra ({trials_dl} trials)...")
        tune_gum = tune_gumnet_ultra(X_train, y_train, X_val, y_val, time_steps, n_features, n_trials=trials_dl)
        best_gum = tune_gum["best_params"]
        model_gum, _ = train_gumnet_ultra(
            X_train, y_train, X_val, y_val, time_steps=time_steps, n_features=n_features, horizon=1, spike_lambda=3.0,
            epochs=150, batch_size=64, cnn_filters=best_gum["cnn_filters"], gru_units=best_gum["gru_units"], 
            dropout=best_gum["dropout"], lr=best_gum["lr"]
        )
        pred_gum    = model_gum.predict(X_test, verbose=0).flatten()
        yp_gum_inv  = inverse_transform_y(pred_gum, scaler_y)
        res_gum     = compute_metrics(yt_inv, yp_gum_inv, name="GUMNet-Ultra")
        plot_actual_vs_predicted(test_dates, yt_inv, yp_gum_inv, "GUMNet-Ultra", res_gum, save_path=f"{target_dir}/actual_vs_pred_GUMNet_Ultra.png")

        # So sanh
        plot_model_comparison_bar([
            {**res_lgbm, "y_pred": yp_lgbm_inv}, 
            {**res_itf, "y_pred": yp_itf_inv}, 
            {**res_gum, "y_pred": yp_gum_inv}
        ], save_path=f"{target_dir}/model_comparison.png")

        # --- 4. Walk-Forward (LightGBM) ---
        print(f"\n[Walk-Forward] LightGBM ({wf_folds} folds)...")
        n_total = len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_test"])
        X_all = np.concatenate([X_train, X_val, X_test])
        y_all = np.concatenate([y_train, y_val, y_test])
        folds = walkforward_folds(n_total=n_total, n_pretrain=nt, n_folds=wf_folds)
        wf_out = walk_forward_lgbm(X_all, y_all, scaler_y, folds, best_lgbm, date_index)
        plot_walkforward(wf_out["fold_results"], wf_out["summary"], save_path=f"{target_dir}/walk_forward.png")

        # --- 5. Multi-Horizon (GUMNet-Ultra chi test nhanh) ---
        print(f"\n[Multi-Horizon] LightGBM & GUMNet-Ultra {HORIZONS}...")
        df_feat = splits["df"]
        feature_cols = splits["feature_cols"]
        X_sc_full = scaler_X.transform(df_feat[feature_cols].values)
        y_sc_full = scaler_y.transform(df_feat[[target]].values).flatten()

        horizon_data_lgbm = make_multihorizon_windows(X_sc_full, y_sc_full, time_steps=time_steps, horizons=HORIZONS)
        def lgbm_predict(Xte): return model_lgbm.predict(Xte.reshape(len(Xte), -1))
        df_mh_lgbm = evaluate_multihorizon(lgbm_predict, horizon_data_lgbm, scaler_y, n_train=nt, n_val=nv, model_name="LightGBM")
        plot_multihorizon_bar(df_mh_lgbm, "LightGBM", save_path=f"{target_dir}/multihorizon_LGBM.png")

        rows_gum = []
        for h in HORIZONS:
            ts_h = SEQ_LEN_BY_HORIZON.get(h, time_steps)
            hw_h = make_multihorizon_windows(X_sc_full, y_sc_full, time_steps=ts_h, horizons=[h])
            Xw_h, yw_h = hw_h[h]
            nt_h, nv_h = int(len(Xw_h)*0.8), int(len(Xw_h)*0.1)
            Xtr_h, ytr_h = Xw_h[:nt_h], yw_h[:nt_h]
            Xvl_h, yvl_h = Xw_h[nt_h:nt_h+nv_h], yw_h[nt_h:nt_h+nv_h]
            Xte_h, yte_h = Xw_h[nt_h+nv_h:], yw_h[nt_h+nv_h:]

            m_h, _ = train_gumnet_ultra(
                Xtr_h, ytr_h, Xvl_h, yvl_h, time_steps=ts_h, n_features=n_features, horizon=h, spike_lambda=3.0,
                epochs=80, batch_size=64, cnn_filters=best_gum["cnn_filters"], gru_units=best_gum["gru_units"], 
                dropout=best_gum["dropout"], lr=best_gum["lr"]
            )
            pred_h   = m_h.predict(Xte_h, verbose=0).flatten()
            yt_inv_h = scaler_y.inverse_transform(yte_h.reshape(-1,1)).flatten()
            yp_inv_h = scaler_y.inverse_transform(pred_h.reshape(-1,1)).flatten()
            m = compute_metrics(yt_inv_h, yp_inv_h, name=f"H={h}")
            m["Horizon"] = h
            rows_gum.append(m)
        df_mh_gum = pd.DataFrame(rows_gum).set_index("Horizon")
        plot_multihorizon_bar(df_mh_gum, "GUMNet-Ultra", save_path=f"{target_dir}/multihorizon_GUMNet_Ultra.png")

        # Luu tong ket
        all_targets_results.append({
            "Target": target,
            "LGBM_R2": res_lgbm["R2"],
            "LGBM_MAPE": res_lgbm["MAPE(%)"],
            "iTrans_R2": res_itf["R2"],
            "iTrans_MAPE": res_itf["MAPE(%)"],
            "GUM_Ultra_R2": res_gum["R2"],
            "GUM_Ultra_MAPE": res_gum["MAPE(%)"],
        })
        print(f"\n[HOAN TAT] {target}")

    print("\n" + "="*65)
    print("  TONG KET TAT CA CAC TARGETS")
    print("="*65)
    df_summary = pd.DataFrame(all_targets_results)
    print(df_summary.to_string(index=False))
    df_summary.to_csv(f"{CHARTS_DIR}/Summary_4_Targets.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Target Forecast Pipeline")
    parser.add_argument("--quick",    action="store_true", help="Chay nhanh cho developer test")
    parser.add_argument("--fast-tune",action="store_true", help="Chay luot optuna (20 lgbm, 10 DL)")
    args = parser.parse_args()

    if args.quick:
        TARGETS = ["MG95"] # Test 1 cai thoi cho nhanh
        main(optuna_trials=1, horizons=[1], wf_folds=2, fast_tune=True)
    else:
        main(optuna_trials=100, horizons=[1, 5, 10], wf_folds=8, fast_tune=args.fast_tune)
