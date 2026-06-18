# Oil-Forecasting

**Forecasting domestic refined-fuel prices (MG95, MG92, DO 0.001%, DO 0.05%) from global crude
benchmarks, macro indicators, and news sentiment.**

| | |
|---|---|
| **Team** | Tran Manh Hung - Nguyen Phuoc Toan - Nguyen Huu Tuan Phat |
| **Horizon** | 1, 5, 10, 30, 60 trading days |
| **Champion model** | Jump-Gated ARIMAX -> CatBoost (residual) |
| **Best score (Notebook 04, H=1)** | **MAE 1.2571 - RMSE 2.6541 - MAPE 1.26% · SMAPE 1.26% - R² 0.9779** |
| **Data span** | 2008-05-01 -> 2026-05-08 (~4,600 daily rows) |

---

## 1. Overview & Introduction

Oil-Forecasting predicts the daily price of Vietnamese refined-fuel products from a panel of
global drivers — WTI and Brent crude, the US Dollar Index, the Geopolitical Risk index (GPR),
and a custom **news-sentiment** signal built from war / political-economy / natural-disaster
headlines. The project combines classical statistical models, gradient boosting, deep-learning
sequence models, and a hybrid **Jump-Gated ARIMAX → CatBoost** champion, evaluated across five
forecast horizons.

The motivation is practical: refined-fuel prices track crude with a lag and a refining margin
(the *crack spread*), but they also jump on macro shocks and geopolitical events. A model that
blends a strong linear backbone (ARIMAX) with a non-linear residual learner (CatBoost), gated by
recent volatility, captures both the smooth co-movement and the shock-driven jumps.

*(All work is reproducible from the staged dataset and the `src/` pipeline; the notebooks
`01`–`05` document EDA, baseline modeling, the full model suite, the multi-horizon study, and
champion-improvement experiments.)*

---

## 2. The Data

Daily series, 2008-05-01 → 2026-05-08, one row per trading day. Targets are four refined products;
predictors are crude benchmarks, USD Index, and the GPR geopolitical-risk index.

| Group | Columns |
|---|---|
| **Targets** | `MG95`, `MG92`, `MG97`, `DO 0.001%`, `DO 0.05%`, `NAPHTHA`, `KERO`, `FO 180` |
| **Crude** | `WTI`, `BRT DTD`, `BRT KH`, `Brent_EU_Daily`, `WTI_Daily`, `Brent_Global_Monthly`, `WTI_Monthly` |
| **Macro / risk** | `USD_Index`, `GPR` |
| **News (engineered)** | per-topic daily news count, mean / sum sentiment, attention intensity |

### Price overview

MG95 moves tightly with crude but with its own margin and shock dynamics (GFC 2008, the 2014 OPEC
glut, COVID-19 2020, and the 2022 Russia–Ukraine war are all visible).

![Price overview](docs/images/price_overview.png)

### Correlation structure

Refined products and crude benchmarks are highly co-linear (r ≈ 0.9+); USD Index is mildly
negative; GPR is weakly correlated but matters in shock regimes.

![Correlation heatmap](docs/images/correlation_heatmap.png)

### Returns & volatility

Daily returns are fat-tailed and centered near zero; volatility clusters strongly around the
2008, 2020, and 2022 shocks — motivating the **Jump Gate** in the champion model.

![Returns and volatility](docs/images/returns_volatility.png)

### Seasonality

Additive weekly decomposition shows a dominant low-frequency trend, a modest annual seasonal
component, and shock-driven residuals.

![Decomposition](docs/images/decomposition.png)

---

## 3. Data Processing

Two data streams are merged on the calendar date.

**(a) Market & macro panel** — `data/processed/clean_data_exo_ver1.csv`. Loaded with a datetime
index, forward/back-filled for non-trading gaps, then feature-engineered (Section 4).

**(b) News sentiment pipeline** — `news-crawler/` (Node.js + Python):

1. **Crawl** headlines + timestamps from Federal Reserve (2006+), oilprice.com (2009+), and GDELT
   (2017+, incl. CNBC and OPEC via domain filter), tagged `war` / `political_economy` /
   `natural_disaster`.
2. **Score** each headline to a sentiment value in **[-1, 1]** with **MiniMax-M3** — sign = market
   direction, magnitude = importance (batched, cached, resumable).
3. **Aggregate** to daily features (per-topic news count, mean / sum sentiment, attention
   intensity) → `daily_features.csv`, left-joined to the price panel (0-filled on quiet days).

*(Why these sources: CNBC and OPEC block direct crawling — CNBC's `robots.txt` disallows AI agents
and OPEC sits behind Cloudflare — so both are pulled compliantly through GDELT's domain filter.)*

---

## 4. Feature Engineering

Built in `src/data_loader.py` (`load_and_engineer`) plus the notebooks. All features are causal
(known at time *t* when forecasting *t+H*).

| Family | Features |
|---|---|
| **Lags** | target & WTI lags {1, 2, 3, 5, 7, 14, 30} (from PACF analysis) |
| **Rolling stats** | MA-7 / MA-30 and rolling volatility for target & WTI |
| **Rate of change** | 7- and 30-day percentage change |
| **Spreads** | `Crack_Spread` (target − WTI), `Brent_WTI_Spread` |
| **Cyclical (date)** | `Month_sin/cos`, `DOW_sin/cos`, and **`Sin(Date)` / `Cos(Date)`** = day-of-year `DOY_sin/cos` (annual seasonality) |
| **News** | per-topic daily count, mean / sum sentiment, |sentiment| intensity |

Scaling uses **RobustScaler** (median / IQR) so 2008 / 2020 / 2022 outliers do not distort the
fit. Train/val/test are split **chronologically** (80 / 10 / 10) — never random — to avoid
look-ahead leakage. For the deep models, a sliding window builds `[samples, time_steps, features]`
tensors with horizon-dependent sequence length.

---

## 5. Modeling Approach

Three families plus a hybrid champion, all evaluated on the same test window and metrics
(**MAE, RMSE, MAPE, SMAPE, R²**).

| Family | Models |
|---|---|
| **Statistical** | ARIMAX, SARIMA (rolling one-step / H-step via `extend`) |
| **Linear / tree** | Ridge / Linear Regression, LightGBM (Optuna-tuned); Logistic Regression for up/down direction |
| **Deep learning** | LSTM, iTransformer (inverted attention), GUMNet-Lite & GUMNet-Ultra (gated CNN-BiGRU mixture-of-experts), PatchTST, TFT |
| **Champion (hybrid)** | **Jump-Gated ARIMAX → CatBoost**: ARIMAX gives the linear + exogenous forecast, CatBoost learns the non-linear **residual**, and a **Jump Gate** (sigmoid of recent-volatility z-score) controls how much residual correction to apply during turbulent periods |

**Champion-improvement experiments (Notebook 05)** add five residual-side techniques on top of the
champion: (1) **LSH** analog residuals, (2) **SAX + date→text + Log-Transform Hashing**, (3)
**Topological Data Analysis** (sublevel-set persistence + Takens embedding — "Sequence" + "Star"),
(4) **B-spline** basis with automatic quantile knots, and (5) an **NNLS stacking ensemble**.

---

## 6. Tech Stack & Libraries

| Layer | Tools |
|---|---|
| **Language / runtime** | Python 3.9 (DL stack), Node.js 22 (news crawler) |
| **Data** | pandas, numpy, scipy |
| **Statistical** | statsmodels (SARIMAX) |
| **ML** | scikit-learn, LightGBM, CatBoost, Optuna |
| **Deep learning** | TensorFlow / Keras (LSTM, iTransformer, GUMNet); PyTorch + neuralforecast (PatchTST, TFT) |
| **Signal / topology** | scipy splines & MST, custom SAX / LSH / sublevel-persistence |
| **News sentiment** | OpenAI client → MiniMax-M3 (via TokenRouter) |
| **Viz** | matplotlib, seaborn |

Environment is reproducible via `setup_env.ps1` / `setup_env.bat` + `requirements-py39.txt`
(verified by `verify_env.py`). See `SETUP.md`.

---

## 7. Results & Insights (MG95)

### 7.1 Multi-horizon

**Jump-Gated ARIMAX-CatBoost is the project's single best model** — it wins at H = 1, 5, 10 and is
competitive at 30 / 60. Best score at H=1: **MAE 1.2571; R² 0.9779**. Deep-learning models
consistently underperform on this dataset.

![Multi-horizon results](docs/images/results_multihorizon.png)

MAE (lower is better) / R² (higher is better) by horizon:

| Model | H=1 | H=5 | H=10 | H=30 | H=60 |
|---|---|---|---|---|---|
| **Jump-Gated ARIMAX-CatBoost** | **1.257 / 0.978** | **2.906 / 0.903** | **3.923 / 0.793** | 6.518 / **0.527** | 7.161 / 0.465 |
| ARIMAX | 1.429 / 0.975 | 2.953 / 0.899 | 3.992 / 0.789 | **6.390** / 0.500 | **7.122 / 0.490** |
| SARIMA | 1.511 / 0.971 | 3.770 / 0.852 | 4.879 / 0.677 | 8.580 / 0.027 | 10.99 / −0.38 |
| Ridge (Linear) | 1.542 / 0.970 | 3.725 / 0.848 | 4.786 / 0.722 | 9.855 / −0.12 | 9.446 / −0.02 |
| LightGBM | 2.237 / 0.938 | 5.704 / 0.616 | 7.238 / 0.215 | 9.685 / −0.02 | 12.32 / −0.27 |
| GUMNet-Ultra (best DL) | 4.179 / 0.827 | 5.692 / 0.611 | 8.819 / 0.269 | 11.42 / 0.087 | 10.71 / −0.38 |
| LSTM | 4.226 / 0.777 | 5.653 / 0.579 | 7.360 / 0.277 | 9.345 / 0.100 | 18.01 / −1.53 |

### 7.2 Champion-improvement experiments

Notebook 05 stress-tests five residual-side enhancements on a **harder evaluation setup** (ARIMAX
fit on the first 50% and rolled H-step over the remaining ~50% — a much longer test window than
nb04's last-10%). Scores are therefore higher than nb04 and **not directly comparable**; this
notebook ranks *which enhancement helps*, not the headline accuracy.

The winner shifts with horizon: the **ensemble** edges H=1, the **Jump-Gated base** wins H=5, and
**B-spline (auto knots)** dominates the long horizons (H=10/30/60) — confirming that the topological
/ basis features pay off in the shock-driven, longer-horizon regime. **LSH consistently hurts.**

![Improvement experiments](docs/images/results_improvements.png)

Best MAE per horizon (Notebook 05 setup):

| Horizon | Best technique | MAE | R² |
|---|---|---|---|
| H=1 | (5) Ensemble (NNLS stack) | 1.434 | 0.974 |
| H=5 | Jump-Gated ARIMAX-CatBoost (base) | 2.915 | 0.902 |
| H=10 | (4) B-spline (auto) | 3.819 | 0.802 |
| H=30 | (4) B-spline (auto) | 6.028 | 0.518 |
| H=60 | (4) B-spline (auto) | 6.654 | 0.485 |

> The single best result of the whole project remains **Notebook 04's Jump-Gated ARIMAX-CatBoost at
> H=1 (MAE 1.2571, R² 0.9779)** — nb05's numbers come from a deliberately harder rolling setup.

### 7.3 Four-target summary

R² / MAPE(%) on the test set, full `main.py` pipeline:

| Target | LightGBM | iTransformer | GUMNet-Ultra |
|---|---|---|---|
| **MG95** | **0.943 / 2.21** | 0.806 / 3.46 | 0.938 / 2.40 |
| **MG92** | 0.933 / 2.77 | 0.867 / 3.60 | **0.937 / 2.22** |
| **DO 0.001%** | **0.795 / 3.31** | 0.591 / 5.42 | 0.754 / 4.00 |
| **DO 0.05%** | 0.720 / 3.34 | 0.598 / 5.42 | **0.807 / 4.04** |

### Key insights

- **Simpler wins.** The linear-backbone hybrid (ARIMAX + CatBoost residual) beats every deep
  model. Refined-fuel prices are dominated by lag-1 autocorrelation and crude co-movement, which a
  strong linear model captures almost fully at short horizons.
- **The Jump Gate adds the most at mid horizons (H = 5–10)**, exactly where ARIMAX starts to
  decay but structure still exists.
- **Signal decays sharply with horizon** - R² falls from ~0.98 (H=1) to ~0.49 (H=60). Beyond ~30
  days the series is close to a random walk and all models converge toward the naive baseline.
- **News + topological / symbolic features** give marginal H=1 gains but are aimed at the
  shock-driven tail; their value grows at longer horizons and in crisis windows.

---

## 8. Project Structure

```
Oil-Forecasting/
├── README.md                  (this report)
├── SETUP.md                   environment setup guide
├── main.py                    full 4-target pipeline entry point
├── requirements-py39.txt      pinned deps (TF + PyTorch + CatBoost)
├── setup_env.ps1 / .bat       one-command environment setup
├── verify_env.py              environment checker
├── data/processed/            clean_data_exo_ver1.csv (market + macro panel)
├── src/                       data_loader, features, evaluation, models/
├── notebooks/                 01 EDA - 02 baseline - 03 all-models - 04 multi-horizon - 05 champion-improvements
├── news-crawler/              Node + Python news sentiment pipeline (crawl -> score -> aggregate)
├── results/                   metrics CSVs + charts/ (per-model & per-target)
└── docs/images/               figures used in this report
```

---

*Oil-Forecasting — Tran Manh Hung, Nguyen Phuoc Toan, Nguyen Huu Tuan Phat.*
