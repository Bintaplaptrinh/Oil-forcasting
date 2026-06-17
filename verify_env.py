#!/usr/bin/env python3
# Verify the Python 3.9 environment for XANG_DAU_FORECAST.
# Run:  python verify_env.py
import os, sys, importlib
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")   # repo models use Keras-2 API
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from pathlib import Path

print("=" * 60)
print("Python:", sys.version.split()[0], "(", sys.executable, ")")
print("=" * 60)

mods = ["numpy","pandas","scipy","sklearn","statsmodels","matplotlib",
        "seaborn","lightgbm","optuna","tensorflow","tf_keras","torch",
        "neuralforecast","openai"]
ok = True
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  OK   {m:15s} {getattr(mod,'__version__','?')}")
    except Exception as e:
        ok = False
        print(f"  FAIL {m:15s} {type(e).__name__}: {e}")

# data file present?
root = Path(__file__).resolve().parent
data = root / "data" / "processed" / "clean_data_exo_ver1.csv"
print("-" * 60)
print("Data file:", "FOUND" if data.exists() else "MISSING ->", data)

# tiny TF smoke test on the repo's SOTA models
print("-" * 60)
print("Smoke-testing src deep models (1 epoch on random data)...")
try:
    import numpy as np
    sys.path.insert(0, str(root))
    from src.models.hybrid_sota import train_itransformer, train_gumnet_ultra
    T, F = 10, 6
    X = np.random.randn(40, T, F).astype("float32")
    y = np.random.randn(40).astype("float32")
    m, _ = train_itransformer(X[:30], y[:30], X[30:], y[30:],
                              time_steps=T, n_features=F, epochs=1, batch_size=8)
    print("  OK   iTransformer -> pred", m.predict(X[:2], verbose=0).shape)
    m2, _ = train_gumnet_ultra(X[:30], y[:30], X[30:], y[30:],
                               time_steps=T, n_features=F, epochs=1, batch_size=8)
    print("  OK   GUMNet-Ultra trained")
except Exception as e:
    ok = False
    print("  FAIL TF smoke:", repr(e))

# torch / neuralforecast import
print("-" * 60)
try:
    import torch
    print("  OK   torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
    from neuralforecast.models import PatchTST, TFT  # noqa
    print("  OK   neuralforecast PatchTST/TFT importable")
except Exception as e:
    ok = False
    print("  FAIL torch/neuralforecast:", repr(e))

print("=" * 60)
print("RESULT:", "ALL GOOD — open notebooks/03_final_all_models.ipynb" if ok
      else "Some checks FAILED — see lines above")
print("=" * 60)
