# Environment Setup (Python 3.9)

TensorFlow does not support your main Python 3.14, so the notebooks run on **Python 3.9.13**.
This sets up an isolated virtual env (`.venv39`) with every library — including TensorFlow
(iTransformer/GUMNet/LSTM) and PyTorch + neuralforecast (PatchTST/TFT) — and a Jupyter kernel.

## 1. Install (one command)

**PowerShell** (recommended):
```powershell
cd E:\PCDOC\xangdau\XANG_DAU_FORECAST\XANG_DAU_FORECAST
# if scripts are blocked, run once: Set-ExecutionPolicy -Scope Process Bypass
.\setup_env.ps1
```

**or Command Prompt**:
```bat
cd E:\PCDOC\xangdau\XANG_DAU_FORECAST\XANG_DAU_FORECAST
setup_env.bat
```

If the auto-detect can't find Python 3.9, pass its path:
```powershell
.\setup_env.ps1 -Python "C:\Users\Admin\AppData\Local\Programs\Python\Python39\python.exe"
```
```bat
setup_env.bat "C:\Users\Admin\AppData\Local\Programs\Python\Python39\python.exe"
```

The script: creates `.venv39`, installs `requirements-py39.txt`, registers the Jupyter
kernel **"Python 3.9 (xangdau)"**, and runs `verify_env.py`.

## 2. Run the notebook

Open `notebooks/03_final_all_models.ipynb`, then **Kernel → Change Kernel →
"Python 3.9 (xangdau)"** and Run All.

Launch Jupyter from the activated env if you like:
```powershell
.\.venv39\Scripts\Activate.ps1
python -m notebook        # or: python -m jupyterlab
```

## 3. Verify anytime
```powershell
.\.venv39\Scripts\Activate.ps1
python verify_env.py
```
It checks every import, the data file, and 1-epoch-trains iTransformer + GUMNet-Ultra to
confirm TensorFlow works.

## Notes
- **Keras 2 compatibility:** TF 2.17 ships Keras 3, but the repo models use the Keras-2 API.
  The notebooks set `TF_USE_LEGACY_KERAS=1` (via the installed `tf-keras`) before importing
  TensorFlow, so everything runs unchanged. `verify_env.py` does the same.
- **CPU only:** torch/TF install CPU builds (fine for these models). Training the DL models
  is the slow part — lower `CONFIG['dl_epochs']`/`nf_steps` for a quick pass, raise for accuracy.
- **Data:** ensure `data/processed/clean_data_exo_ver1.csv` exists (already staged).
- **News (optional):** to populate news features, run the `news-crawler` pipeline
  (crawl → sentiment.py → aggregate.py) to produce `news-crawler/data/daily_features.csv`.
