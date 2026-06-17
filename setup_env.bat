@echo off
REM ============================================================
REM  XANG_DAU_FORECAST environment setup (Python 3.9)
REM  Usage:
REM    setup_env.bat                          (auto-detect Python 3.9)
REM    setup_env.bat "C:\Path\to\python.exe"  (explicit 3.9 interpreter)
REM ============================================================
setlocal
set "ROOT=%~dp0"
set "VENV=%ROOT%.venv39"

if not "%~1"=="" (
  set "PY=%~1"
) else (
  py -3.9 --version >nul 2>&1
  if errorlevel 1 ( set "PY=python" ) else ( set "PY=py -3.9" )
)

echo === Using Python: %PY% ===
%PY% --version || (echo Could not run Python 3.9 & exit /b 1)

echo === Creating virtual env: %VENV% ===
%PY% -m venv "%VENV%" || (echo venv creation failed & exit /b 1)
call "%VENV%\Scripts\activate.bat"

echo === Installing packages (this can take several minutes) ===
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "%ROOT%requirements-py39.txt" || (echo pip install failed & exit /b 1)

echo === Registering Jupyter kernel ===
python -m ipykernel install --user --name xangdau-py39 --display-name "Python 3.9 (xangdau)"

echo === Verifying environment ===
python "%ROOT%verify_env.py"

echo.
echo ============================================================
echo  DONE. Open the notebook and select kernel:
echo    "Python 3.9 (xangdau)"
echo  To activate later:  "%VENV%\Scripts\activate.bat"
echo ============================================================
endlocal
