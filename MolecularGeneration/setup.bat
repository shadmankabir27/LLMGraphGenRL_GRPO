@echo off
echo ============================================================
echo  LLM Graph Generation - Automatic Setup
echo ============================================================
echo.

:: Step 1: Create virtual environment
echo [Step 1/4] Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: Could not create virtual environment.
    echo Make sure Python is installed and added to PATH.
    echo Download Python from https://python.org
    pause
    exit /b 1
)
echo   Done!
echo.

:: Step 2: Activate it
echo [Step 2/4] Activating virtual environment...
call .venv\Scripts\activate
echo   Done!
echo.

:: Step 3: Install PyTorch with CUDA support (for NVIDIA GPU)
echo [Step 3/4] Installing PyTorch with GPU support...
echo   (This is a large download ~2GB, please be patient)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo WARNING: GPU version failed. Trying CPU-only version...
    pip install torch torchvision
)
echo   Done!
echo.

:: Step 4: Install remaining dependencies
echo [Step 4/4] Installing other dependencies...
pip install networkx requests torch-geometric
echo   Done!
echo.

echo ============================================================
echo  Setup complete! 
echo ============================================================
echo.
echo  Next steps:
echo    1. Make sure Ollama is installed (https://ollama.com)
echo    2. Open a terminal and run: ollama pull llama3.1:8b
echo    3. To run the pipeline:
echo       .venv\Scripts\activate
echo       python pipeline_full.py
echo.
pause
