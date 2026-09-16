@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo   Douyin Favorite Knowledge Base - Launcher
echo ============================================

rem -- 1. Check Python --
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo         Install Python 3.10+ from https://www.python.org/downloads/
    echo         Check "Add python.exe to PATH" during install.
    pause
    exit /b 1
)

rem -- 2. Create venv if missing --
if not exist ".venv\Scripts\python.exe" (
    echo [1/5] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
)

rem -- 3. Install dependencies (Tsinghua mirror, fast in CN) --
echo [2/5] Installing dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --quiet
if errorlevel 1 (
    echo [ERROR] pip install failed. Check your network and retry.
    pause
    exit /b 1
)

rem -- Chromium for Playwright (mirror CDN, much faster in CN) --
set PLAYWRIGHT_DOWNLOAD_HOST=https://registry.npmmirror.com/-/binary/playwright
".venv\Scripts\python.exe" -m playwright install chromium

rem -- 3. Whisper model (~461 MB, one-time, saved into models/) --
if exist "models\faster-whisper-small\model.bin" (
    echo [3/5] Whisper model already downloaded, skip.
) else (
    echo [3/5] Pre-downloading Whisper model. Details and progress below...
    ".venv\Scripts\python.exe" "scripts\download_model.py" small
)

rem -- 4. BGE embedding model (~120 MB, one-time, for AI Q&A semantic search) --
rem     Missing it is not fatal: Q&A falls back to keyword-only retrieval.
if exist "models\bge-small-zh-v1.5\onnx\model_quantized.onnx" (
    echo [4/5] Embedding model already downloaded, skip.
) else (
    echo [4/5] Pre-downloading BGE embedding model. Details and progress below...
    ".venv\Scripts\python.exe" "scripts\download_model.py" bge
)

rem -- 5. .env from template --
if not exist ".env" (
    copy .env.example .env >nul
    echo.
    echo [ACTION NEEDED] .env was created from template.
    echo Open .env and fill in LLM_API_KEY with your key
    echo (any OpenAI-compatible service, e.g. DeepSeek https://platform.deepseek.com)
    echo Then run start.bat again. Summary and Q-A need this key.
    echo.
)

rem -- 6. Launch --
echo [5/5] Starting web UI at http://127.0.0.1:8642 ...
echo First time? Click "Scan QR to Login" then "Sync Favorites" in the page.
start "" cmd /c "timeout /t 3 >nul & start http://127.0.0.1:8642"
".venv\Scripts\python.exe" main.py web --port 8642

pause
