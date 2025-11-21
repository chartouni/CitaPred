@echo off
REM CitaPred UI Launcher for Windows

echo =========================================
echo     CitaPred - Citation Predictor
echo =========================================
echo.
echo Choose an interface to launch:
echo.
echo 1) Streamlit Web App (Full-featured)
echo 2) Gradio Interface (Simple demo)
echo 3) FastAPI REST API
echo 4) Install dependencies
echo 5) Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    echo Starting Streamlit Web App...
    echo URL: http://localhost:8501
    echo.
    streamlit run app.py
    goto end
)

if "%choice%"=="2" (
    echo.
    echo Starting Gradio Interface...
    echo URL: http://localhost:7860
    echo.
    python app_gradio.py
    goto end
)

if "%choice%"=="3" (
    echo.
    echo Starting FastAPI REST API...
    echo API: http://localhost:8000
    echo Docs: http://localhost:8000/docs
    echo.
    python api.py
    goto end
)

if "%choice%"=="4" (
    echo.
    echo Installing dependencies...
    pip install -r requirements.txt
    echo.
    echo Installation complete!
    pause
    goto end
)

if "%choice%"=="5" (
    echo.
    echo Goodbye!
    goto end
)

echo.
echo Invalid choice. Please run the script again.
pause

:end
