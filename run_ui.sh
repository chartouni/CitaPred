#!/bin/bash

# CitaPred UI Launcher
# Quick script to launch different interfaces

echo "========================================="
echo "    CitaPred - Citation Predictor"
echo "========================================="
echo ""
echo "Choose an interface to launch:"
echo ""
echo "1) Streamlit Web App (Full-featured)"
echo "2) Gradio Interface (Simple demo)"
echo "3) FastAPI REST API"
echo "4) Install dependencies"
echo "5) Exit"
echo ""
read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Launching Streamlit Web App..."
        echo "📍 URL: http://localhost:8501"
        echo ""
        streamlit run app.py
        ;;
    2)
        echo ""
        echo "🚀 Launching Gradio Interface..."
        echo "📍 URL: http://localhost:7860"
        echo ""
        python app_gradio.py
        ;;
    3)
        echo ""
        echo "🚀 Launching FastAPI REST API..."
        echo "📍 API: http://localhost:8000"
        echo "📚 Docs: http://localhost:8000/docs"
        echo ""
        python api.py
        ;;
    4)
        echo ""
        echo "📦 Installing dependencies..."
        pip install -r requirements.txt
        echo ""
        echo "✅ Installation complete!"
        ;;
    5)
        echo ""
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo ""
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac
