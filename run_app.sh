#!/bin/bash

# CitaPred Web Application Launcher
# Simple script to launch the Streamlit web interface

echo "🚀 Launching CitaPred Web Application..."
echo ""

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit is not installed!"
    echo "Installing dependencies..."
    pip install -r requirements.txt
fi

# Check if models exist
if [ ! -d "models" ] || [ -z "$(ls -A models)" ]; then
    echo "⚠️  Warning: No trained models found in models/ directory"
    echo ""
    echo "Please train models first:"
    echo "  python scripts/train_classification.py"
    echo "  python scripts/train_regression.py"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Launch Streamlit app
echo "✅ Starting web application..."
echo "📱 Open your browser at: http://localhost:8501"
echo ""

streamlit run app.py
