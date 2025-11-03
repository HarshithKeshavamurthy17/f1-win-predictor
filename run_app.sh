#!/bin/bash

# F1 Race Predictor - Launch Script

echo "🏎️  Starting F1 Race Win Predictor..."
echo "=================================="

# Check if environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "f1-win" ]]; then
    echo "⚠️  Activating f1-win environment..."
    conda activate f1-win
fi

# Install streamlit if needed
python -c "import streamlit" 2>/dev/null || pip install streamlit plotly

echo "🚀 Launching Streamlit app..."
echo "📱 Open your browser to: http://localhost:8501"
echo ""

streamlit run app.py --server.port 8501

