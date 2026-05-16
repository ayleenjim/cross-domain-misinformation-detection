#!/bin/bash

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
echo "Opening virtual environment..."

if [ ! -f "venv/.requirements_installed" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
    touch venv/.requirements_installed
else
    echo "Requirements already installed. Skipping install."
fi

echo "Starting Streamlit app..."
streamlit run app/streamlit_app.py