#!/bin/bash

# Setup Environment Script
# Creates virtual environment and installs dependencies

set -e  # Exit on error

echo "======================================"
echo "Sign Language Recognition Setup"
echo "======================================"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
echo "✓ Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel -q
echo "✓ Pip upgraded"

# Install requirements
echo ""
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
else
    echo "✗ requirements.txt not found"
    exit 1
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p checkpoints
mkdir -p results
mkdir -p logs
echo "✓ Directories created"

# Download MediaPipe model (optional)
echo ""
echo "Verifying MediaPipe installation..."
python3 -c "import mediapipe" && echo "✓ MediaPipe available" || echo "✗ MediaPipe not available"

# Run tests
echo ""
echo "Running basic tests..."
if [ -d "tests" ]; then
    python3 -m pytest tests/ -v --tb=short 2>/dev/null || echo "⚠ Some tests failed (check after full setup)"
fi

echo ""
echo "======================================"
echo "Setup Complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Activate environment: source venv/bin/activate"
echo "2. Prepare data: python preprocessing/preprocess.py --malayalam_path <path> --isl_path <path>"
echo "3. Train model: python training/train.py"
echo "4. Evaluate: python training/evaluate.py"
echo "5. Run inference: python inference/infer.py --model checkpoints/best_model.pth --input <image>"
echo ""
