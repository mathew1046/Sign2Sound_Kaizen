#!/bin/bash

# Complete Pipeline Runner
# Orchestrates preprocessing, training, and evaluation

set -e  # Exit on error

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================"
echo "Sign Language Recognition Pipeline"
echo "======================================"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}⚠ Activating virtual environment...${NC}"
    source venv/bin/activate
fi

# Stage 1: Preprocessing
echo ""
echo -e "${YELLOW}Stage 1: Data Preprocessing${NC}"
echo "======================================"

if [ -z "$MALAYALAM_PATH" ] || [ -z "$ISL_PATH" ]; then
    echo -e "${RED}✗ Please set MALAYALAM_PATH and ISL_PATH environment variables${NC}"
    echo "  Export: export MALAYALAM_PATH=/path/to/malayalam/data"
    echo "          export ISL_PATH=/path/to/isl/data"
    echo ""
    echo "Skipping preprocessing..."
else
    echo "Processing Malayalam data: $MALAYALAM_PATH"
    echo "Processing ISL data: $ISL_PATH"
    
    python3 preprocessing/preprocess.py \
        --malayalam_path "$MALAYALAM_PATH" \
        --isl_path "$ISL_PATH" \
        --output data/processed \
        --augment_count 50 \
        --max_seq_len 60
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Preprocessing complete${NC}"
    else
        echo -e "${RED}✗ Preprocessing failed${NC}"
        exit 1
    fi
fi

# Stage 2: Training
echo ""
echo -e "${YELLOW}Stage 2: Model Training${NC}"
echo "======================================"

python3 training/train.py \
    --config training/config.yaml \
    --device cuda 
    
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Training complete${NC}"
else
    echo -e "${RED}✗ Training failed${NC}"
    exit 1
fi

# Stage 3: Evaluation
echo ""
echo -e "${YELLOW}Stage 3: Model Evaluation${NC}"
echo "======================================"

python3 training/evaluate.py \
    --model checkpoints/best_model.pth \
    --device cuda

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Evaluation complete${NC}"
else
    echo -e "${RED}✗ Evaluation failed${NC}"
    exit 1
fi

# Stage 4: Generate reports
echo ""
echo -e "${YELLOW}Stage 4: Generating Reports${NC}"
echo "======================================"

echo "Generating inference demo scripts..."

# Create summary report
cat > results/PIPELINE_SUMMARY.txt << EOF
Sign Language Recognition - Pipeline Execution Summary
======================================================

Generated: $(date)
Hostname: $(hostname)
User: $(whoami)

Directory Structure:
  - data/: Dataset and processed data
  - models/: Model source code
  - training/: Training pipeline
  - inference/: Inference modules
  - results/: Training metrics and visualizations
  - checkpoints/: Model checkpoints
  - logs/: Training logs

Output Files:
  - Training metrics: results/training_metrics.json
  - Training curves: results/training_curves.png
  - Test metrics: results/test_metrics.json
  - Per-class metrics: results/per_class_metrics.csv
  - Confusion matrix: results/confusion_matrix.png
  - Classification report: results/classification_report.txt
  - Best model: checkpoints/best_model.pth

Next Steps:
  1. Run inference: python inference/infer.py --model checkpoints/best_model.pth --input <image>
  2. Real-time demo: python inference/realtime_demo.py --model checkpoints/best_model.pth
  3. Analyze results: jupyter notebook notebooks/03_results_visualization.ipynb

For more information, see README.md
EOF

cat results/PIPELINE_SUMMARY.txt

echo -e "${GREEN}✓ Reports generated${NC}"

# Final summary
echo ""
echo "======================================"
echo -e "${GREEN}✓ Pipeline execution complete!${NC}"
echo "======================================"
echo ""
echo "Results saved to:"
echo "  - Training metrics: results/"
echo "  - Model checkpoint: checkpoints/best_model.pth"
echo "  - Visualizations: results/"
echo ""
echo "To run inference on new images:"
echo "  python inference/infer.py --model checkpoints/best_model.pth --input <image_path>"
echo ""
echo "To run real-time demo:"
echo "  python inference/realtime_demo.py --model checkpoints/best_model.pth"
echo ""
