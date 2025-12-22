#!/bin/bash
#SBATCH --job-name=resnet50
#SBATCH --partition=gpuH200x8
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --account=bdjd-delta-gpu

export WANDB_SERVICE_WAIT=120
export WANDB_START_METHOD=thread

module load gcc/11.4.0
module load cuda/12.4.0

export PATH=/work/hdd/bdjd/miniconda3/bin:$PATH
source /work/hdd/bdjd/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_fresh

# WandB API Key Login
export WANDB_API_KEY=747cf95b0b1187309131da4aff9bbb74d1238170

# Set PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/work/hdd/bdjd/Iso:/work/hdd/bdjd/Iso

# Create .env file with OpenRouter API key
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
OPENROUTER_API_KEY=sk-or-v1-YOUR_ACTUAL_OPENROUTER_API_KEY_HERE
EOF
    echo "✅ .env file created with OpenRouter API key"
    echo "❌ WARNING: You MUST replace YOUR_ACTUAL_OPENROUTER_API_KEY_HERE with your real API key!"
    echo "❌ The workflow will fail without a valid API key!"
else
    echo "✅ .env file already exists"
fi

# Verify the API key is properly set (basic check)
if grep -q "YOUR_ACTUAL_OPENROUTER_API_KEY_HERE" .env; then
    echo "❌ ERROR: Please update .env file with your actual OpenRouter API key!"
    echo "❌ Replace 'YOUR_ACTUAL_OPENROUTER_API_KEY_HERE' with your real API key"
    exit 1
fi

# ImageNet dataset is already available, just check the path
echo "Checking ImageNet dataset availability..."
IMAGENET_PATH="/work/hdd/bdjd/imagenet"
if [ -d "$IMAGENET_PATH" ]; then
    echo "ImageNet dataset found at: $IMAGENET_PATH"
    ls -la "$IMAGENET_PATH"
else
    echo "ERROR: ImageNet dataset not found at expected path: $IMAGENET_PATH"
    exit 1
fi

# DEFINE EXPERIMENT PARAMETERS
MODEL_NAME="resnet50"
MACS_TARGET_G="2.06"
DATASET="imagenet"

# CREATE ORGANIZED OUTPUT DIRECTORY STRUCTURE
BASE_OUTPUT_DIR="/work/hdd/bdjd"
EXPERIMENT_DIR="${BASE_OUTPUT_DIR}/expf_cl/${MODEL_NAME}_${DATASET}_macs${MACS_TARGET_G/./p}g"
JOB_OUTPUT_DIR="${EXPERIMENT_DIR}/job_${SLURM_JOB_ID}"

# Create all necessary subdirectories
mkdir -p "$JOB_OUTPUT_DIR"
mkdir -p "${JOB_OUTPUT_DIR}/models"
mkdir -p "${JOB_OUTPUT_DIR}/logs"
mkdir -p "${JOB_OUTPUT_DIR}/checkpoints"

if [ $? -eq 0 ]; then
    echo "✅ Created experiment directory: $EXPERIMENT_DIR"
    echo "✅ Created job output directory: $JOB_OUTPUT_DIR"
else
    echo "❌ Failed to create output directories"
    exit 1
fi

# REDIRECT ALL OUTPUT TO ORGANIZED LOG FILE
LOG_FILE="${JOB_OUTPUT_DIR}/logs/pruning_workflow_${SLURM_JOB_ID}.txt"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "🚀 STARTING PRUNING EXPERIMENT"
echo "========================================="
echo "Model: $MODEL_NAME"
echo "Ratio: $RATIO"
echo "Dataset: $DATASET"
echo "Job ID: $SLURM_JOB_ID"
echo "Experiment Dir: $EXPERIMENT_DIR"
echo "Job Output Dir: $JOB_OUTPUT_DIR"
echo "Log File: $LOG_FILE"
echo "Final models will be saved to: ${JOB_OUTPUT_DIR}/models"
echo "Intermediate checkpoints will be saved to: ${JOB_OUTPUT_DIR}/checkpoints"
echo "========================================="

# FIXED: Run the pruning workflow with organized output directories
echo "Starting ImageNet workflow..."
python3 mac_pruning_12/main.py \
    --model "$MODEL_NAME" \
    --macs_target_g "$MACS_TARGET_G" \
    --macs_overshoot_tolerance_pct 5.0 \
    --macs_undershoot_tolerance_pct 15.0 \
    --dataset "$DATASET" \
    --data_path "$IMAGENET_PATH" \
    --max_revisions 1000 \
    --accuracy_threshold 30.0 \
    --imagenet_subset 0.3 \
    --output_dir "${JOB_OUTPUT_DIR}/models" \
    --checkpoint_dir "${JOB_OUTPUT_DIR}/checkpoints" \
    --wandb_project "Isomorphic_Pruning_Experiments" \
    --wandb_name "${MODEL_NAME}_${DATASET}_macs${MACS_TARGET_G/./p}_job${SLURM_JOB_ID}"


# Check if the workflow completed successfully
if [ $? -eq 0 ]; then
    echo "========================================="
    echo "✅ PRUNING WORKFLOW COMPLETED SUCCESSFULLY"
    echo "========================================="
    
    # Create experiment summary
    SUMMARY_FILE="${JOB_OUTPUT_DIR}/EXPERIMENT_SUMMARY.txt"
    cat > "$SUMMARY_FILE" << EOF
PRUNING EXPERIMENT SUMMARY
==========================
Job ID: $SLURM_JOB_ID
Timestamp: $(date)
Model: $MODEL_NAME
Dataset: $DATASET
Target Ratio: $RATIO
Experiment Directory: $EXPERIMENT_DIR
Job Output Directory: $JOB_OUTPUT_DIR

FILES GENERATED:
================
- Main Log: $LOG_FILE
- Final Models: ${JOB_OUTPUT_DIR}/models/
- Intermediate Checkpoints: ${JOB_OUTPUT_DIR}/checkpoints/
- This Summary: $SUMMARY_FILE

SLURM JOB INFO:
===============
Job ID: $SLURM_JOB_ID
Partition: $SLURM_JOB_PARTITION
Nodes: $SLURM_JOB_NUM_NODES
Start Time: $(date)
EOF

    echo "📊 Experiment summary saved to: $SUMMARY_FILE"
    echo "📁 All outputs organized in: $JOB_OUTPUT_DIR"
    echo "📜 Complete log available at: $LOG_FILE"
    
    # List all generated files in organized directories
    echo ""
    echo "📂 GENERATED FILES IN ORGANIZED STRUCTURE:"
    echo "Final Models:"
    find "${JOB_OUTPUT_DIR}/models/" -name "*.pth" 2>/dev/null | head -10
    echo "Intermediate Checkpoints:"
    find "${JOB_OUTPUT_DIR}/checkpoints/" -name "*.pth" 2>/dev/null | head -10
    echo "Logs:"
    find "${JOB_OUTPUT_DIR}/logs/" -name "*.txt" 2>/dev/null | head -5
    
else
    echo "========================================="
    echo "❌ PRUNING WORKFLOW FAILED"
    echo "========================================="
    echo "Check the log file for details: $LOG_FILE"
    exit 1
fi

echo "🎉 Experiment completed! Check $JOB_OUTPUT_DIR for all results."
