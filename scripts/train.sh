#!/usr/bin/env bash
# =============================================================================
# Script: train.sh
# Project: BAR-Sum (Factually Faithful Extractive Text Summarization via DRL)
# Author: Masoud Keikhosravi
# Affiliation: Islamic Azad University, Mashhad Branch
# Description:
#     End-to-end training execution script for the DRL Agent with 
#     Bounded Adaptive Inverse-Variance Reward Controller.
# =============================================================================

set -e # Exit on error

echo "======================================================================="
echo "                BAR-Sum: DRL MODEL TRAINING PIPELINE                   "
echo "======================================================================="

# 1. Configuration & Directories setup
CONFIG_PATH="config.yaml"
CHECKPOINT_DIR="checkpoints"
LOG_DIR="logs_and_results"
SEED=42
GPU_ID=0

mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${LOG_DIR}

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTHONUNBUFFERED=1

echo "[*] Setting Random Seed: ${SEED}"
echo "[*] Using GPU ID: ${GPU_ID}"
echo "[*] Checkpoints target: ${CHECKPOINT_DIR}/"
echo "[*] Logs directory:     ${LOG_DIR}/"
echo ""

# 2. Dataset Verification
if [ ! -d "data/cnn_dailymail" ] && [ ! -d "data/xsum" ]; then
    echo "[!] Warning: Processed datasets not found in data/."
    echo "[*] Running preprocessing pipeline first..."
    python preprocess.py --config ${CONFIG_PATH}
fi

# 3. Main Training Execution
echo "-----------------------------------------------------------------------"
echo "[*] Starting DRL Training with Bounded Adaptive Rewards..."
echo "-----------------------------------------------------------------------"

python train.py \
    --config ${CONFIG_PATH} \
    --seed ${SEED} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --log_dir ${LOG_DIR} \
    --w_min 0.10 \
    --w_max 0.45 \
    --ppo_epochs 4 \
    --clip_eps 0.2 \
    --gamma 0.99 \
    --lr_actor 1e-4 \
    --lr_critic 5e-4 \
    2>&1 | tee ${LOG_DIR}/training_console_output.log

echo ""
echo "======================================================================="
echo " [SUCCESS] Training pipeline completed successfully."
echo " Model Checkpoints: ${CHECKPOINT_DIR}/best_model.pt"
echo " Training Logs:     ${LOG_DIR}/training_logs.json"
echo "======================================================================="
