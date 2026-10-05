#!/usr/bin/env bash
# ==============================================================================
# Script: download_datasets.sh
# Project: BAR-Sum (Multi-Objective DRL for Extractive Text Summarization)
# Author: Masoud Keikhosravi
# Description: Automates directory setup and downloads standard benchmark
#              datasets (CNN/DailyMail and XSum) via Hugging Face.
# ==============================================================================

# Exit immediately if a command exits with a non-zero status
set -e

echo "=========================================================="
echo " Starting Dataset Acquisition for BAR-Sum Benchmark Suite"
echo "=========================================================="

# 1. Ensure target storage directories exist under data/
mkdir -p data/raw/cnn_dailymail
mkdir -p data/raw/xsum
mkdir -p logs_and_results

# 2. Set Python path to project root to allow src package imports
export PYTHONPATH="${PYTHONPATH}:."

# 3. Download CNN/DailyMail dataset (version 3.0.0)
echo "[1/2] Downloading CNN/DailyMail (version 3.0.0)..."
python src/data_pipeline/download_datasets.py \
    --dataset_name "cnn_dailymail" \
    --dataset_version "3.0.0" \
    --output_dir "data/raw/cnn_dailymail" \
    2>&1 | tee logs_and_results/download_cnn_dm.log

# 4. Download XSum dataset
echo "[2/2] Downloading Extreme Summarization (XSum)..."
python src/data_pipeline/download_datasets.py \
    --dataset_name "xsum" \
    --output_dir "data/raw/xsum" \
    2>&1 | tee logs_and_results/download_xsum.log

echo "=========================================================="
echo " Dataset acquisition completed successfully."
echo " Datasets stored in: data/raw/"
echo " Execution logs saved in: logs_and_results/"
echo "=========================================================="
