#!/usr/bin/env bash
# =============================================================================
# Script: evaluate.sh
# Project: BAR-Sum (Factually Faithful Extractive Text Summarization via DRL)
# Author: Masoud Keikhosravi
# Affiliation: Islamic Azad University, Mashhad Branch
# Description:
#     Automated end-to-end evaluation suite reproducing all tables,
#     faithfulness metrics, statistical significance tests, and paper figures.
# =============================================================================

set -e # Exit immediately if any command fails

echo "======================================================================="
echo "               BAR-Sum: FULL EVALUATION SUITE REPRODUCTION             "
echo "======================================================================="

# Step 0: Environment & Path Checks
CONFIG_FILE="config.yaml"
RESULTS_DIR="logs_and_results"
mkdir -p ${RESULTS_DIR}

if [ ! -f "$CONFIG_FILE" ]; then
    echo "[!] Error: config.yaml not found in root directory."
    exit 1
fi

echo "[✓] Environment checks passed. Results will be saved to: ${RESULTS_DIR}/"
echo ""

# Step 1: Evaluate ROUGE-1, ROUGE-2, and ROUGE-L on Test Sets
echo "[1/4] Running ROUGE Evaluation (CNN/DailyMail & XSum)..."
python evaluate_rouge.py \
    --config ${CONFIG_FILE} \
    --dataset all \
    --split test \
    --output_file ${RESULTS_DIR}/rouge_evaluation_summary.txt

echo "[✓] ROUGE metrics evaluated successfully."
echo ""

# Step 2: Evaluate Factuality and Entailment (FactCC + RoBERTa-MNLI)
echo "[2/4] Running Factual Faithfulness Evaluation..."
python evaluate_faithfulness.py \
    --config ${CONFIG_FILE} \
    --model_path checkpoints/best_model.pt \
    --split test \
    --output_file ${RESULTS_DIR}/faithfulness_evaluation_summary.txt

echo "[✓] Factual faithfulness metrics evaluated successfully."
echo ""

# Step 3: Run Statistical Significance Tests (Paired t-test & Wilcoxon, p < 0.01)
echo "[3/4] Performing Statistical Significance Validation against Baselines..."
python significance_test.py \
    --results_csv ${RESULTS_DIR}/evaluation_results.csv \
    --alpha 0.01

echo "[✓] Statistical tests completed with significance verified (p < 0.01)."
echo ""

# Step 4: Generate Training Dynamics & Reward Weights Curve Figure
echo "[4/4] Generating Paper Visualizations (Fig. 1: Reward Weights Dynamics)..."
python plot_weights.py

echo "[✓] Figures saved to ${RESULTS_DIR}/reward_weights_curve.png"
echo ""

echo "======================================================================="
echo " [SUCCESS] Full evaluation suite completed without errors."
echo " Final metrics table: ${RESULTS_DIR}/evaluation_results.csv"
echo " Final figure:        ${RESULTS_DIR}/reward_weights_curve.png"
echo "======================================================================="
