"""
BAR-Sum: End-to-End One-Click Reproducibility Orchestrator
IEEE Q1 Publication Pipeline:
1. Download Datasets -> 2. Train PPO Agent -> 3. Evaluate (ROUGE/Faith/Sig) -> 4. Plot Figures
"""

import os
import sys
import time
import json
import subprocess
from datetime import datetime

PYTHON_EXEC = sys.executable

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def run_step(step_name: str, cmd: list, required: bool = True):
    print(f"\n[*] [STAGE] Starting: {step_name}...")
    start_time = time.time()
    
    result = subprocess.run(cmd, text=True)
    elapsed = time.time() - start_time
    
    if result.returncode != 0:
        print(f"[✘] ERROR in {step_name} (Exit code: {result.returncode})")
        if required:
            print("[!] Halting execution due to critical failure.")
            sys.exit(result.returncode)
    else:
        print(f"[✔] COMPLETED: {step_name} in {elapsed:.2f} seconds.")
    return result.returncode

def main():
    start_total = time.time()
    print_header("BAR-Sum: Full End-to-End IEEE Reproducibility Pipeline")
    print(f"[*] Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[*] Python Executable: {PYTHON_EXEC}")
    print(f"[*] Working Directory: {os.getcwd()}")

    # 1. Dataset Downloading
    run_step(
        "1. Benchmark Dataset Acquisition (CNN/DM & XSum)",
        [PYTHON_EXEC, "scripts/download_datasets.py", "--dataset", "all", "--samples", "50"]
    )

    # 2. PPO Agent Training
    run_step(
        "2. Bounded Multi-Objective PPO Agent Training",
        [PYTHON_EXEC, "scripts/train.py", "--data", "data/raw/cnn_dm_test.json", "--epochs", "5"]
    )

    # 3. Full Evaluation (ROUGE + Faithfulness + Significance)
    run_step(
        "3. Evaluation Pipeline (ROUGE, Faithfulness & Significance)",
        [PYTHON_EXEC, "scripts/evaluate.py", "--faithfulness", "--significance"]
    )

    # 4. Publication Figures Generation
    run_step(
        "4. Publication-Ready Figure Generation (300 DPI / PDF)",
        [PYTHON_EXEC, "scripts/plot_results.py"]
    )

    # 5. Final Summary Table Compilation
    print_header("FINAL RESEARCH SUMMARY & REPRODUCIBILITY REPORT")
    
    rouge_path = os.path.join("logs_and_results", "rouge_results.json")
    faith_path = os.path.join("logs_and_results", "faithfulness_results.json")
    sig_path = os.path.join("logs_and_results", "significance_results.json")

    r1, r2, rl = "N/A", "N/A", "N/A"
    faith = "N/A"
    p_ttest, p_wilcox = "N/A", "N/A"

    if os.path.exists(rouge_path):
        with open(rouge_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Support both root dictionary structure and overall_metrics key
            metrics = data.get("overall_metrics", data)
            r1 = f"{metrics.get('rouge1_f1', metrics.get('rouge1', 0)):.2f}%"
            r2 = f"{metrics.get('rouge2_f1', metrics.get('rouge2', 0)):.2f}%"
            rl = f"{metrics.get('rougeL_f1', metrics.get('rougeL', 0)):.2f}%"

    if os.path.exists(faith_path):
        with open(faith_path, "r", encoding="utf-8") as f:
            faith = f"{json.load(f).get('mean_faithfulness', 0):.2f}%"

    if os.path.exists(sig_path):
        with open(sig_path, "r", encoding="utf-8") as f:
            s_data = json.load(f)
            p_ttest = f"{s_data.get('p_value_ttest', s_data.get('p_value', 0)):.4e}"
            p_wilcox = f"{s_data.get('p_value_wilcoxon', 0):.4e}"

    print(f"  +-------------------------------------------------------------+")
    print(f"  | Metric / Indicator                     | Achieved Value     |")
    print(f"  +-------------------------------------------------------------+")
    print(f"  | ROUGE-1 F1                             | {r1:<18} |")
    print(f"  | ROUGE-2 F1                             | {r2:<18} |")
    print(f"  | ROUGE-L F1                             | {rl:<18} |")
    print(f"  | Factual Faithfulness (NLI)             | {faith:<18} |")
    print(f"  | Paired t-test p-value                  | {p_ttest:<18} |")
    print(f"  | Wilcoxon Signed-Rank p-value           | {p_wilcox:<18} |")
    print(f"  | Checkpoint Path                        | checkpoints/       |")
    print(f"  | Figures Output Path                    | logs_and_results/  |")
    print(f"  +-------------------------------------------------------------+")

    total_time = time.time() - start_total
    print(f"\n[✔] ALL PIPELINE STAGES FINISHED IN {total_time:.2f} SECONDS.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
