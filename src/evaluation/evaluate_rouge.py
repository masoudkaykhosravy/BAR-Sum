"""
BAR-Sum: Multi-Objective Deep Reinforcement Learning Extractive Summarization
Evaluation Module: Robust ROUGE-1, ROUGE-2, and ROUGE-L F1 Metrics.
Strictly adheres to standard IEEE evaluation benchmarks on CNN/DailyMail and XSum.
"""

import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer


def extract_pair(item):
    """
    Extracts prediction and reference from an item supporting various key naming conventions.
    """
    pred_keys = ["predicted_summary", "prediction", "pred", "hypothesis", "system_summary", "generated_summary"]
    ref_keys = ["reference_summary", "reference", "ref", "gold", "target", "ground_truth"]

    pred = None
    ref = None

    for k in pred_keys:
        if k in item and item[k]:
            pred = str(item[k]).strip()
            break

    for k in ref_keys:
        if k in item and item[k]:
            ref = str(item[k]).strip()
            break

    return pred, ref


def evaluate_predictions(predictions_path, output_path="./logs_and_results/rouge_results.json"):
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

    with open(predictions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Predictions file must contain a JSON list of objects.")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    r1_scores = []
    r2_scores = []
    rl_scores = []

    valid_pairs = 0
    for item in tqdm(data, desc="Evaluating ROUGE"):
        pred, ref = extract_pair(item)
        if pred and ref:
            scores = scorer.score(ref, pred)
            r1_scores.append(scores["rouge1"].fmeasure * 100)
            r2_scores.append(scores["rouge2"].fmeasure * 100)
            rl_scores.append(scores["rougeL"].fmeasure * 100)
            valid_pairs += 1

    mean_r1 = float(np.mean(r1_scores)) if r1_scores else 0.0
    mean_r2 = float(np.mean(r2_scores)) if r2_scores else 0.0
    mean_rl = float(np.mean(rl_scores)) if rl_scores else 0.0

    print("\n=======================================================")
    print(" [✔] ROUGE EVALUATION RESULTS (F1, %)")
    print("=======================================================")
    print(f" Evaluated Samples : {valid_pairs}")
    print(f" ROUGE-1 F1       : {mean_r1:.2f}%")
    print(f" ROUGE-2 F1       : {mean_r2:.2f}%")
    print(f" ROUGE-L F1       : {mean_rl:.2f}%")
    print("=======================================================")

    results = {
        "evaluated_samples": valid_pairs,
        "rouge1": round(mean_r1, 2),
        "rouge2": round(mean_r2, 2),
        "rougeL": round(mean_rl, 2),
        "individual_scores": {
            "rouge1": r1_scores,
            "rouge2": r2_scores,
            "rougeL": rl_scores,
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"\n[✔] ROUGE metrics saved to: {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ROUGE scores for BAR-Sum.")
    parser.add_argument(
        "--pred_path",
        type=str,
        default="./logs_and_results/predictions.json",
        help="Path to predictions JSON file"
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./logs_and_results/rouge_results.json",
        help="Path to save output results"
    )
    args = parser.parse_args()

    evaluate_predictions(args.pred_path, args.out_path)


if __name__ == "__main__":
    main()
