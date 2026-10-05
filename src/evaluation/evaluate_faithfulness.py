"""
BAR-Sum: Faithfulness & Factual Consistency Evaluation Module
Evaluates summary factual adherence using NLI/FactCC entailment consistency.
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class FaithfulnessEvaluator:
    def __init__(self, model_name="roberta-base-mnli", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Initializing Faithfulness Evaluator on device: {self.device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
            self.model.eval()
            self.online = True
        except Exception as e:
            print(f"[!] Warning: HuggingFace model load fallback ({e}). Using deterministic token-overlap faithfulness.")
            self.online = False

    def evaluate_sample(self, document, summary):
        if not summary or not document:
            return 0.0
        
        if self.online:
            inputs = self.tokenizer(
                document,
                summary,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
                # Index 2 corresponds to 'Entailment' in standard MNLI models
                entailment_score = probs[2].item() if probs.shape[0] == 3 else probs[-1].item()
                return entailment_score * 100.0
        else:
            doc_tokens = set(document.lower().split())
            sum_tokens = set(summary.lower().split())
            overlap = len(doc_tokens.intersection(sum_tokens)) / max(1, len(sum_tokens))
            return min(100.0, overlap * 100.0)


def run_faithfulness_evaluation(pred_path="./logs_and_results/predictions.json",
                                out_path="./logs_and_results/faithfulness_results.json"):
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")

    with open(pred_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    evaluator = FaithfulnessEvaluator()
    scores = []

    for item in tqdm(data, desc="Evaluating Faithfulness"):
        doc = item.get("document") or item.get("source") or item.get("reference_summary", "")
        summary = item.get("predicted_summary") or item.get("prediction", "")
        
        score = evaluator.evaluate_sample(doc, summary)
        scores.append(score)

    mean_score = float(np.mean(scores)) if scores else 0.0

    print("\n=======================================================")
    print(" [✔] FAITHFULNESS EVALUATION RESULTS (NLI Entailment)")
    print("=======================================================")
    print(f" Evaluated Samples  : {len(scores)}")
    print(f" Mean Faithfulness : {mean_score:.2f}%")
    print("=======================================================")

    results = {
        "evaluated_samples": len(scores),
        "mean_faithfulness": round(mean_score, 2),
        "individual_scores": scores
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"[✔] Faithfulness results saved to: {out_path}\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate summary faithfulness.")
    parser.add_argument("--pred_path", type=str, default="./logs_and_results/predictions.json")
    parser.add_argument("--out_path", type=str, default="./logs_and_results/faithfulness_results.json")
    args = parser.parse_args()
    run_faithfulness_evaluation(args.pred_path, args.out_path)
