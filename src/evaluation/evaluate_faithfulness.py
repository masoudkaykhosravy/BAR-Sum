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
    def __init__(self, model_name="roberta-large-mnli", device=None):
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


def run_faithfulness_evaluation(
    pred_path="./logs_and_results/predictions.json",
    dataset_path="./data/raw/cnn_dm_test.json",
    out_path="./logs_and_results/faithfulness_results.json",
    max_doc_sents=40
):
    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Predictions file not found: {pred_path}")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with open(pred_path, "r", encoding="utf-8") as f:
        preds = json.load(f)

    with open(dataset_path, "r", encoding="utf-8") as f:
        ds = json.load(f)

    # Build id -> document_text map
    doc_map = {}
    for ex in ds:
        ex_id = ex.get("id")
        sents = ex.get("sentences", [])
        if isinstance(sents, list):
            sents = sents[:max_doc_sents]  # keep NLI premise bounded
            doc_text = " ".join([str(x).strip() for x in sents if str(x).strip()])
        else:
            doc_text = str(sents)
        doc_map[ex_id] = doc_text

    evaluator = FaithfulnessEvaluator()
    scores = []
    used = 0
    skipped = 0
    missing_doc = 0

    for item in tqdm(preds, desc="Evaluating Faithfulness (Doc->Pred)"):
        ex_id = item.get("id")
        summary = item.get("prediction", "")

        doc = doc_map.get(ex_id, "")
        if not doc:
            missing_doc += 1
            skipped += 1
            continue
        if not summary:
            skipped += 1
            continue

        score = evaluator.evaluate_sample(doc, summary)
        scores.append(score)
        used += 1

    mean_score = float(np.mean(scores)) if scores else 0.0

    print("\n=======================================================")
    print(" [✔] FAITHFULNESS (NLI Entailment) — Document ⇒ Prediction")
    print("=======================================================")
    print(f" Total predictions : {len(preds)}")
    print(f" Used samples      : {used}")
    print(f" Skipped samples   : {skipped} (missing_doc={missing_doc})")
    print(f" Mean Faithfulness : {mean_score:.2f}%")
    print("=======================================================")

    results = {
        "total_predictions": len(preds),
        "used_samples": used,
        "skipped_samples": skipped,
        "missing_doc": missing_doc,
        "mean_faithfulness": round(mean_score, 2),
        "individual_scores": scores
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"[✔] Faithfulness results saved to: {out_path}\n")
    return results

if __name__ == "__main__":
    # ابتدا parser را تعریف می‌کنیم
    parser = argparse.ArgumentParser(description="Evaluate summary faithfulness.")
    
    # سپس آرگومان‌ها را اضافه می‌کنیم
    parser.add_argument("--pred_path", type=str, default="./logs_and_results/predictions.json")
    parser.add_argument("--dataset_path", type=str, default="./data/raw/cnn_dm_test.json")
    parser.add_argument("--out_path", type=str, default="./logs_and_results/faithfulness_results.json")
    parser.add_argument("--max_doc_sents", type=int, default=40)
    
    # در نهایت پارامترها را می‌خوانیم و تابع اصلی را صدا می‌زنیم
    args = parser.parse_args()
    
    run_faithfulness_evaluation(
        pred_path=args.pred_path,
        dataset_path=args.dataset_path,
        out_path=args.out_path,
        max_doc_sents=args.max_doc_sents
    )


