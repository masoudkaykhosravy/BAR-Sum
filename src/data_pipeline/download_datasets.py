"""
Data Preprocessing & Greedy Oracle Extractor for BAR-Sum Framework
Paper: Multi-Objective Deep Reinforcement Learning with Bounded Adaptive Rewards
Author: Masoud Keikhosravi (masoud.keikhosravi@iau.ac.ir)
"""

import os
import argparse
import yaml
import torch
from datasets import load_from_disk
from transformers import AutoTokenizer
import nltk
from rouge_score import rouge_scorer
from tqdm import tqdm

# Ensure NLTK punkt is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess and extract greedy oracle labels for DRL training.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--dataset", type=str, default="cnn_dailymail", help="Target dataset name")
    parser.add_argument("--max_doc_sentences", type=int, default=50, help="Max sentences per document")
    parser.add_argument("--max_sent_tokens", type=int, default=50, help="Max tokens per sentence")
    parser.add_argument("--budget", type=int, default=3, help="Extractive summary budget (k)")
    return parser.parse_args()


def compute_greedy_oracle(doc_sentences, reference_text, budget=3):
    """
    Computes greedy oracle extractive sentence indices maximizing ROUGE-1/2 F1 score.
    Used as ground-truth reference for supervised warm-up / baseline comparison.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    selected_indices = []
    current_summary = ""
    
    for _ in range(min(budget, len(doc_sentences))):
        best_idx = -1
        best_score = -1.0
        
        for i, sent in enumerate(doc_sentences):
            if i in selected_indices:
                continue
            
            candidate_summary = current_summary + " " + sent if current_summary else sent
            score = scorer.score(reference_text, candidate_summary)['rouge2'].fmeasure
            
            if score > best_score:
                best_score = score
                best_idx = i
                
        if best_idx != -1 and best_score > 0:
            selected_indices.append(best_idx)
            current_summary += " " + doc_sentences[best_idx]
        else:
            break
            
    return sorted(selected_indices)


def preprocess_dataset(dataset_path, output_path, tokenizer_name="bert-base-uncased", 
                       max_sentences=50, max_tokens=50, budget=3):
    print(f"[*] Loading dataset from disk: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    processed_splits = {}

    for split in ['train', 'validation', 'test']:
        if split not in dataset:
            continue
            
        print(f"\n[+] Processing split: '{split}' ({len(dataset[split])} samples)...")
        processed_data = []
        
        for item in tqdm(dataset[split], desc=f"Preprocessing {split}"):
            article_text = item.get("article", item.get("document", ""))
            summary_text = item.get("highlights", item.get("summary", ""))
            
            # Sentence segmentation
            raw_sentences = nltk.sent_tokenize(article_text.strip())[:max_sentences]
            if not raw_sentences or not summary_text.strip():
                continue
                
            # Compute Greedy Oracle Labels
            oracle_labels = compute_greedy_oracle(raw_sentences, summary_text, budget=budget)
            
            # Tokenize each sentence individually
            tokenized_sents = tokenizer(
                raw_sentences,
                padding='max_length',
                truncation=True,
                max_length=max_tokens,
                return_tensors="pt"
            )
            
            sample = {
                "input_ids": tokenized_sents["input_ids"],
                "attention_mask": tokenized_sents["attention_mask"],
                "raw_sentences": raw_sentences,
                "reference_summary": summary_text,
                "oracle_labels": oracle_labels,
                "num_sentences": len(raw_sentences)
            }
            processed_data.append(sample)

        output_file = os.path.join(output_path, f"{split}_preprocessed.pt")
        os.makedirs(output_path, exist_ok=True)
        torch.save(processed_data, output_file)
        print(f"[✔] Saved {len(processed_data)} preprocessed samples to: {output_file}")


def main():
    args = parse_args()
    
    # Load parameters from config.yaml if available
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
            data_dir = cfg.get("paths", {}).get("data_dir", "./data")
            tokenizer_name = cfg.get("model", {}).get("encoder_backbone", "bert-base-uncased")
            max_sentences = cfg.get("dataset", {}).get("max_doc_sentences", args.max_doc_sentences)
            max_tokens = cfg.get("dataset", {}).get("max_sentence_length", args.max_sent_tokens)
            budget = cfg.get("dataset", {}).get("max_summary_sentences", args.budget)
            dataset_name = cfg.get("dataset", {}).get("name", args.dataset)
    else:
        data_dir = "./data"
        tokenizer_name = "bert-base-uncased"
        max_sentences = args.max_doc_sentences
        max_tokens = args.max_sent_tokens
        budget = args.budget
        dataset_name = args.dataset

    input_dataset_dir = os.path.join(data_dir, dataset_name)
    output_processed_dir = os.path.join(data_dir, f"{dataset_name}_processed")

    preprocess_dataset(
        dataset_path=input_dataset_dir,
        output_path=output_processed_dir,
        tokenizer_name=tokenizer_name,
        max_sentences=max_sentences,
        max_tokens=max_tokens,
        budget=budget
    )
    print("\n[✔] All preprocessing completed successfully. Data is ready for PPO training.")


if __name__ == "__main__":
    main()
