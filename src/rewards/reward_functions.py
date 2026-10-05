"""
=============================================================================
Module: reward_functions.py
Project: BAR-Sum (Multi-Objective Extractive Text Summarization via DRL)
Author: Masoud Keikhosravi
Affiliation: Islamic Azad University, Mashhad Branch
Description:
    Implementation of the four objective reward functions:
    1. ROUGE Reward (Semantic Coverage / Salience)
    2. NLI Faithfulness Reward (FactCC / Entailment against Source)
    3. Redundancy Penalty Reward (Diversity / Anti-Repetition)
    4. Fluency & Coherence Reward (Sentence Ordering / Sequential Integrity)
=============================================================================
"""

from typing import Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class RewardFunctions:
    """
    Computes component reward signals for generated extractive summaries.
    """
    def __init__(
        self,
        nli_model_name: str = "roberta-large-mnli",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = torch.device(device)
        
        # 1. Initialize ROUGE Scorer (F-1 metrics)
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], 
            use_stemmer=True
        )

        # 2. Initialize NLI Model for Factuality/Entailment
        self.tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(self.device)
        self.nli_model.eval()

        # MNLI label mapping for RoBERTa: 0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT
        self.entailment_idx = 2

    # -------------------------------------------------------------------------
    # 1. ROUGE Reward (Content Salience & Coverage)
    # -------------------------------------------------------------------------
    def compute_rouge_reward(self, summary_text: str, reference_text: str) -> float:
        """
        Computes composite ROUGE score: average of ROUGE-1, ROUGE-2, and ROUGE-L F1.
        """
        if not summary_text.strip() or not reference_text.strip():
            return 0.0

        scores = self.rouge_scorer.score(reference_text, summary_text)
        r1 = scores["rouge1"].fmeasure
        r2 = scores["rouge2"].fmeasure
        rl = scores["rougeL"].fmeasure

        return float((r1 + r2 + rl) / 3.0)

    # -------------------------------------------------------------------------
    # 2. Faithfulness Reward (Factual Consistency via NLI Entailment)
    # -------------------------------------------------------------------------
    def compute_faithfulness_reward(
        self, 
        selected_sentences: List[str], 
        source_doc_text: str
    ) -> float:
        """
        Calculates NLI entailment probability between source document (Premise)
        and selected summary sentences (Hypothesis).
        """
        if not selected_sentences or not source_doc_text.strip():
            return 0.0

        summary_text = " ".join(selected_sentences)
        
        # Truncate source document to fit model sequence constraints (512 tokens)
        inputs = self.tokenizer(
            source_doc_text,
            summary_text,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.nli_model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            # Entailment probability represents factual faithfulness
            entailment_prob = probs[0, self.entailment_idx].item()

        return float(entailment_prob)

    # -------------------------------------------------------------------------
    # 3. Redundancy Reward (Penalizes Inter-Sentence Repetition)
    # -------------------------------------------------------------------------
    def compute_redundancy_reward(self, selected_sentences: List[str]) -> float:
        """
        Evaluates inter-sentence diversity. High overlap yields lower reward.
        Returns value in [0, 1], where 1 represents zero pairwise repetition.
        """
        k = len(selected_sentences)
        if k <= 1:
            return 1.0

        # Calculate word sets (Jaccard similarity between pairs)
        word_sets = [set(s.lower().split()) for s in selected_sentences]
        pair_overlaps = []

        for i in range(k):
            for j in range(i + 1, k):
                intersection = len(word_sets[i].intersection(word_sets[j]))
                union = len(word_sets[i].union(word_sets[j]))
                sim = (intersection / union) if union > 0 else 0.0
                pair_overlaps.append(sim)

        avg_overlap = np.mean(pair_overlaps) if pair_overlaps else 0.0
        # Diversity score: 1.0 - overlap
        return float(np.clip(1.0 - avg_overlap, 0.0, 1.0))

    # -------------------------------------------------------------------------
    # 4. Fluency & Coherence Reward (Sentence Ordering / Sequential Continuity)
    # -------------------------------------------------------------------------
    def compute_fluency_reward(
        self, 
        selected_indices: List[int], 
        total_doc_sentences: int
    ) -> float:
        """
        Rewards preserving natural document flow and penalizes out-of-order 
        or overly dispersed sentence extractions.
        """
        k = len(selected_indices)
        if k <= 1:
            return 1.0

        # Sort verification: checking relative sequential order
        inversions = 0
        for i in range(k):
            for j in range(i + 1, k):
                if selected_indices[i] > selected_indices[j]:
                    inversions += 1

        max_inversions = (k * (k - 1)) / 2.0
        order_score = 1.0 - (inversions / max_inversions) if max_inversions > 0 else 1.0

        # Gap regularization: Penalize excessive dispersion across huge documents
        norm_span = (max(selected_indices) - min(selected_indices)) / max(1, total_doc_sentences)
        dispersion_penalty = 0.1 * norm_span

        fluency_score = np.clip(order_score - dispersion_penalty, 0.0, 1.0)
        return float(fluency_score)

    # -------------------------------------------------------------------------
    # Composite Dispatcher
    # -------------------------------------------------------------------------
    def compute_all_rewards(
        self,
        selected_sentences: List[str],
        selected_indices: List[int],
        source_doc_text: str,
        reference_summary: str,
        total_doc_sentences: int
    ) -> Dict[str, float]:
        """
        Computes all four rewards simultaneously and returns a dictionary.
        """
        summary_text = " ".join(selected_sentences)

        r_rouge = self.compute_rouge_reward(summary_text, reference_summary)
        r_faith = self.compute_faithfulness_reward(selected_sentences, source_doc_text)
        r_redundancy = self.compute_redundancy_reward(selected_sentences)
        r_fluency = self.compute_fluency_reward(selected_indices, total_doc_sentences)

        return {
            "rouge": r_rouge,
            "faithfulness": r_faith,
            "redundancy": r_redundancy,
            "fluency": r_fluency
        }
