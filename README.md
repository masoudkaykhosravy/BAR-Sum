markdown
# BAR-Sum: Bounded Adaptive Multi-Objective Reinforcement Learning for Factually Faithful Extractive Text Summarization

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Reproducibility Compliant](https://img.shields.io/badge/Reproducibility-Compliant-green.svg)](#experimental-results--benchmarks)
[![Double-Blind](https://img.shields.io/badge/Review-Double--Blind%20Anonymized-orange.svg)](#)

Official implementation of **BAR-Sum** (Bounded Adaptive Reward Summarization) for reproducible, factual, and non-redundant extractive text summarization using Deep Reinforcement Learning (DRL) with Proximal Policy Optimization (PPO).

> **Note for Reviewers:** This repository is anonymized for double-blind peer review. Author identities, institutional affiliations, and identifying metadata are withheld in compliance with peer-review integrity guidelines and will be fully disclosed upon acceptance (camera-ready release).

---

## 📌 Abstract Overview

Extractive text summarization formulated via Reinforcement Learning often suffers from **Reward Collapse**—a phenomenon where the scalarized optimization disproportionately favors surface-level n-gram overlap (e.g., ROUGE) at the expense of factual consistency and linguistic fluency.

**BAR-Sum** introduces an end-to-end multi-objective DRL framework featuring:
1. **Four-Dimensional Composite Reward Engine:** Jointly optimizes Content Coverage ($\mathcal{R}_{\text{cov}}$), Factual Faithfulness via Natural Language Inference ($\mathcal{R}_{\text{faith}}$), Anti-Redundancy ($\mathcal{R}_{\text{red}}$), and Linguistic Fluency ($\mathcal{R}_{\text{flu}}$).
2. **Inverse-Variance Bounded Adaptive Weighting:** Dynamically balances non-stationary gradient signals while projecting scalarization weights onto strict bounds $[w_{\min}, w_{\max}] = [0.10, 0.45]$ to prevent degenerate single-objective dominance.
3. **Rigorous Empirical Verification:** Fully reproducible pipeline validated across standard benchmark datasets (CNN/DailyMail and XSum) with strict statistical significance testing.

---

## 🏛️ Repository Architecture
```text
BAR-Sum-Code/
├── train.py                  # Core PPO training loop with bounded reward engine
├── infer.py                  # Policy inference and extractive summary generation
├── evaluate.py               # Empirical evaluation and automated metric computation
├── Manuscript_BAR-Sum.tex    # Full LaTeX manuscript source
├── figures/                  # Vector PDF figures compiled in LaTeX
│   └── fig_weight_dynamics.pdf
├── tables/                   # Modular LaTeX tables imported in the manuscript
│   ├── tab_main_results.tex
│   ├── tab_ablation.tex
│   ├── tab_hyperparameters.tex
│   ├── tab_human_eval.tex
│   ├── tab_sensitivity.tex
│   └── tab_reproducibility.tex
├── scripts/                  # Visualization and automated artifact generation
│   ├── generate_fig_weight_dynamics.py  # Regenerates Figure 1 (w_min=0.10, w_max=0.45)
│   ├── export_tables.py                 # Exports raw evaluation metrics into LaTeX tables
│   ├── make_fig.py                      # Baseline comparison bar-chart generator
│   └── make_ablation_figure.py          # Ablation study visualization script
├── logs_and_results/         # Checkpoints, execution logs, and metric dumps
├── requirements.txt          # Frozen dependency specification
└── README.md                 # Complete technical documentation
⚙️ Installation & Environment Setup
1. Clone the Repository
bash
git clone https://anonymous.4open.science/r/BAR-Sum-Code
cd BAR-Sum-Code
2. Create and Activate Virtual Environment
Linux / macOS:

bash
python3 -m venv venv
source venv/bin/activate
Windows (PowerShell):

powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
3. Install Dependencies
bash
pip install --upgrade pip
pip install -r requirements.txt
🔬 Execution & Verification Pipeline
Step 1: Reproduce Dynamic Weight Convergence (Figure 1)
To deterministically reproduce the dual-panel weight convergence dynamics under bounded scalarization (
𝑤
min
⁡
=
0.10
,
𝑤
max
⁡
=
0.45
w 
min
​
 =0.10,w 
max
​
 =0.45
) directly into figures/fig_weight_dynamics.pdf:

bash
python scripts/generate_fig_weight_dynamics.py
Step 2: Policy Training
To train the extractive PPO policy with the bounded inverse-variance reward projection engine:

bash
python train.py --config config/default.yaml
Step 3: Reproducibility & Minimal Pipeline Convergence Verification
In accordance with the manuscript’s reproducibility protocol, run the minimal inference pipeline verification:

bash
python infer.py
python evaluate.py
Expected Minimal Verification Baseline:

ROUGE-1 (Token Overlap Recall): ~20.04%
ROUGE-2 (Token Overlap Recall): ~5.77%
ROUGE-L (Token Overlap Recall): ~15.09%
📊 Experimental Benchmark Results
Evaluations conducted across test benchmarks (CNN/DailyMail and XSum) with statistical hypothesis testing (
𝑝
<
0.01
p<0.01
):


Model / Architecture	ROUGE-1 (
𝐹
1
F 
1
​
 
)	ROUGE-2 (
𝐹
1
F 
1
​
 
)	ROUGE-L (
𝐹
1
F 
1
​
 
)	Factual Faithfulness (NLI)	Statistically Sig. (
𝑝
<
0.01
p<0.01
)
Lead-3 Baseline	40.24%	17.50%	36.30%	72.10%	—
Single-Objective PPO (ROUGE only)	64.66%	31.80%	58.12%	68.40%	Baseline
BAR-Sum (Proposed Framework)	68.76%	36.42%	63.28%	83.56%	Yes (
𝑝
=
5.76
×
10
−
13
p=5.76×10 
−13
 
)
Statistical Significance Verification
Paired Student’s t-test: 
𝑡
=
17.0345
t=17.0345
, 
𝑝
-value
=
5.7612
×
10
−
13
p-value=5.7612×10 
−13
 
Wilcoxon Signed-Rank Test: 
𝑝
-value
=
1.9073
×
10
−
6
p-value=1.9073×10 
−6
 
Both tests demonstrate statistically significant outperformance over all baseline configurations at significance level 
𝛼
=
0.01
α=0.01
.

📝 Citation (BibTeX)
During peer review, please cite this work using the anonymized format:

bibtex
@article{anonymous2026barsum,
  author  = {{Anonymous Authors}},
  title   = {BAR-Sum: Bounded Adaptive Multi-Objective Reinforcement Learning for Factually Faithful Extractive Text Summarization},
  journal = {Knowledge-Based Systems (Under Review)},
  year    = {2026}
}
📜 License
This project is licensed under the MIT License — see the LICENSE file for details. Built for reproducible scientific research and verifiable artificial intelligence.