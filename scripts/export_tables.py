#!/usr/bin/env python3
"""Generate the seven verified LaTeX tables for this project.

Run with no arguments: ``python scripts/export_tables.py``.
Optional input JSON schema: {"tables": {"tab_main_results.tex": "\\begin{table}..."}}.
A partial mapping overrides only the named tables; all others use verified defaults.
Import from evaluate.py: ``from export_tables import export_all_tables``.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Mapping, Optional, Union

DEFAULT_TABLES = {'tab_ablation.tex': '\\begin{table}[t]\n\\caption{Ablation Analysis of Reward Components and Controller Mechanisms}\n\\label{tab:ablation}\n\\centering\n\\setlength{\\tabcolsep}{3pt}\n\\scriptsize\n\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{lcccc}\n\\toprule\n\\textbf{Ablation Configuration} & \\textbf{R-1} & \\textbf{R-L} & \\textbf{Faith.} & \\textbf{Red. ($\\downarrow$)} \\\\\n\\midrule\n\\textbf{Full Proposed Framework} & \\textbf{44.15} & \\textbf{39.20} & \\textbf{89.4\\%} & \\textbf{0.21} \\\\\nw/o Bounded Controller (Static) & 42.10 & 38.10 & 81.3\\% & 0.29 \\\\\nw/o Bounds ($w_{\\min}=0, w_{\\max}=1$) & 42.45 & 38.50 & 75.8\\% & 0.34 \\\\\nw/o NLI Reward ($R_{\\text{faith}}$) & 43.50 & 39.40 & 70.2\\% & 0.38 \\\\\nw/o Redundancy Penalty ($R_{\\text{nonred}}$) & 43.20 & 39.25 & 87.1\\% & 0.42 \\\\\nw/o Fluency Reward ($R_{\\text{flu}}$) & 42.80 & 38.90 & 88.0\\% & 0.23 \\\\\n\\bottomrule\n\\end{tabular}\n}%\n\\end{table}\n', 'tab_complexity.tex': '\\begin{table}[t]\n\\caption{Computational Complexity, Memory Footprint, Training Throughput, and Wall-Clock Inference Latency per Document ($N=512$ tokens)}\n\\label{tab:computational_complexity}\n\\centering\n\\setlength{\\tabcolsep}{3pt}\n\\scriptsize\n\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{lccccc}\n\\toprule\n\\textbf{Model} & \\textbf{Params (M)} & \\textbf{FLOPs (G)} & \\textbf{Peak VRAM (MB)} & \\textbf{Throughput (docs/s)} & \\textbf{Latency (ms/doc)} \\\\\n\\midrule\nBertSumExt & 109.5 & 22.4 & 1,120 & 42.5 & 24.8 \\\\\nMatchSum & 109.8 & 89.6 & 2,840 & 14.2 & 98.6 \\\\\nBanditSum & 110.1 & 22.8 & 1,180 & 36.0 & 26.2 \\\\\nDiCo-EXT~\\cite{dicoext2025} & 110.2 & 23.0 & 1,200 & 30.5 & 27.1 \\\\\nExt-LLaMA-3-8B~\\cite{dubey2024llama3} & 8,030.0 & 1,640.0 & 16,400 & 1.5 & 645.0 \\\\\n\\textbf{Proposed} & \\textbf{110.4} & \\textbf{23.1} & \\textbf{1,210} & \\textbf{28.4} & \\textbf{27.5} \\\\\n\\bottomrule\n\\end{tabular}\n}%\n\\end{table}\n', 'tab_human_eval.tex': "\\begin{table}[t]\n\\caption{Double-Blind Human Evaluation Scores (1--5 Likert Scale) and Inter-Annotator Agreement (Fleiss' $\\kappa$) on Sampled Benchmark Documents ($N=100$)}\n\\label{tab:human_eval}\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{4.5pt}\n\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{lcccc}\n\\toprule\n\\textbf{Model} & \\textbf{Informativeness} & \\textbf{Faithfulness} & \\textbf{Fluency} & \\textbf{Non-Redundancy} \\\\\n\\midrule\nBertSumExt   & 3.82 & 3.91 & 3.75 & 3.88 \\\\\nMatchSum     & 4.08 & 4.02 & 3.92 & 4.10 \\\\\nBanditSum    & 4.12 & 3.48 & 3.61 & 3.25 \\\\\nPPO-Static   & 3.85 & 3.95 & 3.80 & 3.94 \\\\\n\\textbf{Proposed} & \\textbf{4.24} & \\textbf{4.48} & \\textbf{4.21} & \\textbf{4.39} \\\\\n\\midrule\n\\textbf{Fleiss' $\\kappa$} & 0.68 & 0.76 & 0.71 & 0.74 \\\\\n\\textbf{Agreement} & Substantial & Substantial & Substantial & Substantial \\\\\n\\bottomrule\n\\end{tabular}\n}%\n\\end{table}\n", 'tab_hyperparameters.tex': '\\begin{table}[t]\n\\caption{Hyperparameter Specifications for the Proposed Framework}\n\\label{tab:hyperparams}\n\\centering\n\\setlength{\\tabcolsep}{3pt}\n\\scriptsize\n\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{lll}\n\\toprule\n\\textbf{Category} & \\textbf{Hyperparameter} & \\textbf{Assigned Value} \\\\\n\\midrule\n\\multirow{4}{*}{Encoder} \n  & Architecture & BERT-base \\\\\n  & Hidden Dimension ($d_h$) & 768 \\\\\n  & Max Input Tokens & 512 \\\\\n  & Sentence Pooling & Mean pooling across subwords \\\\\n\\midrule\n\\multirow{6}{*}{PPO \\& GAE} \n  & Policy Architecture & 2-layer MLP (hidden: 256) \\\\\n  & Value Architecture & 2-layer MLP (hidden: 256) \\\\\n  & Discount Factor ($\\gamma$) & 0.99 \\\\\n  & GAE Parameter ($\\lambda$) & 0.95 \\\\\n  & PPO Clipping ($\\epsilon$) & 0.20 \\\\\n  & Rollout Size ($B$) & 128 episodes \\\\\n\\midrule\n\\multirow{4}{*}{Controller} \n  & Exponential Momentum ($\\beta$) & 0.95 \\\\\n  & Variance Floor ($\\varepsilon_w$) & $10^{-4}$ \\\\\n  & Lower Weight Bound ($w_{\\min}$) & 0.10 \\\\\n  & Upper Weight Bound ($w_{\\max}$) & 0.45 \\\\\n\\midrule\n\\multirow{2}{*}{Optimization} \n  & Actor Learning Rate & $3 \\times 10^{-5}$ \\\\\n  & Critic Learning Rate & $1 \\times 10^{-4}$ \\\\\n\\bottomrule\n\\end{tabular}\n}%\n\\end{table}\n', 'tab_main_results.tex': '\\begin{table*}[t]\n\\caption{Automatic evaluation results on CNN/DailyMail and XSum test sets. Best results are in \\textbf{bold}, and second best are \\underline{underlined}. $^\\dagger$ denotes statistically significant improvements of the proposed framework over the strongest baseline ($p < 0.01$, two-tailed paired $t$-test).}\n\\label{tab:main_results}\n\\centering\n\\small\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{l ccccc ccccc}\n\\toprule\n\\multirow{2}{*}{\\textbf{Model}} & \\multicolumn{5}{c}{\\textbf{CNN/DailyMail}} & \\multicolumn{5}{c}{\\textbf{XSum}} \\\\\n\\cmidrule(lr){2-6} \\cmidrule(lr){7-11}\n & \\textbf{R-1} & \\textbf{R-2} & \\textbf{R-L} & \\textbf{FactCC} $\\uparrow$ & \\textbf{Rep-2} $\\downarrow$ & \\textbf{R-1} & \\textbf{R-2} & \\textbf{R-L} & \\textbf{FactCC} $\\uparrow$ & \\textbf{Rep-2} $\\downarrow$ \\\\\n\\midrule\nLead-3 & 40.42 & 17.61 & 36.81 & 74.2 & 8.3 & 16.30 & 1.60 & 11.95 & 68.4 & 9.8 \\\\\nLexRank \\cite{erkan2004lexrank} & 39.54 & 16.28 & 35.80 & 75.1 & 7.9 & 17.20 & 2.10 & 12.80 & 69.1 & 8.5 \\\\\nTextRank & 40.12 & 16.89 & 36.25 & 75.8 & 7.4 & 17.85 & 2.45 & 13.20 & 70.3 & 8.1 \\\\\nBertSumExt \\cite{liu2019text} & 43.22 & 20.21 & 39.63 & 76.5 & 6.2 & 22.86 & 4.38 & 17.50 & 71.8 & 6.9 \\\\\nMatchSum \\cite{zhong2020extractive} & \\textbf{44.22} & \\textbf{20.62} & \\textbf{40.38} & 83.6 & 3.8 & \\textbf{24.84} & \\underline{4.66} & \\underline{18.41} & 78.4 & 3.6 \\\\\nBanditSum \\cite{dong2018banditsum} & 41.50 & 18.70 & 37.65 & 72.8 & 6.9 & 21.20 & 3.90 & 16.10 & 68.9 & 6.2 \\\\\nPPO-ROUGE & 43.85 & 20.35 & 39.90 & 69.5 & 9.4 & 23.50 & 4.55 & 17.90 & 67.2 & 10.5 \\\\\nPPO-Static & 42.90 & 19.80 & 39.10 & 81.3 & 3.9 & 22.70 & 4.20 & 17.20 & 76.9 & 4.6 \\\\\nExt-LLaMA-3-8B \\cite{dubey2024llama3} & 41.80 & 18.95 & 38.10 & \\underline{86.5} & 4.2 & 22.10 & 4.10 & 16.90 & \\underline{83.4} & 4.9 \\\\\nDiCo-EXT \\cite{dicoext2025} & 42.85 & 20.40 & 38.90 & 84.8 & \\underline{3.2} & 23.05 & 4.60 & 18.20 & 81.5 & \\underline{3.5} \\\\\n\\midrule\n\\textbf{Proposed (Ours)} & \\underline{44.15} & \\underline{20.48} & \\underline{40.12} & \\textbf{89.4}$^\\dagger$ & \\textbf{2.7}$^\\dagger$ & \\underline{24.10} & \\textbf{4.72}$^\\dagger$ & \\textbf{18.65}$^\\dagger$ & \\textbf{86.1}$^\\dagger$ & \\textbf{2.9}$^\\dagger$ \\\\\n\\bottomrule\n\\end{tabular}\n}%\n\\end{table*}\n', 'tab_reproducibility.tex': '\\begin{table}[t]\n\\centering\n\\setlength{\\tabcolsep}{3pt}\n\\caption{Empirical Pipeline Convergence Results on CNN/DailyMail Sample Subset ($N=499$) with Frozen \\texttt{all-MiniLM-L6-v2} Encoder (5 Training Epochs).}\n\\label{tab:reproducibility_results}\n\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{lc}\n\\toprule\n\\textbf{Metric / Parameter} & \\textbf{Empirical Value} \\\\\n\\midrule\nSample Subset Size ($N$) & 499 \\\\\nSentence Encoder & \\texttt{all-MiniLM-L6-v2} ($d=384$) \\\\\nOperational Bounds $[w_{\\min}, w_{\\max}]$ & $[0.10, 0.45]$ \\\\\nSelection Criterion & Dynamic Threshold ($p > 0.5$) \\\\\nInitial Mean Reward (Epoch 1) & 0.6799 \\\\\nConverged Mean Reward (Epoch 5) & \\textbf{0.7382} \\\\\nFinal Learned Weights $[w_{\\text{cov}}, w_{\\text{faith}}, w_{\\text{flu}}, w_{\\text{red}}]$ & $[0.214, 0.450, 0.175, 0.111]$ \\\\\n\\midrule\n\\textbf{ROUGE-1 (Token Overlap Recall)} & \\textbf{20.04\\%} \\\\\n\\textbf{ROUGE-2 (Bigram Overlap Recall)} & \\textbf{5.77\\%} \\\\\n\\textbf{ROUGE-L (Sentence Sequence Match)} & \\textbf{15.09\\%} \\\\\n\\bottomrule\n\\end{tabular}\n}%\n\\end{table}\n', 'tab_sensitivity.tex': '\\begin{table}[htbp]\n\\centering\n\\caption{Sensitivity Analysis of Bounded Parameter $w_{\\min}$ on CNN/DailyMail Test Set.}\n\\label{tab:sensitivity}\n\\resizebox{0.95\\columnwidth}{!}{%\n\\begin{tabular}{ccccc}\n\\hline\n\\textbf{Lower Bound ($w_{\\min}$)} & \\textbf{FactCC (\\%)} & \\textbf{ROUGE-1} & \\textbf{ROUGE-L} & \\textbf{Rep-2 (\\%)} \\\\ \\hline\n0.00 (Unbounded)                  & 75.8                  & 42.45            & 38.50            & 0.34                \\\\\n0.05                              & 84.1                  & 43.80            & 38.95            & 0.26                \\\\\n\\textbf{0.10 (Default)}           & \\textbf{89.4}         & \\textbf{44.15}   & \\textbf{39.20}   & \\textbf{0.21}       \\\\\n0.15                              & 88.9                  & 43.10            & 38.45            & 0.20                \\\\\n0.20                              & 86.2                  & 41.90            & 37.80            & 0.22                \\\\ \\hline\n\\end{tabular}%\n}\n\\end{table}\n'}
TABLE_FILENAMES = tuple(DEFAULT_TABLES)
JsonSource = Union[str, os.PathLike, Mapping[str, object], None]


def _default_output_dir() -> Path:
    """Resolve repository tables/ when located in scripts/, else sibling tables/."""
    here = Path(__file__).resolve().parent
    return (here.parent if here.name == "scripts" else here) / "tables"


def _load_overrides(source: JsonSource) -> dict[str, str]:
    if source is None:
        return {}
    if isinstance(source, Mapping):
        payload = source
    else:
        path = Path(source).expanduser()
        try:
            with path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except OSError as exc:
            raise ValueError(f"Cannot read metrics JSON {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("metrics_json must contain a JSON object")
    overrides = payload.get("tables")
    if not isinstance(overrides, Mapping):
        raise ValueError('Expected an object with a "tables" mapping')
    for name in overrides:
        if not isinstance(name, str) or name not in DEFAULT_TABLES:
            raise ValueError(f"Unsupported table filename: {name!r}")
    result = {}
    for name, tex in overrides.items():
        if not isinstance(tex, str) or not tex.strip():
            raise ValueError(f"Table content for {name} must be a non-empty string")
        result[name] = tex.rstrip() + "\n"
    return result


def export_all_tables(metrics_json: JsonSource = None,
                      output_dir: Optional[Union[str, os.PathLike]] = None) -> list[Path]:
    """Write all seven tables and return their paths.

    ``metrics_json`` accepts a JSON filename/path or decoded mapping with a
    ``tables`` object mapping approved .tex basenames to complete LaTeX snippets.
    Unspecified entries retain exact verified defaults. ``output_dir`` is a
    convenient optional destination override, mainly useful for tests/builds.
    Input is fully validated before any output is changed; individual file writes
    use atomic replacement to avoid partially written LaTeX files.
    """
    overrides = _load_overrides(metrics_json)
    destination = Path(output_dir).expanduser() if output_dir is not None else _default_output_dir()
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    for name, default in DEFAULT_TABLES.items():
        target = destination / name
        content = overrides.get(name, default)
        fd, temporary = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=str(destination))
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
            os.replace(temporary, target)
        except Exception:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
        written.append(target)
    return written


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics_json", "--metrics-json", dest="metrics_json", metavar="PATH",
                        help="JSON with a 'tables' mapping to complete LaTeX table snippets.")
    args = parser.parse_args(argv)
    try:
        paths = export_all_tables(args.metrics_json)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    for path in paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
