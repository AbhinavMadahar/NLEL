# NLEL Experiments — Paper-Accurate Scaffold (v2)

Implements the labeller–tuner overlay (NLEL) and baselines with the seven paper-alignment fixes:
1) Distinct ToT baselines; 2) Verification wired from Π; 3) Branch quota wired from Π;
4) Retrieval respected (stub); 5) success@compute aggregation; 6) Default seeds=5;
7) Ablation switches for Λ/Ψ/trust-region/verification/quantization/random labels.


### Baselines
- `cot`, `sc_cot`, `tot`, `tot_verifier`, and **`react` (ReAct-style, tool calls stubbed offline; where allowed)**.

### Splits
- Public **test** splits by default (GSM8K, MATH subset, ARC-Challenge). For **StrategyQA**, we default to **test** split but the HF test partition lacks public labels; the runner will emit results without accuracy when gold is unavailable, matching the paper’s wording on public test splits.


## Admissions Minimal Subset – LToT-Compatible API

This repository exposes an **LToT-style admissions subset** API for running the minimal evaluation slice described in the paper:
- Methods: **ToT**, **ToT+Verifier**, **NLEL (JPE)** (same ToT selector, fixed child reasoner)
- Budgets: **1.0×** (equal compute vs ToT) and optionally **0.5×** (constrained)
- Metrics: **Accuracy/EM**, **success@compute**, **tokens‑per‑success** (primary), optional **verifier accept‑rate**

### Quickstart

```bash
pip install -r src/requirements.txt

export MODEL="hf:meta-llama/Llama-3-8B-Instruct"   # your 7–8B instruct
export DATASET="gsm8k"                              # gsm8k | math_subset | strategyqa | arc_challenge
export N=250                                        # 200–300 recommended

bash src/scripts/admissions_min.sh
```

Or invoke the Python CLI directly:

```bash
python -m nlel.contrib.admissions_min run   --dataset gsm8k   --model "$MODEL"   --n 250   --methods tot,tot+verifier,nlel   --budgets 1.0,0.5   --out runs/admissions_gsm8k

python -m nlel.contrib.admissions_min report   --runs runs/admissions_gsm8k   --table runs/admissions_gsm8k/admissions_table.csv   --fig runs/admissions_gsm8k/admissions_tokens_per_success.png
```

**Notes.** The shim preserves **the same child reasoner** and **the same ToT selector**; NLEL only adds the labeller–tuner overlay with **schema validation** and **trust‑region projection**. Metrics match the paper’s reporting (success@compute, tokens‑per‑success). fileciteturn17file0
