# NLEL Experiments — Paper-Accurate Scaffold (v2)

Implements the labeller–tuner overlay (NLEL) and baselines with the seven paper-alignment fixes:
1) Distinct ToT baselines; 2) Verification wired from Π; 3) Branch quota wired from Π;
4) Retrieval respected (stub); 5) success@compute aggregation; 6) Default seeds=5;
7) Ablation switches for Λ/Ψ/trust-region/verification/quantization/random labels.


### Baselines
- `cot`, `sc_cot`, `tot`, `tot_verifier`, and **`react` (ReAct-style, tool calls stubbed offline; where allowed)**.

### Splits
- Public **test** splits by default (GSM8K, MATH subset, ARC-Challenge). For **StrategyQA**, we default to **test** split but the HF test partition lacks public labels; the runner will emit results without accuracy when gold is unavailable, matching the paper’s wording on public test splits.
