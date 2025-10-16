
# Pilot v2 — $100 AWS subset (0.5×, paired, global token cap)

**Design.** Paired ToT vs NLEL on GSM8K at **0.5×** budget only. Enforce a **per‑arm global cap of 4,000 tokens**, which **includes controller tokens** (Λ/Ψ/verifier) for NLEL. Add a single micro‑ablation (`nlel:no_labeller`) for attribution. Sample size: **n=650** paired items (seed=42).

**Run.**
```bash
bash scripts/pilot_v2.sh
# or:
python -m nlel.experiments.run_pilot_v2   --benchmark gsm8k --n-items 650 --seed 42   --global-token-cap 4000 --include-controller-tokens   --arms tot nlel nlel:no_labeller   --outdir runs_v2
```

**Outputs.**
- `runs_v2/pilot_v2_summary.json` — accuracy (with 95% CIs), McNemar, tokens‑per‑success.
- JSONL logs per arm are kept in `runs_v2/`.

**Notes.** This orchestrator reuses the v1 split‑role runner and performs **post‑hoc enforcement** of the global token cap when computing metrics; controller tokens are aggregated from Bedrock usage fields if present.
