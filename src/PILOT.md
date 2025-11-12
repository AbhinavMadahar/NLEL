# Split-Role Pilot on AWS Bedrock

This pilot runs Tree-of-Thought (ToT) vs NLEL with split-role models:
- **Reasoner**: Anthropic Claude 3 Sonnet (`anthropic.claude-3-sonnet-20240229-v1:0`).
- **Labeller (Λ)**, **Tuner (Ψ)**, **Verifier**: Cohere Command-R (`cohere.command-r`).

## Requirements
- Python 3.10+
- `boto3` configured with AWS credentials and region (e.g., `AWS_REGION=us-east-1`).
- Bedrock model access enabled for the chosen models in your account.

## Run (GSM8K, n=40, seed=42, 1.0× budget)
```bash
python -m nlel.experiments.run_experiment_splitrole \
  --benchmark gsm8k --limit 40 --seed 42 --budget-multiplier 1.0 \
  --model.reasoner bedrock:anthropic.claude-3-sonnet-20240229-v1:0 \
  --model.labeller bedrock:cohere.command-r \
  --model.tuner    bedrock:cohere.command-r \
  --model.verifier bedrock:cohere.command-r \
  --max-depth 3 --max-labels 2 --ledger-max-rows 8 \
  --trust-region-r 0.15 --outdir ./runs
```

Outputs: two JSONL files (ToT and NLEL) and a terminal summary.

> **Update:** See `PILOT_V2.md` for the $100 pilot design at 0.5× with a global token cap and a micro‑ablation.
