#!/usr/bin/env bash
set -euo pipefail

# Defaults (can be overridden by env)
: "${MODEL:=hf:meta-llama/Llama-3-8B-Instruct}"
: "${DATASET:=gsm8k}"
: "${N:=250}"
: "${METHODS:=tot,tot+verifier,nlel}"
: "${BUDGETS:=1.0,0.5}"
: "${OUT:=runs/admissions_${DATASET}}"

python -m nlel.contrib.admissions_min run \
  --dataset "${DATASET}" \
  --model "${MODEL}" \
  --n "${N}" \
  --methods "${METHODS}" \
  --budgets "${BUDGETS}" \
  --out "${OUT}"

python -m nlel.contrib.admissions_min report \
  --runs "${OUT}" \
  --table "${OUT}/admissions_table.csv" \
  --fig "${OUT}/admissions_tokens_per_success.png"
