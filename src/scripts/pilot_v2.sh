#!/usr/bin/env bash
set -euo pipefail
python -m nlel.experiments.run_pilot_v2 \
  --benchmark gsm8k --n-items 650 --seed 42 \
  --global-token-cap 4000 --include-controller-tokens \
  --arms tot nlel nlel:no_labeller \
  --outdir runs_v2
