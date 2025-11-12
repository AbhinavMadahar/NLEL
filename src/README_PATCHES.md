# Paper Alignment & Local-Only Backend

This package applies the following changes to your codebase:

- **Local HF backend**: `src/nlel/models/hf_local.py`, resolver updates in `src/nlel/models/base.py`.
- **Experiment wrapper**: `src/nlel/experiments/run_experiment_paper.py` exposes paper-aligned guards and success@compute.
- **Ledger tags**: Pareto/dominated tagging in `src/nlel/ledger/ledger.py` `render_block()`.
- **Offline datasets**: `HF_DATASETS_OFFLINE` honored in `src/nlel/data/loaders.py` (if present).
- **Requirements**: local-only stack in `src/requirements.txt`.

## Usage

Examples (offline possible after caching):

```bash
# Run with paper-aligned pilot guards and a success@compute cap:
python -m nlel.experiments.run_experiment_paper paper \
  --benchmark gsm8k --controller nlel --model hf:/models/your-local-model \
  --max-depth 3 --max-labels 2 --ledger-max-rows 8 \
  --budget-multiplier 0.5 --cap-tokens 4000
```

See the manuscript (§§4–6) for the experimental contract this aligns to.
