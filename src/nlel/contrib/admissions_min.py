
"""
Admissions-subset compatibility shim.

This CLI mirrors the LToT admissions minimal-subset API, but dispatches into
NLEL's existing engines without changing their behavior.

Two subcommands:

  1) run     – orchestrate runs across methods (controllers) and budgets.
  2) report  – materialize the required table + figure from the produced CSV/JSONL.

Example:
  python -m nlel.contrib.admissions_min run \
      --dataset gsm8k \
      --model hf:meta-llama/Llama-3-8B-Instruct \
      --n 250 \
      --methods tot,tot+verifier,nlel \
      --budgets 1.0,0.5 \
      --out runs/admissions_gsm8k

  python -m nlel.contrib.admissions_min report \
      --runs runs/admissions_gsm8k \
      --table admissions_table.csv \
      --fig admissions_tokens_per_success.png
"""
from __future__ import annotations

import os, json, math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import typer
from rich import print
from rich.table import Table

from ..experiments.run_experiment import main as _run_one  # Typer function; callable as regular def
from ..eval import metrics as _metrics

app = typer.Typer(add_completion=False)

def _norm_methods(s: str) -> List[str]:
    raw = [t.strip().lower() for t in s.replace(";",",").split(",") if t.strip()]
    out = []
    for m in raw:
        if m in ("tot+verifier","tot_verifier","totv"): out.append("tot_verifier")
        elif m in ("tot","cot","sc_cot","react","nlel"): out.append(m)
        else: raise typer.BadParameter(f"Unknown method '{m}'. Use: tot, tot+verifier, nlel (and optionally cot, sc_cot, react).")
    # Deduplicate while preserving order
    seen = set(); dedup=[]
    for m in out:
        if m not in seen:
            dedup.append(m); seen.add(m)
    return dedup

def _norm_budgets(s: str) -> List[float]:
    vals = []
    for tok in s.replace(";",",").split(","):
        tok = tok.strip()
        if not tok: continue
        try:
            vals.append(float(tok))
        except Exception:
            raise typer.BadParameter(f"Budget '{tok}' is not a float.")
    if not vals: vals = [1.0]
    return vals

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

@app.command()
def run(
    dataset: str = typer.Option(..., "--dataset", "--task", "--benchmark", help="gsm8k | math_subset | strategyqa | arc_challenge"),
    model: str = typer.Option(..., "--model", help="Model spec (e.g., hf:<name-or-path>)"),
    n: Optional[int] = typer.Option(None, "--n", "--limit", help="Number of examples (use 200–300 for admissions slice)"),
    methods: str = typer.Option("tot,tot+verifier,nlel", "--methods", "--controllers", help="Comma list: tot, tot+verifier, nlel"),
    budgets: str = typer.Option("1.0,0.5", "--budgets", help="Comma list of budget multipliers (e.g., 1.0,0.5)"),
    out: str = typer.Option("runs/admissions", "--out", "--outdir", help="Output directory (single dir for all runs)"),
    seeds: Optional[str] = typer.Option(None, "--seeds", help="Comma-separated seeds"),
    max_depth: int = typer.Option(3, "--max-depth", help="Max reasoning depth"),
    max_labels: int = typer.Option(2, "--max-labels", help="Max labels per parent (NLEL)"),
    ledger_max_rows: int = typer.Option(8, "--ledger-max-rows", help="Max rows in the in-prompt ledger (NLEL)"),
    verifier_passes: int = typer.Option(1, "--verifier-passes", help="Passes for tot+verifier"),
    verifier_strictness: float = typer.Option(0.5, "--verifier-strictness", help="Strictness for tot+verifier"),
):
    """
    Run admissions-minimal subset across methods × budgets with an LToT-compatible API.
    """
    methods_l = _norm_methods(methods)
    budgets_l = _norm_budgets(budgets)
    outdir = Path(out)
    _ensure_dir(outdir)

    # Set env guards as in the paper wrapper so underlying code picks them up.
    os.environ["NLEL_MAX_DEPTH"] = str(max_depth)
    os.environ["NLEL_LEDGER_MAX_ROWS"] = str(ledger_max_rows)
    os.environ["NLEL_MAX_LABELS"] = str(max_labels)
    os.environ["NLEL_VERIFIER_PASSES"] = str(verifier_passes)
    os.environ["NLEL_VERIFIER_STRICTNESS"] = str(verifier_strictness)

    # Pretty print plan
    table = Table(title="Admissions subset plan")
    table.add_column("Dataset"); table.add_column("Model"); table.add_column("n")
    table.add_column("Methods"); table.add_column("Budgets"); table.add_column("Out")
    table.add_row(dataset, model, str(n or "all"), ", ".join(methods_l), ", ".join(f"{b:.2f}×" for b in budgets_l), str(outdir))
    print(table)

    for controller in methods_l:
        for b in budgets_l:
            print(f"[bold]→ Running[/bold] {controller} @ {b:.2f}×")
            # Note: call the Typer-decorated function as a normal function. This bypasses CLI parsing and prints its own summary.
            _run_one(
                benchmark=dataset,
                controller=controller,
                model=model,
                limit=n,
                outdir=str(outdir),
                seeds=seeds,
                budget_multiplier=b,
                sc_samples=5,
                ablate_labeller=False,
                ablate_tuner=False,
                no_trust_region=False,
                ignore_verifier_control=False,
                quantize_controls=0,
                random_labels=False,
                report_sac=False,
            )

@app.command()
def report(
    runs: str = typer.Option(..., "--runs", "--indir", help="Directory passed to `--out` in `run`"),
    table: str = typer.Option("admissions_table.csv", "--table", "--out-table", help="CSV table path"),
    fig: str = typer.Option("admissions_tokens_per_success.png", "--fig", "--out-figure", help="Figure path (PNG)"),
):
    """
    Generate one CSV table and one figure (tokens-per-success) for the admissions subset.
    """
    from .report_minimal import build_table_and_figure
    build_table_and_figure(Path(runs), Path(table), Path(fig))
    print(f"[bold]Wrote table:[/bold] {table}")
    print(f"[bold]Wrote figure:[/bold] {fig}")

if __name__ == "__main__":
    app()
