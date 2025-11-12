import os, json
from typing import Optional, Dict, Any, List
import typer
from rich import print
from rich.table import Table
from ..models.base import get_model
from ..controllers.nlel import Labeller, TunerJPE, run_instance
from ..controllers.tot_baseline import run_tot, ToTParams
from ..controllers.verifier import Verifier
from ..data.loaders import get_loader
from ..utils import ensure_dir, now_ts, safe_jsonl_write, set_seed

app = typer.Typer(add_completion=False, no_args_is_help=True)

@app.command()
def main(
    benchmark: str = typer.Option(..., "--benchmark", "-b", help="gsm8k | strategyqa | arc_challenge | math_subset"),
    limit: Optional[int] = typer.Option(40, "--limit", help="Max examples to run"),
    seed: int = typer.Option(42, "--seed", help="Seed for item sampling / any RNG"),
    budget_multiplier: float = typer.Option(1.0, "--budget-multiplier", help="Budget multiplier: 0.5, 1.0, 2.0 (uses 8000 * mult)"),
    # Role-specific models
    reasoner_model: str = typer.Option(..., "--model.reasoner", help="e.g., bedrock:anthropic.claude-3-sonnet-20240229-v1:0"),
    labeller_model: str = typer.Option(..., "--model.labeller", help="e.g., bedrock:cohere.command-r"),
    tuner_model: str = typer.Option(..., "--model.tuner", help="e.g., bedrock:cohere.command-r"),
    verifier_model: Optional[str] = typer.Option(None, "--model.verifier", help="Optional; defaults to labeller model"),
    # NLEL knobs
    max_depth: int = typer.Option(3, "--max-depth"),
    max_labels: int = typer.Option(2, "--max-labels"),
    ledger_max_rows: int = typer.Option(8, "--ledger-max-rows"),
    trust_region_r: float = typer.Option(0.15, "--trust-region-r"),
    no_trust_region: bool = typer.Option(False, "--no-trust-region"),
    quantize_bits: int = typer.Option(0, "--quantize-bits", help="Per-field quantization bits for Π (0 = none)"),
    no_labeller: bool = typer.Option(False, "--no-labeller", help="Freeze Λ to default label L_def for No-Λ ablation"),
    # Output
    outdir: str = typer.Option("./runs", "--outdir"),
):
    """
    Runs a compute-constrained pilot with split-role models:
      - reasoner: strong model (e.g., Claude 3 Sonnet)
      - labeller/tuner/verifier: cost-efficient model (e.g., Command-R)
    Also runs a ToT baseline under the reasoner for side-by-side comparison.
    """
    set_seed(seed)
    # Override global guards from CLI to match preregistration
    from .. import config as _CFG
    _CFG.MAX_DEPTH = max_depth
    _CFG.LEDGER_MAX_ROWS = ledger_max_rows
    # Also update controller module-level constants
    import nlel.controllers.nlel as _NLEL_MOD
    _NLEL_MOD.MAX_DEPTH = max_depth
    _NLEL_MOD.LEDGER_MAX_ROWS = ledger_max_rows
    ensure_dir(outdir)
    ts = now_ts()

    # Instantiate models
    model_reasoner = get_model(reasoner_model)
    model_labeller = get_model(labeller_model)
    model_tuner    = get_model(tuner_model)
    model_verifier = get_model(verifier_model) if verifier_model else model_labeller

    # Controllers
    labeller = Labeller(model=model_labeller, max_labels=max_labels, frozen=no_labeller)
    tuner    = TunerJPE(model=model_tuner, trust_region_r=trust_region_r, no_trust_region=no_trust_region, quantize_bits=quantize_bits)
    verifier = Verifier(model=model_verifier)

    loader = get_loader(benchmark)
    budget_tokens = int(8000 * float(budget_multiplier))

    rows_tot: List[Dict[str, Any]] = []
    rows_nlel: List[Dict[str, Any]] = []

    # Iterate items (single seed)
    for i, item in enumerate(loader(split="test")):
        if limit is not None and i >= limit: break
        q = item["question"]; gold = item["answer"]

        # ToT baseline (reasoner-only)
        tot_params = ToTParams()
        res_tot = run_tot(task=q, gold_answer=gold, model=model_reasoner, budget_tokens=budget_tokens, params=tot_params, with_verifier=False, verifier=None)
        rows_tot.append({"id": item["id"], **res_tot})

        # NLEL (split-role; reasoner + Λ/Ψ + verifier)
        res_nlel = run_instance(task=q, gold_answer=gold, model=model_reasoner, budget_tokens=budget_tokens,
                                labeller=labeller, tuner=tuner, verifier=verifier, ignore_verifier_control=False)
        rows_nlel.append({"id": item["id"], **res_nlel})

    # Save outputs
    ensure_dir(outdir)
    run_tag = f"{benchmark}.splitrole.{ts}.seed{seed}.n{len(rows_nlel)}.bm{budget_multiplier}"
    jsonl_tot = os.path.join(outdir, f"{run_tag}.tot.jsonl")
    jsonl_nl  = os.path.join(outdir, f"{run_tag}.nlel.jsonl")
    safe_jsonl_write(jsonl_tot, rows_tot)
    safe_jsonl_write(jsonl_nl, rows_nlel)

    # Summaries
    def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len([r for r in rows if r.get("correct") is not None])
        acc = (sum(1 for r in rows if r.get("correct")) / n) if n else float('nan')
        avg_tokens = sum(r.get("tokens_total", 0) for r in rows) / max(1, len(rows))
        return {"n": n, "acc": acc, "avg_tokens": avg_tokens}

    sum_tot = summarize(rows_tot); sum_nl = summarize(rows_nlel)

    # Pretty print
    table = Table(title=f"Pilot summary: {benchmark} (split-role)")
    table.add_column("Controller", justify="left", style="bold")
    table.add_column("Examples", justify="right", style="cyan")
    table.add_column("Accuracy", justify="right", style="green")
    table.add_column("Avg tokens", justify="right", style="yellow")
    table.add_row("ToT (reasoner only)", str(sum_tot["n"]), f"{sum_tot['acc']:.3f}" if sum_tot['acc']==sum_tot['acc'] else "nan", f"{sum_tot['avg_tokens']:.1f}")
    table.add_row("NLEL (split-role)",  str(sum_nl["n"]),  f"{sum_nl['acc']:.3f}"  if sum_nl['acc']==sum_nl['acc'] else "nan",  f"{sum_nl['avg_tokens']:.1f}")
    print(table)
    print(f"[bold]ToT JSONL:[/bold] {jsonl_tot}")
    print(f"[bold]NLEL JSONL:[/bold] {jsonl_nl}")

if __name__ == "__main__":
    app()