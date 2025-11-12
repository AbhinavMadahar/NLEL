
"""
Paper-aligned wrapper: exposes guard knobs (max_depth, max_labels, ledger_max_rows),
verifier controls, and success@compute cap reporting.
"""
import os, json
import typer
from rich.table import Table
from .run_experiment import main as _orig_main  # type: ignore

app = typer.Typer(add_completion=False)

@app.callback()
def _():
    pass

@app.command("paper")
def main_paper(
    benchmark: str = typer.Option(..., help="gsm8k | math_subset | strategyqa | arc_challenge"),
    controller: str = typer.Option("nlel", help="cot | sc_cot | tot | tot_verifier | react | nlel"),
    model: str = typer.Option("dummy:tiny", help="Model spec, e.g., hf:/path/to/model or dummy:tiny"),
    limit: int = typer.Option(None, help="Limit number of examples"),
    outdir: str = typer.Option(None, help="Output directory"),
    seeds: str = typer.Option(None, help="Comma-separated seeds"),
    budget_multiplier: float = typer.Option(1.0, help="Compute budget multiplier vs default (1.0)"),
    sc_samples: int = typer.Option(5, help="Self-consistency samples (sc_cot)"),
    # Guards
    max_depth: int = typer.Option(3, help="Pilot uses 3"),
    max_labels: int = typer.Option(2, help="Pilot uses 2"),
    ledger_max_rows: int = typer.Option(8, help="Pilot uses 8"),
    # Verifier
    verifier_passes: int = typer.Option(1, help="Verification passes for tot_verifier"),
    verifier_strictness: float = typer.Option(0.5, help="Verification strictness for tot_verifier"),
    # Success@compute cap
    cap_tokens: int = typer.Option(4000, help="Report success@compute at this token cap; 0 disables"),
):
    # Set env guards so underlying modules can pick them up
    os.environ["NLEL_MAX_DEPTH"] = str(max_depth)
    os.environ["NLEL_LEDGER_MAX_ROWS"] = str(ledger_max_rows)
    os.environ["NLEL_MAX_LABELS"] = str(max_labels)
    os.environ["NLEL_VERIFIER_PASSES"] = str(verifier_passes)
    os.environ["NLEL_VERIFIER_STRICTNESS"] = str(verifier_strictness)
    os.environ["NLEL_CAP_TOKENS"] = str(cap_tokens)
    # Delegate to original main (which will still write CSV/JSONL)
    _orig_main(
        benchmark=benchmark, controller=controller, model=model, limit=limit,
        outdir=outdir, seeds=seeds, budget_multiplier=budget_multiplier, sc_samples=sc_samples
    )
    # If a summary JSON exists, augment success@compute if possible (soft best-effort).
    try:
        if outdir and os.path.isdir(outdir):
            cand = [f for f in os.listdir(outdir) if f.endswith("_summary.json")]
            if cand:
                path = os.path.join(outdir, cand[-1])
                with open(path, "r", encoding="utf-8") as f:
                    summ = json.load(f)
                summ["guards"] = {"max_depth": max_depth, "max_labels": max_labels, "ledger_max_rows": ledger_max_rows}
                summ["verifier"] = {"passes": verifier_passes, "strictness": verifier_strictness}
                summ["cap_tokens"] = cap_tokens
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(summ, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

if __name__ == "__main__":
    app()
