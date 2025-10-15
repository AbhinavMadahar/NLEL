import json, os
from typing import Optional, Dict, Any, List
import typer
from rich import print
from rich.table import Table
from ..models.base import get_model
from ..controllers.cot import run_cot, run_sc_cot
from ..controllers.nlel import Labeller, TunerJPE, run_instance
from ..controllers.tot_baseline import run_tot, ToTParams
from ..controllers.react_baseline import run_react
from ..controllers.verifier import Verifier
from ..eval.evaluator import ValueEstimator
from ..data.loaders import get_loader
from ..utils import ensure_dir, now_ts, safe_jsonl_write, set_seed
from ..config import DEFAULT_SEEDS

app = typer.Typer(add_completion=False)

def make_outdir(outdir: Optional[str]) -> str:
    if outdir: ensure_dir(outdir); return outdir
    path = os.path.join("results", now_ts()); ensure_dir(path); return path

@app.command()
def main(
    benchmark: str = typer.Option(..., help="gsm8k | math_subset | strategyqa | arc_challenge"),
    controller: str = typer.Option("nlel", help="cot | sc_cot | tot | tot_verifier | react | nlel"),
    model: str = typer.Option("dummy:tiny", help="Model spec, e.g., openai:gpt-4o-mini or dummy:tiny"),
    limit: int = typer.Option(None, help="Limit number of examples"),
    outdir: str = typer.Option(None, help="Output directory"),
    seeds: str = typer.Option(None, help="Comma-separated seeds (default: five preregistered seeds)"),
    budget_multiplier: float = typer.Option(1.0, help="Compute budget multiplier vs default (1.0)"),
    sc_samples: int = typer.Option(5, help="Self-consistency samples (sc_cot)"),
    ablate_labeller: bool = typer.Option(False, "--ablate-labeller", help="Freeze Λ to L_def"),
    ablate_tuner: bool = typer.Option(False, "--ablate-tuner", help="Freeze Ψ to Π₀"),
    no_trust_region: bool = typer.Option(False, "--no-trust-region", help="Disable trust-region projection"),
    ignore_verifier_control: bool = typer.Option(False, "--ignore-verifier-control", help="Do not use Π.verify_* fields"),
    quantize_controls: int = typer.Option(0, "--quantize-controls", help="Quantize continuous Π fields to 2^bits levels"),
    random_labels: bool = typer.Option(False, "--random-labels", help="Random label strings"),
    report_sac: bool = typer.Option(False, "--report-sac", help="Run at {0.5,1.0,2.0}x budgets and write aggregate CSV")
):
    outdir = make_outdir(outdir)
    seeds_list = [int(s) for s in seeds.split(",")] if seeds else list(DEFAULT_SEEDS)
    loader = get_loader(benchmark)
    base_model = get_model(model)

    def run_one(bmult: float) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for seed in seeds_list:
            set_seed(seed)
            if controller == "cot":
                for ex in loader(split="test", subset=limit):
                    res = run_cot(ex["question"], base_model, max_tokens=256, gold_answer=ex.get("answer"))
                    rows.append({"seed": seed, "id": ex["id"], "controller": controller, "benchmark": benchmark, "tokens_total": res.get("tokens_total", 0), "correct": res.get("correct"), "final": res.get("final"), "budget_multiplier": bmult})
            elif controller == "sc_cot":
                for ex in loader(split="test", subset=limit):
                    res = run_sc_cot(ex["question"], base_model, samples=sc_samples, max_tokens=256, gold_answer=ex.get("answer"))
                    rows.append({"seed": seed, "id": ex["id"], "controller": controller, "benchmark": benchmark, "tokens_total": res.get("tokens_total", 0), "correct": res.get("correct"), "final": res.get("final"), "budget_multiplier": bmult})
            elif controller in ("tot","tot_verifier"):
                verifier = Verifier(model=base_model) if controller == "tot_verifier" else None
                for ex in loader(split="test", subset=limit):
                    res = run_tot(ex["question"], base_model, gold_answer=ex.get("answer"), with_verifier=(controller=="tot_verifier"), verifier=verifier, verifier_passes=1, verifier_strictness=0.5, budget_tokens=int(8000*bmult))
                    rows.append({"seed": seed, "id": ex["id"], "controller": controller, "benchmark": benchmark, "tokens_total": res.get("tokens_total", 0), "correct": res.get("correct"), "final": res.get("final"), "budget_multiplier": bmult})
            elif controller == "nlel":
                labeller = Labeller(model=base_model, max_labels=3, random_labels=random_labels, frozen=ablate_labeller)
                tuner = TunerJPE(model=base_model, trust_region_r=0.15, no_trust_region=no_trust_region, quantize_bits=quantize_controls, frozen=ablate_tuner)
                verifier = Verifier(model=base_model)
                for ex in loader(split="test", subset=limit):
                    res = run_instance(ex["question"], gold_answer=ex.get("answer"), model=base_model, budget_tokens=int(8000*bmult), labeller=labeller, tuner=tuner, verifier=verifier, ignore_verifier_control=ignore_verifier_control)
                    rows.append({"seed": seed, "id": ex["id"], "controller": controller, "benchmark": benchmark, "tokens_total": res.get("tokens_total", 0), "correct": res.get("correct"), "final": res.get("final"), "budget_multiplier": bmult})
            else:
            elif controller == "react":
                for ex in loader(split="test", subset=limit):
                    res = run_react(ex["question"], base_model, max_steps=6, max_tokens=256, gold_answer=ex.get("answer"))
                    rows.append({"seed": seed, "id": ex["id"], "tokens_total": res.get("tokens_total", 0), "correct": res.get("correct"), "final": res.get("final"), "budget_multiplier": bmult})
                raise ValueError(f"Unsupported controller: {controller}")
        return rows

    all_rows: List[Dict[str, Any]] = []
    if report_sac:
        for b in [0.5, 1.0, 2.0]:
            all_rows.extend(run_one(b))
    else:
        all_rows.extend(run_one(budget_multiplier))

    jsonl_path = os.path.join(outdir, f"{benchmark}_{controller}.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in all_rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from ..eval.metrics import summarize
    per_df, agg_df = summarize(all_rows)
    per_csv = os.path.join(outdir, f"{benchmark}_{controller}_perrun.csv")
    agg_csv = os.path.join(outdir, f"{benchmark}_{controller}_aggregate.csv")
    try:
        per_df.to_csv(per_csv, index=False); agg_df.to_csv(agg_csv, index=False)
    except Exception: pass

    n = len([r for r in all_rows if r.get("correct") is not None])
    acc = (sum(1 for r in all_rows if r.get("correct")) / n) if n else float('nan')
    avg_tokens = sum(r.get("tokens_total", 0) for r in all_rows) / max(1, len(all_rows))
    table = Table(title=f"Run summary: {benchmark} · {controller}")
    table.add_column("Examples", justify="right", style="cyan", no_wrap=True)
    table.add_column("Accuracy", justify="right", style="green")
    table.add_column("Avg tokens", justify="right", style="yellow")
    table.add_row(str(n), f"{acc:.3f}" if acc==acc else "nan", f"{avg_tokens:.1f}")
    print(table)
    print(f"[bold]Saved details:[/bold] {jsonl_path}")
    print(f"[bold]Per-run CSV:[/bold] {per_csv}")
    print(f"[bold]Aggregate CSV:[/bold] {agg_csv}")

if __name__ == "__main__":
    app()