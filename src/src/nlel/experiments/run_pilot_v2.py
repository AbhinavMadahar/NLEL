
import subprocess, sys, os, json
from pathlib import Path
import typer
from typing import Optional, List
from ..eval.postprocess_pilot_v2.py import main as postprocess_main  # type: ignore

app = typer.Typer(add_completion=False, no_args_is_help=True)

@app.command()
def main(
    benchmark: str = typer.Option("gsm8k", "--benchmark", help="Benchmark name (gsm8k only for pilot v2)"),
    n_items: int = typer.Option(650, "--n-items", min=1),
    seed: int = typer.Option(42, "--seed"),
    global_token_cap: int = typer.Option(4000, "--global-token-cap", help="Per-arm cap in tokens at 0.5x; includes controllers for NLEL."),
    outdir: Path = typer.Option(Path("./runs_v2"), "--outdir", help="Output directory"),
    include_controller_tokens: bool = typer.Option(True, "--include-controller-tokens/--no-include-controller-tokens"),
    arms: List[str] = typer.Option(["tot","nlel","nlel:no_labeller"], "--arms", help="Controllers to run via the v1 runner; 'nlel:no_labeller' freezes Λ."),
):
    """
    Orchestrates the $100 pilot v2:
      - Runs the original split-role pilot entry point for each requested arm.
      - Enforces 0.5x via a global token cap in post-processing (4k tokens per arm).
      - Produces a paired summary with McNemar's test and bootstrap CIs.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    # Map arm -> JSONL path
    results = {}
    for arm in arms:
        controller = arm
        v1_cmd = [
            sys.executable, "-m", "nlel.experiments.run_experiment_splitrole",
            "--benchmark", benchmark, "--limit", str(n_items), "--seed", str(seed),
            "--budget-multiplier", "0.5",
            "--max-depth","3","--max-labels","2","--ledger-max-rows","8","--trust-region-r","0.15",
            "--outdir", str(outdir)
        ]
        if arm.startswith("nlel"):
            v1_cmd += ["--controller", "nlel"]
            if ":no_labeller" in arm:
                v1_cmd += ["--no-labeller"]  # expected ablation switch in v1
        else:
            v1_cmd += ["--controller", "tot"]
        print(">>> Running:", " ".join(v1_cmd))
        subprocess.run(v1_cmd, check=True)
        # The v1 runner should announce the JSONL paths; by convention we save to:
        jsonl_path = outdir / f"{benchmark}_{arm}_seed{seed}.jsonl"
        if not jsonl_path.exists():
            # Try fallback names
            for p in outdir.glob("*.jsonl"):
                if arm in p.name and benchmark in p.name:
                    jsonl_path = p; break
        results[arm] = jsonl_path
    # Post-process ToT vs NLEL under global cap
    tot = results.get("tot")
    nlel = results.get("nlel") or results.get("nlel:no_labeller")
    if not (tot and nlel and tot.exists() and nlel.exists()):
        raise SystemExit(f"Missing JSONL outputs. Found: {results}")
    summary_json = outdir / "pilot_v2_summary.json"
    from nlel.eval.postprocess_pilot_v2 import main as post_main
    post_main(tot, nlel, cap_tokens=global_token_cap, out_json=summary_json)
    print(f"[pilot_v2] summary written to {summary_json}")

if __name__ == "__main__":
    app()
