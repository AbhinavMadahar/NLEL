
"""
Admissions minimal-subset reporting: produces one table (CSV)
and one figure (tokens-per-success grouped bars).

Assumptions:
- Per-run CSVs live in <runs>/ with names like "<benchmark>_<controller>_perrun.csv"
  reflecting nlel.experiments.run_experiment outputs.
- Each row contains at least: controller, budget_multiplier, correct, tokens_total.
- Optional: 'verified' boolean for verifier accept-rate.

This is deliberately simple and side-effect free.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _find_perrun_csvs(runs: Path) -> List[Path]:
    return sorted(runs.glob("*_*_perrun.csv"))

def _read_concat(csvs: List[Path]) -> pd.DataFrame:
    dfs = []
    for p in csvs:
        try:
            df = pd.read_csv(p)
            # Infer controller name from filename if missing
            if "controller" not in df.columns:
                # filename like "gsm8k_tot_perrun.csv" -> "tot"
                stem = p.stem
                parts = stem.split("_")
                if len(parts) >= 3:
                    controller = parts[-2]
                else:
                    controller = "unknown"
                df["controller"] = controller
            dfs.append(df)
        except Exception:
            continue
    if not dfs: return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    # normalize types
    if "correct" in df.columns: df["correct"] = df["correct"].astype("boolean")
    if "verified" in df.columns: df["verified"] = df["verified"].astype("boolean")
    if "budget_multiplier" in df.columns:
        df["budget_multiplier"] = pd.to_numeric(df["budget_multiplier"], errors="coerce")
    return df

def _tokens_per_success(df: pd.DataFrame) -> float:
    ok = df[df["correct"] == True]
    if "tokens_total" in ok.columns and len(ok):
        return float(ok["tokens_total"].median())
    # graceful fallback: try sum of controller+child tokens if present
    fields = [c for c in ok.columns if c.startswith("tokens_") and c.endswith("_total")]
    if fields and len(ok):
        return float(ok[fields].sum(axis=1).median())
    return float("nan")

def build_table_and_figure(runs: Path, out_table: Path, out_fig: Path) -> None:
    csvs = _find_perrun_csvs(runs)
    df = _read_concat(csvs)
    if df.empty:
        raise RuntimeError(f"No per-run CSVs found in {runs}")

    controllers = list(dict.fromkeys(df["controller"].tolist()))  # preserve file order
    budgets = sorted([b for b in df["budget_multiplier"].dropna().unique().tolist() if b in (0.5, 1.0)])

    rows = []
    for ctrl in controllers:
        for b in budgets:
            sub = df[(df["controller"] == ctrl) & (df["budget_multiplier"] == b)]
            if not len(sub): continue
            n = int(sub["correct"].notna().sum()) if "correct" in sub.columns else int(len(sub))
            acc = float(sub["correct"].mean()) if "correct" in sub.columns else float("nan")
            tps = _tokens_per_success(sub)
            var = float(sub["verified"].mean()) if "verified" in sub.columns and sub["verified"].notna().any() else float("nan")
            rows.append({
                "controller": ctrl,
                "budget_x": b,
                "n": n,
                "accuracy": acc,
                "success_at_compute": acc,  # at budget b
                "tokens_per_success": tps,
                "verifier_accept_rate": var
            })

    out_table.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_table, index=False)

    # Figure: tokens-per-success grouped bars
    if rows:
        tbl = pd.DataFrame(rows)
        ctrl_order = list(dict.fromkeys(tbl["controller"].tolist()))
        bud_order = sorted(tbl["budget_x"].unique().tolist())
        width = 0.8 / max(1, len(bud_order))
        x = np.arange(len(ctrl_order))
        fig, ax = plt.subplots(figsize=(max(6, len(ctrl_order)*1.2), 3.6))
        for i, b in enumerate(bud_order):
            y = [float(tbl[(tbl["controller"]==c) & (tbl["budget_x"]==b)]["tokens_per_success"].values[0])
                 if not tbl[(tbl["controller"]==c) & (tbl["budget_x"]==b)].empty else np.nan
                 for c in ctrl_order]
            ax.bar(x + i*width, y, width, label=f"{b:.1f}×")
        ax.set_xlabel("Controller")
        ax.set_ylabel("Tokens per success (median)")
        ax.set_xticks(x + (len(bud_order)-1)*width/2)
        ax.set_xticklabels(ctrl_order, rotation=0)
        ax.legend(title="Budget")
        fig.tight_layout()
        out_fig.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_fig, dpi=200, bbox_inches="tight")
        plt.close(fig)
