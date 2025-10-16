
import json, sys, math
from typing import Dict, Any, List, Tuple
from pathlib import Path
import numpy as np
from .mcnemar import mcnemar
from .bootstrap import bootstrap_ci

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

TOKEN_KEYS = [
    "tokens_total", "total_tokens", "tokens",  # generic
    "tokens_child_total", "tokens_child", "child_tokens",
    "tokens_controller_total", "controller_tokens",
    "prompt_tokens", "completion_tokens"
]

def approx_total_tokens(row: Dict[str, Any]) -> int:
    # Try common keys
    for k in TOKEN_KEYS:
        if k in row and isinstance(row[k], (int, float)):
            return int(row[k])
    # Sum usage fields if present
    total = 0
    if "usage" in row and isinstance(row["usage"], dict):
        u = row["usage"]
        total += int(u.get("prompt_tokens", 0)) + int(u.get("completion_tokens", 0))
    # Sum nested calls usage if present
    for subk in ("calls", "trace", "log"):
        v = row.get(subk, None)
        if isinstance(v, list):
            for ev in v:
                u = ev.get("usage", {})
                total += int(u.get("prompt_tokens", 0)) + int(u.get("completion_tokens", 0))
    # Fallback: estimate from strings
    for txtk in ("prompt", "completion", "reasoning", "final", "output"):
        if txtk in row and isinstance(row[txtk], str):
            total += max(1, len(row[txtk]) // 4)
    return int(total)

def under_cap(row: Dict[str, Any], cap_tokens: int, include_controller: bool = True) -> bool:
    # If separate fields exist, sum accordingly
    if include_controller:
        child = int(row.get("tokens_child_total", row.get("tokens_child", 0)))
        ctrl  = int(row.get("tokens_controller_total", row.get("controller_tokens", 0)))
        if child or ctrl:
            return (child + ctrl) <= cap_tokens
    # Else use approx_total_tokens
    return approx_total_tokens(row) <= cap_tokens

def pair_rows(a: List[Dict[str, Any]], b: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    # index by id
    idx = {str(r.get("id")): r for r in a if "id" in r}
    pairs = []
    for r in b:
        k = str(r.get("id"))
        if k in idx:
            pairs.append((idx[k], r))
    return pairs

def evaluate_pairs(tot_rows: List[Dict[str, Any]], nlel_rows: List[Dict[str, Any]], cap_tokens: int) -> Dict[str, Any]:
    pairs = pair_rows(tot_rows, nlel_rows)
    # Filter to items under cap for each arm independently
    keep = []
    for a, b in pairs:
        ua = under_cap(a, cap_tokens, include_controller=False)  # ToT has no controllers
        ub = under_cap(b, cap_tokens, include_controller=True)
        if ua and ub:
            keep.append((a, b))
    # metrics
    y_tot = [bool(x[0].get("correct")) for x in keep if "correct" in x[0]]
    y_nle = [bool(x[1].get("correct")) for x in keep if "correct" in x[1]]
    n = len(y_tot)
    if n == 0:
        raise RuntimeError("No paired items under cap with 'correct' field present.")
    # discordant counts
    b01 = sum((not yt) and yn for yt, yn in zip(y_tot, y_nle))
    b10 = sum(yt and (not yn) for yt, yn in zip(y_tot, y_nle))
    chi2, pval = mcnemar(b01, b10)
    acc_tot = sum(y_tot) / n
    acc_nle = sum(y_nle) / n
    # tokens-per-success
    def tps(rows):
        toks = []
        for r in rows:
            if bool(r.get("correct")):
                toks.append(approx_total_tokens(r))
        if not toks: return float("inf")
        return float(np.median(toks))
    tps_tot = tps([x[0] for x in keep])
    tps_nle = tps([x[1] for x in keep])
    # bootstrap CIs for accuracy
    rng = np.random.default_rng(42)
    def boot_ci_binary(vals):
        vals = np.array(vals, dtype=int)
        n = len(vals)
        stats = []
        for _ in range(10000):
            idx = rng.choice(np.arange(n), size=n, replace=True)
            stats.append(np.mean(vals[idx]))
        lo, hi = float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))
        return lo, hi
    ci_tot = boot_ci_binary(y_tot)
    ci_nle = boot_ci_binary(y_nle)
    return {
        "n_paired": n,
        "cap_tokens": cap_tokens,
        "accuracy_tot": acc_tot,
        "accuracy_nlel": acc_nle,
        "accuracy_ci_tot": ci_tot,
        "accuracy_ci_nlel": ci_nle,
        "mcnemar_b01": b01,
        "mcnemar_b10": b10,
        "mcnemar_chi2": chi2,
        "mcnemar_p": pval,
        "tps_tot": tps_tot,
        "tps_nlel": tps_nle,
    }

def main(tot_jsonl: Path, nlel_jsonl: Path, cap_tokens: int = 4000, out_json: Path = Path("pilot_v2_summary.json")):
    tot_rows = read_jsonl(tot_jsonl)
    nle_rows = read_jsonl(nlel_jsonl)
    res = evaluate_pairs(tot_rows, nle_rows, cap_tokens)
    out_json.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tot", type=Path, required=True)
    ap.add_argument("--nlel", type=Path, required=True)
    ap.add_argument("--cap_tokens", type=int, default=4000)
    ap.add_argument("--out", type=Path, default=Path("pilot_v2_summary.json"))
    args = ap.parse_args()
    main(args.tot, args.nlel, args.cap_tokens, args.out)
