from typing import List, Dict, Any, Tuple
import pandas as pd, numpy as np
BUDGETS = [0.5, 1.0, 2.0]
def summarize(rows: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not rows: return pd.DataFrame(), pd.DataFrame()
    df = pd.DataFrame(rows)
    if "correct" in df.columns: df["correct"] = df["correct"].astype("boolean")
    if "budget_multiplier" not in df.columns: df["budget_multiplier"] = np.nan
    def tps(g):
        toks = g.loc[g["correct"]==True, "tokens_total"]
        return float(np.median(toks)) if len(toks) else float("inf")
    agg = []
    overall_acc = df.loc[df["correct"].notna(), "correct"].mean()
    agg.append({"metric":"accuracy_overall", "value": float(overall_acc)})
    
    # Verification accept rate (overall)
    if "verified" in df.columns:
        mask_v = df["verified"].notna()
        if mask_v.any():
            var = df.loc[mask_v, "verified"].astype("boolean").mean()
            agg.append({"metric":"verification_accept_rate_overall", "value": float(var)})
for b in BUDGETS:
        sub = df[df["budget_multiplier"]==b]
        # accuracy@compute slice
        agg.append({"metric":f"accuracy_at_{b}x", "value": float(sub["correct"].mean()) if len(sub) else float("nan")})
        # verification accept rate@compute slice
        if "verified" in sub.columns and len(sub):
            mask_sv = sub["verified"].notna()
            if mask_sv.any():
                agg.append({"metric":f"verification_accept_rate_at_{b}x", "value": float(sub.loc[mask_sv, "verified"].astype("boolean").mean())})
    agg.append({"metric":"tokens_per_success", "value": tps(df)})
    return df, pd.DataFrame(agg)
