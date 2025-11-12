
import os
try:
    from datasets import DownloadConfig
    _NLEL_LOCAL_ONLY = os.getenv("HF_DATASETS_OFFLINE", "0") == "1"
    def _dlcfg(): return DownloadConfig(local_files_only=_NLEL_LOCAL_ONLY)
except Exception:
    _NLEL_LOCAL_ONLY = False
    def _dlcfg(): return None
from typing import Iterator, Dict, Any, Optional
import datasets
import re
def load_gsm8k(split: str = "test", subset: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    ds = datasets.load_dataset("openai/gsm8k", "main", download_config=_dlcfg())[split]
    for i, row in enumerate(ds):
        if subset is not None and i >= subset: break
        yield {"id": f"gsm8k-{i}", "question": row["question"], "answer": row["answer"]}
def load_strategyqa(split: str = "test", subset: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    ds = datasets.load_dataset("strategyqa", "default", download_config=_dlcfg())[split]
    for i, row in enumerate(ds):
        if subset is not None and i >= subset: break
        ans = row.get("answer", None)
        gold = None if ans is None else ("yes" if ans else "no")
        yield {"id": f"strategyqa-{i}", "question": row["question"], "answer": gold}
def load_arc_challenge(split: str = "test", subset: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    ds = datasets.load_dataset("ai2_arc", "ARC-Challenge", download_config=_dlcfg())[split]
    for i, row in enumerate(ds):
        if subset is not None and i >= subset: break
        stem = row["question"]; choices = row["choices"]["text"]; label = row["choices"]["label"]; gold_label = row["answerKey"]
        options = "\n".join([f"({lab}) {txt}" for lab, txt in zip(label, choices)])
        q = f"{stem}\n{options}\nAnswer with the option letter."
        yield {"id": f"arc-chal-{i}", "question": q, "answer": gold_label.lower()}

def load_math_subset(split: str = "test", subset: Optional[int] = 500) -> Iterator[Dict[str, Any]]:
    """
    Loads the MATH (subset) benchmark consistent with the paper's wording (§6.1),
    using the Hendrycks Competition Math distribution on Hugging Face:
        dataset = "hendrycks/competition_math"
    We parse the gold answer as the final LaTeX \boxed{...} token (if present),
    falling back to any 'answer' field, and skip items we cannot parse.
    Returns dicts with keys: {"id", "question", "answer"}.
    """
    ds = datasets.load_dataset("hendrycks/competition_math", "all", download_config=_dlcfg())[split]
    count = 0
    for i, row in enumerate(ds):
        if subset is not None and count >= subset:
            break
        # Fields vary by config; typical keys: problem, solution, level, type
        q = row.get("problem") or row.get("question") or ""
        sol = row.get("solution") or row.get("answer") or ""
        # Extract last \boxed{...}
        gold = None
        if isinstance(sol, str):
            boxes = re.findall(r"\\boxed\{([^}]*)\}", sol)
            if boxes:
                gold = boxes[-1]
        if gold is None and isinstance(sol, str) and sol.strip():
            # Fallback: Use raw solution if simple token (no whitespace/newlines)
            s = sol.strip()
            if len(s.split()) == 1:
                gold = s
        if not q or gold is None:
            continue
        yield {"id": f"math-compet-{i}", "question": q, "answer": gold}
        count += 1
def get_loader(name: str):
    if name == "gsm8k": return load_gsm8k
    if name == "strategyqa": return load_strategyqa
    if name == "arc_challenge": return load_arc_challenge
    if name == "math_subset": return load_math_subset
    raise ValueError(f"Unknown benchmark: {name}")
