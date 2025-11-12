import os, json, datetime
from typing import Any, Dict, List

def now_ts() -> str:
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def set_seed(seed: int):
    import numpy as np, random
    random.seed(seed); np.random.seed(seed)

def safe_jsonl_write(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
