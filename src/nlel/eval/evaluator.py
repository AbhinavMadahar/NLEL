from typing import Tuple, Dict, Any, Optional
import json, re
from ..models.base import TextModel

def normalize_answer(s: str) -> str:
    s = re.sub(r"\s+"," ", s.strip()).strip(" .\n\t")
    return s.lower()

def parse_final_answer(text: str) -> Optional[str]:
    m = re.search(r"Final Answer\s*:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
    return normalize_answer(m.group(1)) if m else None

class ValueEstimator:
    def __init__(self, model: Optional[TextModel] = None): self.model = model
    def score(self, task: str, candidate: str):
        if self.model is None:
            mu = 0.35; sigma = 0.5; return mu, sigma, {"usage":{"prompt_tokens":0,"completion_tokens":0}}
        from ..prompts import load_prompt
        prompt = load_prompt("evaluator.txt").format(task=task, candidate=candidate)
        resp, meta = self.model.generate(prompt, temperature=0.0, top_p=1.0, max_tokens=64)
        try:
            obj = json.loads(resp); mu = float(obj.get("mu",0.5)); sigma = float(obj.get("sigma",0.5))
        except Exception:
            mu, sigma = 0.5, 0.5
        return mu, sigma, meta

class ExactMatchChecker:
    def __init__(self, gold: str): self.gold = normalize_answer(gold)
    def check(self, pred: str) -> bool:
        ans = parse_final_answer(pred) or normalize_answer(pred)
        return ans == self.gold
