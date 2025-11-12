from typing import Dict, Any, Tuple
from ..models.base import TextModel
from ..prompts import load_prompt
class Verifier:
    def __init__(self, model: TextModel):
        self.model = model
    def verify(self, task: str, candidate: str, strictness: float = 0.5, passes: int = 1) -> Tuple[bool, Dict[str, Any]]:
        accept_votes = 0; usage_total = {"prompt_tokens":0, "completion_tokens":0}
        for _ in range(max(1, passes)):
            resp, meta = self.model.generate(load_prompt("verifier.txt").format(task=task, candidate=candidate, strictness=str(strictness)), temperature=0.0, top_p=1.0, max_tokens=4)
            txt = (resp or "").strip().upper()
            accept = "ACCEPT" in txt and "REJECT" not in txt
            if accept: accept_votes += 1
            u = meta.get("usage", {})
            usage_total["prompt_tokens"] += int(u.get("prompt_tokens",0)); usage_total["completion_tokens"] += int(u.get("completion_tokens",0))
        return accept_votes >= (passes // 2 + 1), {"usage": usage_total}
