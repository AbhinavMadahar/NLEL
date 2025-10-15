from typing import Dict, Any, Optional
from ..models.base import TextModel
from ..tokens import TokenBank
from ..eval.evaluator import ExactMatchChecker
def run_cot(task: str, model: TextModel, max_tokens: int = 256, gold_answer: Optional[str] = None) -> Dict[str, Any]:
    text, meta = model.generate(f"Solve step by step. End with 'Final Answer: <answer>'.\n\nProblem:\n{task}\n", temperature=0.2, top_p=0.9, max_tokens=max_tokens)
    tb = TokenBank(); tb.add(**meta.get("usage", {}))
    correct = None
    if gold_answer is not None:
        checker = ExactMatchChecker(gold_answer); correct = checker.check(text)
    return {"final": text, "tokens_total": tb.total, "correct": correct}
def run_sc_cot(task: str, model: TextModel, samples: int = 5, max_tokens: int = 256, gold_answer: Optional[str] = None) -> Dict[str, Any]:
    from collections import Counter; import re
    tb = TokenBank(); texts = []
    for _ in range(samples):
        t, meta = model.generate(f"Solve step by step and end with 'Final Answer: <answer>'.\n\nProblem:\n{task}\n", temperature=0.4, top_p=0.95, max_tokens=max_tokens)
        texts.append(t); tb.add(**meta.get("usage", {}))
    def extract(x):
        m = re.search(r"Final Answer\s*:\s*(.+)$", x, re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip().lower() if m else x.strip().lower()
    best = Counter(extract(x) for x in texts).most_common(1)[0][0]
    final = f"Self-consistency vote selected: {best}\nFinal Answer: {best}"
    correct = None
    if gold_answer is not None:
        checker = ExactMatchChecker(gold_answer); correct = checker.check(final)
    return {"final": final, "tokens_total": tb.total, "correct": correct}
