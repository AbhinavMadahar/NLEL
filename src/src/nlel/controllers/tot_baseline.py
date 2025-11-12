from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from ..models.base import TextModel
from ..eval.evaluator import ValueEstimator, ExactMatchChecker
from ..tokens import TokenBank
from ..config import DEFAULT_P_TOT, MAX_DEPTH, MAX_TOTAL_EXPANSIONS, beta_at_depth
from .tot import Candidate, tot_select
@dataclass
class ToTParams:
    temperature: float = float(DEFAULT_P_TOT["temperature"])
    top_p: float = float(DEFAULT_P_TOT["top_p"])
    max_tokens: int = int(DEFAULT_P_TOT["max_tokens"])
    repetition_penalty: float = float(DEFAULT_P_TOT["repetition_penalty"])
    gen_count: int = int(DEFAULT_P_TOT["gen_count"])
    branch_quota: int = int(DEFAULT_P_TOT["branch_quota"])
    beta: float = float(DEFAULT_P_TOT["beta"])
def run_tot(task: str, model: TextModel, gold_answer: Optional[str] = None, params: ToTParams = ToTParams(), with_verifier=False, verifier=None, verifier_passes=1, verifier_strictness=0.5, budget_tokens: int = 8000) -> Dict[str, Any]:
    tb = TokenBank(); ve = ValueEstimator(model=model); depth = 0; parent = ""; expansions = 0; best_leaf = None
    while expansions < MAX_TOTAL_EXPANSIONS and depth < MAX_DEPTH and tb.total < budget_tokens:
        prompts = [f"Task:\n{task}\n\nParent step:\n{parent}\n\nDirective: default\n\nContinue reasoning. End with 'Final Answer: <answer>' if possible." for _ in range(params.gen_count)]
        gens = model.batch_generate(prompts, temperature=params.temperature, top_p=params.top_p, max_tokens=params.max_tokens, repetition_penalty=params.repetition_penalty)
        cands: List[Candidate] = []; usage = {"prompt_tokens":0,"completion_tokens":0}
        for text, meta in gens:
            mu, sigma, meta_val = ve.score(task, text); cands.append(Candidate(text=text, mu=mu, sigma=sigma, score=mu + beta_at_depth(depth, base_beta=params.beta) * sigma, usage=meta, label="default", pi=dict(DEFAULT_P_TOT)))
            for m in (meta, meta_val):
                u = m.get("usage", {}); usage["prompt_tokens"] += int(u.get("prompt_tokens",0)); usage["completion_tokens"] += int(u.get("completion_tokens",0))
        tb.add(**usage); expansions += len(cands)
        survivors = tot_select(cands, k=params.branch_quota)
        for cand in survivors:
            if "Final Answer:" in cand.text:
                best_leaf = cand; break
        if best_leaf is not None or tb.total >= budget_tokens: break
        parent = survivors[0].text; depth += 1
    verified = None
    if best_leaf and with_verifier and verifier is not None:
        ok, meta_v = verifier.verify(task, best_leaf.text, passes=verifier_passes, strictness=verifier_strictness); tb.add(**meta_v.get("usage", {})); verified = ok
    correct = None
    if gold_answer is not None and best_leaf is not None:
        checker = ExactMatchChecker(gold_answer); correct = checker.check(best_leaf.text)
    return {"final": getattr(best_leaf, "text", None), "tokens_total": tb.total, "expansions": expansions, "correct": bool(correct) if correct is not None else None, "verified": verified}
