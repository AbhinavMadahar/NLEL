from typing import List, Dict, Any, Tuple, Optional
import json, re
from dataclasses import dataclass, field
from ..models.base import TextModel
from ..prompts import load_prompt
from ..schema import ControlVector, schema_validate_or_default, trust_region_project, quantize_controls
from ..config import DEFAULT_P0, LEDGER_MAX_ROWS, MAX_DEPTH, MAX_TOTAL_EXPANSIONS, beta_at_depth
from ..ledger.ledger import Ledger
from .tot import tot_select, Candidate
from ..tokens import TokenBank
from ..retrieval import retrieval_context
from ..eval.evaluator import ValueEstimator

@dataclass
class Context:
    depth: int = 0
    frontier_sigma_median: float = 0.5
    novelty_median: float = 0.5
    siblings_best_mu: float = 0.0
    siblings_best_sigma: float = 0.5
    label_history: List[str] = field(default_factory=list)
    tokens_used: int = 0
    tokens_budget: int = 10000
    def to_json(self) -> str:
        obj = dict(depth=self.depth, frontier_sigma_median=self.frontier_sigma_median, novelty_median=self.novelty_median,
                   siblings_best_mu=self.siblings_best_mu, siblings_best_sigma=self.siblings_best_sigma,
                   label_history=self.label_history[-8:], tokens_used=self.tokens_used, tokens_budget=self.tokens_budget)
        import json; return json.dumps(obj, ensure_ascii=False)

class Labeller:
    def __init__(self, model: TextModel, max_labels: int = 3, random_labels: bool = False, frozen: bool = False):
        self.model = model; self.max_labels = max_labels; self.random_labels = random_labels; self.frozen = frozen
        self._pool = ["work backward","seek a counterexample","sketch plan","call retrieval; summarize first","prove contrapositive"]
    def emit_labels(self, parent: str, ctx: Context):
        if self.frozen: return (["default"], {"usage":{"prompt_tokens":0,"completion_tokens":0}})
        if self.random_labels:
            import random; labs = random.sample(self._pool, k=min(self.max_labels, len(self._pool)))
            return (labs, {"usage":{"prompt_tokens":0,"completion_tokens":0}})
        resp, meta = self.model.generate(load_prompt("labeller.txt").format(parent=parent[:1000], context_json=ctx.to_json(), max_labels=self.max_labels), temperature=0.3, top_p=0.9, max_tokens=64)
        labels = [s.strip() for s in re.split(r"[;\n]", resp) if s.strip()]
        labels = list(dict.fromkeys(labels))[: self.max_labels] or ["default"]
        return labels, meta

class TunerJPE:
    def __init__(self, model: TextModel, trust_region_r: float = 0.15, no_trust_region: bool = False, quantize_bits: int = 0, frozen: bool = False):
        self.model = model; self.r = trust_region_r; self.no_trust_region = no_trust_region; self.quantize_bits = quantize_bits; self.frozen = frozen
        self.ledger = Ledger(max_rows=LEDGER_MAX_ROWS)
    def emit_controls(self, parent: str, label: str, ctx: Context):
        if self.frozen: return (ControlVector(**DEFAULT_P0), {"usage":{"prompt_tokens":0,"completion_tokens":0}})
        p0 = json.dumps(DEFAULT_P0, ensure_ascii=False); ledger_block = self.ledger.render_block()
        prompt = load_prompt("tuner_jpe.txt").format(p0_json=p0, ledger_block=ledger_block, parent=parent[:1000], label=label, context_json=ctx.to_json())
        resp, meta = self.model.generate(prompt, temperature=0.0, top_p=1.0, max_tokens=256)
        try:
            start = resp.find('{'); end = resp.rfind('}'); obj = json.loads(resp[start:end+1])
        except Exception:
            obj = DEFAULT_P0
        cv = schema_validate_or_default(obj, DEFAULT_P0)
        if not self.no_trust_region: cv = trust_region_project(cv, r=self.r, p0=DEFAULT_P0)
        if self.quantize_bits and self.quantize_bits>0: cv = quantize_controls(cv, bits=self.quantize_bits)
        return cv, meta

def _expand_under_label(task: str, parent: str, label: str, ctx: Context, tuner: TunerJPE, reasoner: TextModel, val_est: ValueEstimator):
    cv, meta_tuner = tuner.emit_controls(parent, label, ctx)
    rctx = retrieval_context(cv.retrieval_weights or {}, novelty=float(ctx.novelty_median))
    prompts = []
    for _ in range(int(cv.gen_count)):
        chunk = f"\n\nRetrieved context:\n{rctx}" if rctx else ""
        prompts.append(f"Task:\n{task}\n\nParent step:\n{parent}{chunk}\n\nDirective: {label}\n\nContinue the reasoning. If you can conclude, write 'Final Answer: <answer>'.")
    gens = reasoner.batch_generate(prompts, temperature=cv.temperature, top_p=cv.top_p, max_tokens=cv.max_tokens, repetition_penalty=cv.repetition_penalty)
    children = []; usage_total = {"prompt_tokens":0,"completion_tokens":0}
    for text, meta in gens:
        mu, sigma, meta_val = val_est.score(task, text); beta_eff = beta_at_depth(ctx.depth, base_beta=float(cv.beta))
        score = mu + beta_eff * sigma
        cand = Candidate(text=text, mu=mu, sigma=sigma, score=score, usage=meta, label=label,
                 pi={**cv.model_dump(), 'beta': float(beta_eff)})
        children.append(cand)
        for m in (meta, meta_val):
            u = m.get("usage", {}); usage_total["prompt_tokens"] += int(u.get("prompt_tokens",0)); usage_total["completion_tokens"] += int(u.get("completion_tokens",0))
    if children:
        tuner.ledger.add({"L": label, "Pi": cv.model_dump(), "mu": float(sum(c.mu for c in children)/len(children)), "sigma": float(sum(c.sigma for c in children)/len(children)), "accept": None, "cost": usage_total})
    return children, usage_total, cv

def run_instance(task: str, gold_answer: Optional[str], model: TextModel, budget_tokens: int = 8000, labeller: Labeller=None, tuner: TunerJPE=None, verifier=None, ignore_verifier_control: bool=False):
    ctx = Context(depth=0, tokens_budget=budget_tokens); parent_text = ""; from ..tokens import TokenBank; tb = TokenBank()
    from ..eval.evaluator import ExactMatchChecker
    val_est = ValueEstimator(model=model)
    total_exp = 0; best_leaf = None
    while total_exp < MAX_TOTAL_EXPANSIONS and ctx.depth < MAX_DEPTH and tb.total < budget_tokens:
        labels, meta_lab = labeller.emit_labels(parent_text, ctx) if labeller else (["default"], {"usage":{"prompt_tokens":0,"completion_tokens":0}})
        all_cands = []; usage_acc = {"prompt_tokens":0,"completion_tokens":0}; branch_quotas = []
        for L in labels:
            kids, usage, cv = _expand_under_label(task, parent_text, L, ctx, tuner, model, val_est)
            all_cands.extend(kids); branch_quotas.append(int(cv.branch_quota))
            usage_acc["prompt_tokens"] += usage["prompt_tokens"]; usage_acc["completion_tokens"] += usage["completion_tokens"]
            total_exp += len(kids)
            if tb.total + usage_acc["prompt_tokens"] + usage_acc["completion_tokens"] >= budget_tokens: break
        tb.add(**usage_acc)
        if not all_cands: break
        k_eff = max(branch_quotas) if branch_quotas else 1
        survivors = tot_select(all_cands, k=k_eff)
        for cand in survivors:
            if "Final Answer:" in cand.text:
                best_leaf = cand; break
        if best_leaf is not None or tb.total >= budget_tokens: break
        parent_text = survivors[0].text; ctx.depth += 1; ctx.label_history.extend([c.label for c in survivors])
    verified = None
    if best_leaf and verifier:
        if ignore_verifier_control:
            passes, strict = 1, 0.5
        else:
            pi = best_leaf.pi or {}; passes = int(pi.get("verify_passes", 1)); strict = float(pi.get("verify_strictness", 0.5))
        ok, meta_v = verifier.verify(task, best_leaf.text, passes=passes, strictness=strict); tb.add(**meta_v.get("usage", {})); verified = ok
    correct = None
    if gold_answer is not None and best_leaf is not None:
        checker = ExactMatchChecker(gold_answer); correct = checker.check(best_leaf.text)
    return {"final": getattr(best_leaf, "text", None), "tokens_total": tb.total, "expansions": total_exp, "correct": bool(correct) if correct is not None else None, "verified": verified}
