from typing import Dict, Any, Optional, Tuple
import re
from ..models.base import TextModel
from ..tokens import TokenBank
from ..eval.evaluator import ExactMatchChecker, ValueEstimator

RE_ACT_HEADER = """You are a helpful reasoner that interleaves Thoughts and Actions.
At each step, you may output one of:
- Thought: <your reasoning>
- Action: Search[<query>]  # ask a retrieval tool
- Action: Lookup[<query>]  # quick fact lookup
- Action: Finish[<final answer>]  # stop and return the final answer

Format STRICTLY as lines starting with 'Thought:' or 'Action:'.
When you decide the final answer, use: Action: Finish[...].
Task:
{task}
Begin.
"""

def _react_step_prompt(history: str, task: str) -> str:
    return RE_ACT_HEADER.format(task=task) + "\n" + history

def _parse_action(s: str) -> Optional[Tuple[str, str]]:
    m = re.search(r"Action\s*:\s*(Search|Lookup|Finish)\[(.*)\]\s*$", s.strip(), re.IGNORECASE)
    if not m: return None
    return m.group(1).capitalize(), m.group(2).strip()

def _tool_exec(kind: str, arg: str) -> str:
    # Stubbed retrieval: no external KB in this repo.
    # Keep deterministic and cheap; mirror paper's "where allowed" note.
    if kind == "Search":
        return f"[stub] No external knowledge base available for query: {arg}."
    if kind == "Lookup":
        return f"[stub] Lookup not available offline; echoing query: {arg}."
    return ""

def run_react(task: str, model: TextModel, max_steps: int = 6, max_tokens: int = 128,
              temperature: float = 0.2, top_p: float = 0.9, gold_answer: Optional[str] = None) -> Dict[str, Any]:
    history = ""
    tb = TokenBank()
    best_final = None
    value = ValueEstimator(model=None)  # heurstic scoring if needed
    for step in range(max_steps):
        prompt = _react_step_prompt(history, task)
        text, meta = model.generate(prompt, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        tb.add(**meta.get("usage", {}))
        # Keep only trailing lines added this step
        added = text.splitlines()[-1] if text.strip() else ""
        if added.lower().startswith("thought:"):
            history += added + "\n"
            continue
        act = _parse_action(added)
        if act is None:
            # If the model didn't emit a recognized action, try to force a Finish at last step
            if step == max_steps - 1:
                history += f"Action: Finish[{text.strip()}]\n"
                best_final = f"{history}\n"
                break
            else:
                history += "Thought: continue reasoning.\n"
                continue
        kind, arg = act
        if kind == "Finish":
            history += added + "\n"
            best_final = f"{history}\n"
            break
        obs = _tool_exec(kind, arg)
        history += added + "\n" + f"Observation: {obs}\n"
    # Ensure a Final Answer format for the evaluator
    final_text = best_final or history
    if "Final Answer:" not in final_text:
        final_text += f"\nFinal Answer: {arg if act else ''}"
    correct = None
    if gold_answer is not None:
        checker = ExactMatchChecker(gold_answer); correct = checker.check(final_text)
    return {"final": final_text, "tokens_total": tb.total, "correct": bool(correct) if correct is not None else None}