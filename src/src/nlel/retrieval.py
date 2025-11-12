def retrieval_context(weights, novelty: float) -> str:
    wgen = float(weights.get("general", 0.0))
    wmath = float(weights.get("math-lemmas", 0.0))
    parts = []
    if wgen > 0.1: parts.append("General background: check arithmetic; keep steps concise.")
    if wmath > 0.1: parts.append("Math lemmas: parity, factoring identities, simple inequalities.")
    if not parts and novelty > 0.7: parts.append("Heuristic: consider a simpler sub-goal or alternative representation.")
    return "\n".join(parts)
