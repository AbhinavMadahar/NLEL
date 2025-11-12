from typing import List, Dict, Any
from dataclasses import dataclass, field
@dataclass
class Candidate:
    text: str
    mu: float
    sigma: float
    score: float
    usage: Dict[str, Any] = field(default_factory=dict)
    label: str = ""
    pi: Dict[str, Any] = field(default_factory=dict)
def tot_select(cands: List[Candidate], k: int) -> List[Candidate]:
    return sorted(cands, key=lambda c: c.score, reverse=True)[:k]
