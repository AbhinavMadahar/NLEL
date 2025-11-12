from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class ControlSchemaBounds:
    temperature: Tuple[float, float] = (0.0, 1.0)
    top_p: Tuple[float, float] = (0.0, 1.0)
    max_tokens: Tuple[int, int] = (32, 512)
    repetition_penalty: Tuple[float, float] = (0.0, 2.0)
    gen_count: Tuple[int, int] = (1, 8)
    branch_quota: Tuple[int, int] = (1, 8)
    beta: Tuple[float, float] = (0.0, 1.0)
    verify_passes: Tuple[int, int] = (0, 5)
    verify_strictness: Tuple[float, float] = (0.0, 1.0)

DEFAULT_SCHEMA_BOUNDS = ControlSchemaBounds()

DEFAULT_P0: Dict[str, object] = {
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 128,
    "repetition_penalty": 1.0,
    "gen_count": 1,
    "branch_quota": 2,
    "beta": 0.15,
    "verify_passes": 1,
    "verify_strictness": 0.5,
    "retrieval_weights": {"general": 0.0, "math-lemmas": 0.0},
}

DEFAULT_P_TOT: Dict[str, object] = {
    "temperature": 0.3,
    "top_p": 0.9,
    "max_tokens": 128,
    "repetition_penalty": 1.0,
    "gen_count": 3,
    "branch_quota": 2,
    "beta": 0.15,
    "verify_passes": 0,
    "verify_strictness": 0.5,
    "retrieval_weights": {"general": 0.0, "math-lemmas": 0.0},
}

def beta_at_depth(depth: int, base_beta: float = 0.15, tau: float = 3.0, min_beta: float = 0.02) -> float:
    import math
    val = base_beta * math.exp(-depth / max(tau, 1e-6))
    return max(min_beta, min(val, 1.0))

LEDGER_MAX_ROWS = 32
MAX_DEPTH = 6
MAX_TOTAL_EXPANSIONS = 128

DEFAULT_SEEDS = [13, 17, 23, 29, 31]
