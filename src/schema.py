from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Any, Tuple
from .config import DEFAULT_SCHEMA_BOUNDS, DEFAULT_P0

class ControlVector(BaseModel):
    temperature: float = Field(0.2, ge=0.0, le=1.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(128, ge=1, le=4096)
    repetition_penalty: float = Field(1.0, ge=0.0, le=4.0)
    gen_count: int = Field(1, ge=1, le=64)
    branch_quota: int = Field(2, ge=1, le=64)
    beta: float = Field(0.15, ge=0.0, le=1.0)
    verify_passes: int = Field(1, ge=0, le=10)
    verify_strictness: float = Field(0.5, ge=0.0, le=1.0)
    retrieval_weights: Dict[str, float] = Field(default_factory=dict)

def _norm_range(val, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    if hi == lo: return 0.0
    return (float(val) - lo) / (hi - lo)

def _denorm_range(t, bounds: Tuple[float, float]) -> float:
    lo, hi = bounds
    return lo + t * (hi - lo)

def trust_region_project(p: ControlVector, r: float, p0: Dict[str, Any] = DEFAULT_P0) -> ControlVector:
    b = DEFAULT_SCHEMA_BOUNDS
    ranges = {
        "temperature": b.temperature,
        "top_p": b.top_p,
        "max_tokens": b.max_tokens,
        "repetition_penalty": b.repetition_penalty,
        "gen_count": b.gen_count,
        "branch_quota": b.branch_quota,
        "beta": b.beta,
        "verify_passes": b.verify_passes,
        "verify_strictness": b.verify_strictness,
    }
    p_dict = p.model_dump()
    out = {}
    for k, v in p_dict.items():
        if k == "retrieval_weights":
            out[k] = dict(v); continue
        base = p0[k]
        nv = _norm_range(v, ranges[k])
        nb = _norm_range(base, ranges[k])
        nv = max(nb - r, min(nb + r, nv))
        out[k] = type(v)(_denorm_range(nv, ranges[k]))
    # normalize retrieval weights to sum<=1
    rw = out.get("retrieval_weights", {})
    total = sum(max(0.0, min(1.0, float(w))) for w in rw.values())
    if total > 1.0 and total > 0.0:
        out["retrieval_weights"] = {k: float(max(0.0, min(1.0, float(w))) / total) for k, w in rw.items()}
    else:
        out["retrieval_weights"] = {k: float(max(0.0, min(1.0, float(w)))) for k, w in rw.items()}
    try:
        return ControlVector(**out)
    except ValidationError:
        return ControlVector(**p0)

def schema_validate_or_default(obj: dict, p0: dict = DEFAULT_P0) -> ControlVector:
    try:
        return ControlVector(**obj)
    except ValidationError:
        return ControlVector(**p0)

def quantize_controls(cv: ControlVector, bits: int) -> ControlVector:
    if bits is None or bits <= 0: return cv
    import math
    levels = max(2, 2 ** int(bits))
    b = DEFAULT_SCHEMA_BOUNDS
    def q(x, lo, hi):
        if hi <= lo: return float(x)
        t = (float(x) - lo) / (hi - lo)
        t = min(1.0, max(0.0, t))
        idx = round(t * (levels - 1))
        tq = idx / (levels - 1)
        return lo + tq * (hi - lo)
    data = cv.model_dump()
    data["temperature"] = q(cv.temperature, *b.temperature)
    data["top_p"] = q(cv.top_p, *b.top_p)
    data["max_tokens"] = int(q(cv.max_tokens, *b.max_tokens))
    data["repetition_penalty"] = q(cv.repetition_penalty, 0.0, 4.0)
    data["beta"] = q(cv.beta, *b.beta)
    data["verify_strictness"] = q(cv.verify_strictness, *b.verify_strictness)
    from pydantic import ValidationError
    try:
        return ControlVector(**data)
    except ValidationError:
        return cv
