
from typing import Iterable, Tuple
import math

def mcnemar(b01: int, b10: int, exact: bool = False) -> Tuple[float, float]:
    """
    McNemar's test on paired binary outcomes.
    Inputs:
      b01: count where A incorrect, B correct
      b10: count where A correct,   B incorrect
    Returns:
      (statistic, p_value) for two-sided test (chi-square approx unless exact=True).
    """
    if exact:
        from math import comb
        n = b01 + b10
        if n == 0:
            return 0.0, 1.0
        # Two-sided exact binomial p-value
        k = min(b01, b10)
        p = sum(comb(n, i) for i in range(0, k+1)) * (0.5 ** n) * 2.0
        return 0.0, min(1.0, p)
    # Continuity-corrected chi-square (Edwards 1948)
    num = abs(b01 - b10) - 1
    if num < 0: num = 0.0
    den = b01 + b10 if (b01 + b10) else 1.0
    chi2 = (num * num) / den
    # Two-sided p-value
    try:
        import mpmath as mp
        p = 1 - mp.gammainc(1/2, 0, chi2/2) / mp.gamma(1/2)
    except Exception:
        # Approximate using math.erfc for df=1
        p = math.erfc(math.sqrt(chi2)/math.sqrt(2))
    return float(chi2), float(p)
