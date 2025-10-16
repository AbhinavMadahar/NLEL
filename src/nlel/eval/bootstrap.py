
from typing import Callable, Sequence, Tuple
import numpy as np

def bootstrap_ci(values: Sequence[float], stat: Callable[[np.ndarray], float], alpha: float = 0.05, n_resamples: int = 10000, random_state: int = 42) -> Tuple[float, float]:
    """
    Basic percentile bootstrap CI for a statistic over 'values'.
    """
    rng = np.random.default_rng(random_state)
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return float('nan'), float('nan')
    stats = []
    n = len(values)
    for _ in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        stats.append(stat(sample))
    lo = float(np.percentile(stats, 100 * alpha / 2))
    hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return lo, hi
