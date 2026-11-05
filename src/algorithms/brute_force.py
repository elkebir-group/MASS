"""
Exhaustive MSTP: try every partition of the rows. A reference for testing and
for small inputs; the number of partitions grows as the Bell numbers
(115,975 for 10 rows), so inputs above MAX_ROWS rows are refused.
"""
from functools import lru_cache

import numpy as np

MAX_ROWS = 10


@lru_cache(maxsize=None)
def _set_partitions(n):
    """Every set partition of n items, as restricted growth strings (P, n)."""
    out = []

    def grow(prefix, top):
        if len(prefix) == n:
            out.append(prefix)
            return
        for v in range(top + 2):
            grow(prefix + (v,), max(top, v))

    grow((0,), 0)
    return np.array(out, dtype=np.int64)


def brute_force_partition(matrix, tau):
    """Same contract as MSTPPartition: a (tau x M) 0/1 array whose row t-1
    marks the most columns compatible with one partition of <= t blocks.
    A column is compatible when it is constant on every block."""
    X = np.asarray(matrix)
    n, m = X.shape
    if n > MAX_ROWS:
        raise ValueError(f"brute_force handles at most {MAX_ROWS} rows (got {n})")
    P = _set_partitions(n)
    blocks = P.max(axis=1) + 1
    keys = P[:, None, :] * 2 + (X.T[None, :, :] != 0)            # (P, M, n)
    seen = np.bitwise_or.reduce(np.left_shift(1, keys), axis=2)
    classes = np.vectorize(lambda v: bin(int(v)).count("1"))(seen) if seen.size else seen
    compat = classes == blocks[:, None]                          # (P, M)
    counts = compat.sum(axis=1)
    out = np.zeros((tau, m), dtype=int)
    for t in range(1, tau + 1):
        allowed = np.flatnonzero(blocks <= t)
        best = allowed[np.argmax(counts[allowed])]               # first maximum wins
        out[t - 1] = compat[best]
    return out
