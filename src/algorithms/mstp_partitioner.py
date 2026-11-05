#!/usr/bin/env python3
"""
MSTP (Max-Subset τ-Partitioning) Algorithm

This module implements the MSTP algorithm for binary matrix partitioning.
The algorithm finds the maximum subset of columns that can be clustered
into at most tau clusters for a given binary matrix.

Memory notes
------------
The search enumerates partitions of the rows (structures) reachable by
successively refining with features (columns). The number of distinct
partitions is what the problem intrinsically requires; everything else is
kept bounded:

* parents are refined in chunks, so per-chunk temporaries never exceed
  ``_CHUNK_CELLS`` cells regardless of how many partitions exist;
* partition labels are stored as int8 (int16 / int32 only for huge tau),
  and dedup keys as uint16 (uint32 only for > 65535 rows);
* partitions are deduplicated against every previously seen partition
  (exact mode), so no partition is expanded twice;
* the best partition per tau is tracked on the fly instead of building a
  compatibility matrix over all partitions at the end.

Results (including tie-breaking between equally good solutions, and beam
behaviour when ``topN`` is finite) are identical to the original dense
implementation.
"""

import numpy as np

# Max number of (child-partition x structure) cells materialised at once.
# Peak temporaries are a small multiple of this in bytes (~10x).
_CHUNK_CELLS = 1 << 22


def _unique_val_maker(X):
    """Map each row of X to an integer id such that equal rows share an id."""
    _, vals1 = np.unique(X[:, 0], return_inverse=True)
    for a in range(1, X.shape[1]):
        vals2_unique, vals2 = np.unique(X[:, a], return_inverse=True)
        vals1 = (vals1 * vals2_unique.shape[0]) + vals2
        _, vals1 = np.unique(vals1, return_inverse=True)
    return vals1


def _preprocess(X):
    """
    Canonicalise features (rows of X, which is features x structures):
    a feature and its complement induce the same split, so flip to the
    minority-ones form, then collapse duplicate features.

    Returns (unique_features, inverse) where inverse maps every original
    feature to its unique feature index.
    """
    mean1 = np.mean(X, axis=1)
    argEven = np.argwhere(mean1 == 0.5)[:, 0]
    if argEven.shape[0] >= 1:
        mean1[argEven] = mean1[argEven] - 0.5 + X[argEven, 0]

    X[mean1 > 0.5] = 1 - X[mean1 > 0.5]

    inverse1 = _unique_val_maker(X)
    _, index1 = np.unique(inverse1, return_index=True)
    return X[index1], inverse1


def _smallest_int(max_value, candidates):
    for dt in candidates:
        if max_value <= np.iinfo(dt).max:
            return dt
    raise ValueError(f"value {max_value} too large")


def _as_void(keys):
    """View each row of a 2-D array as one opaque bytes value (for sort/unique)."""
    keys = np.ascontiguousarray(keys)
    return keys.view(np.dtype((np.void, keys.dtype.itemsize * keys.shape[1]))).ravel()


class _Refiner:
    """Chunked refinement of partitions by every (unique) feature."""

    def __init__(self, X, tau):
        self.F, self.S = X.shape
        self.tau = tau
        self.label_dtype = _smallest_int(2 * tau, (np.int8, np.int16, np.int32))
        # Dedup keys store, for each structure, the last index of its block.
        # Byte-wise ordering of little-endian uint16/uint32 matches that of the
        # int64 keys used by the original implementation (high bytes are 0),
        # so the order of partitions -- and hence tie-breaking -- is preserved.
        self.key_dtype = np.dtype(_smallest_int(max(self.S - 1, 0), (np.uint16, np.uint32))).newbyteorder('<')
        self.X = X.astype(self.label_dtype)
        self.chunk = max(1, _CHUNK_CELLS // max(1, self.F * self.S))

    def _children(self, labels):
        """All refinements of each parent by each feature, plus block counts."""
        c = labels.shape[0]
        child = (labels[:, None, :] * 2 + self.X[None, :, :]).reshape(c * self.F, self.S)
        nvals = 2 * (int(labels.max()) + 1) if labels.size else 2
        present = np.empty((child.shape[0], nvals), dtype=bool)
        for v in range(nvals):
            np.any(child == v, axis=1, out=present[:, v])
        blocks = present.sum(axis=1)
        return child, present, blocks

    def _canonicalise(self, child, present):
        """Compact 0..b-1 labels and last-occurrence dedup keys for children."""
        n, nvals = present.shape
        rank = (np.cumsum(present, axis=1) - 1).astype(self.label_dtype)
        flat_idx = child.astype(np.intp) + (np.arange(n, dtype=np.intp) * nvals)[:, None]
        labels = rank.ravel()[flat_idx]

        last = np.zeros((n, nvals), dtype=self.key_dtype)
        rev = child[:, ::-1]
        for v in range(nvals):
            if present[:, v].any():
                last[:, v] = self.S - 1 - np.argmax(rev == v, axis=1)
        keys = last.ravel()[flat_idx]
        return labels, keys

    def compat(self, labels, blocks):
        """Boolean (n, F): feature f is constant within every block of partition p."""
        out = np.empty((labels.shape[0], self.F), dtype=bool)
        for s in range(0, labels.shape[0], self.chunk):
            e = s + self.chunk
            _, _, cb = self._children(labels[s:e])
            out[s:e] = cb.reshape(-1, self.F) == blocks[s:e, None]
        return out

    def expand(self, labels, blocks, visit, pruner=None, bounds=None):
        """
        Return (cand_labels, cand_keys, cand_bounds, pruned_keys): the
        distinct proper refinements of the parents with <= tau blocks (sorted
        by key), and -- with a pruner -- an upper bound per candidate and
        block count, plus the keys of candidates the pruner rejected.

        For each chunk of parents, in order:
        * ``pruner.alive(bounds, blocks)`` (optional) drops parents whose
          stored bounds can no longer beat the incumbent;
        * ``visit(compat, blocks)`` receives the parents' compatibility (n, F);
        * ``pruner.select(compat, blocks, child_blocks, flat, keys)`` receives
          every candidate refinement -- ``flat`` indexes the (n, F) parent x
          feature grid, ``keys`` are its canonical keys -- and returns a mask
          of candidates worth keeping and their bounds.
        """
        cand_labels, cand_keys, cand_bounds, pruned = [], [], [], []
        pending = 0
        for s in range(0, labels.shape[0], self.chunk):
            e = s + self.chunk
            lab_c, blk_c = labels[s:e], blocks[s:e]
            if pruner is not None:
                live = pruner.alive(bounds[s:e], blk_c)
                if not live.all():
                    lab_c, blk_c = lab_c[live], blk_c[live]
                    if blk_c.shape[0] == 0:
                        continue
            child, present, cb = self._children(lab_c)
            parent_blocks = np.repeat(blk_c, self.F)
            cbm = cb.reshape(-1, self.F)
            compat = cbm == blk_c[:, None]
            visit(compat, blk_c)

            flat = np.flatnonzero((cb > parent_blocks) & (cb <= self.tau))
            if flat.shape[0] == 0:
                continue
            lab, key = self._canonicalise(child[flat], present[flat])
            bnd = None
            if pruner is not None:
                m, bnd = pruner.select(compat, blk_c, cbm, flat, key)
                if not m.all():
                    pruned.append(key[~m])
                lab, key = lab[m], key[m]
                if key.shape[0] == 0:
                    continue
            cand_labels.append(lab)
            cand_keys.append(key)
            cand_bounds.append(bnd)
            pending += key.shape[0]

            # Periodically collapse duplicates across chunks to bound memory.
            if pending > 4 * _CHUNK_CELLS // max(1, self.S):
                cand_labels, cand_keys, cand_bounds = self._merge(cand_labels, cand_keys, cand_bounds)
                pending = cand_keys[0].shape[0]

        empty = np.empty((0, self.S))
        pruned = np.concatenate(pruned) if pruned else empty.astype(self.key_dtype)
        if not cand_keys:
            return (empty.astype(self.label_dtype), empty.astype(self.key_dtype),
                    None if pruner is None else np.empty((0, self.tau + 1), np.int64), pruned)
        cand_labels, cand_keys, cand_bounds = self._merge(cand_labels, cand_keys, cand_bounds)
        return cand_labels[0], cand_keys[0], cand_bounds[0], pruned

    @staticmethod
    def _merge(label_list, key_list, bound_list):
        """Deduplicate by key (sorted); duplicate bounds combine by minimum."""
        labels = np.concatenate(label_list)
        keys = np.concatenate(key_list)
        _, idx, inv = np.unique(_as_void(keys), return_index=True, return_inverse=True)
        bounds = None
        if bound_list[0] is not None:
            # Each parent's bound is valid on its own, so their minimum is too.
            allb = np.concatenate(bound_list)
            order = np.argsort(inv.ravel(), kind='stable')
            starts = np.searchsorted(inv.ravel()[order], np.arange(idx.shape[0]))
            bounds = np.minimum.reduceat(allb[order], starts, axis=0)
        return [labels[idx]], [keys[idx]], [bounds]


# Width of the greedy dive that seeds the branch-and-bound lower bounds.
_DIVE_WIDTH = 16


def _greedy_lower_bounds(ref, tau, count1, width):
    """
    Proven lower bounds lb[b] on the optimum with <= b blocks, from a narrow
    greedy dive: keep the `width` best partitions, refine each by every
    feature, and move to the best one-step refinements (exactly one block
    split in two). A one-step refinement's objective is exact and free:
    w(compat parent) + weight of the features inducing that refinement.
    Costs about tau * width parent expansions.
    """
    S, F = ref.S, ref.F
    lb = np.zeros(tau + 1, dtype=np.int64)
    labels = np.zeros((1, S), dtype=ref.label_dtype)
    blocks = np.ones(1, dtype=np.int64)
    while labels.shape[0]:
        child, present, cb = ref._children(labels)
        wc = (cb.reshape(-1, F) == blocks[:, None]).astype(np.int64) @ count1
        np.maximum.at(lb, blocks, wc)
        if blocks[0] >= tau:
            break
        one = np.flatnonzero(cb == np.repeat(blocks, F) + 1)
        if one.shape[0] == 0:
            break
        lab, key = ref._canonicalise(child[one], present[one])
        p_idx, f_idx = np.divmod(one, F)
        tag = np.concatenate([p_idx.astype('<u4').view(np.uint8).reshape(-1, 4),
                              key.view(np.uint8).reshape(key.shape[0], -1)], axis=1)
        _, first, inv = np.unique(_as_void(tag), return_index=True, return_inverse=True)
        obj = wc[p_idx[first]] + np.bincount(inv.ravel(), weights=count1[f_idx]).astype(np.int64)
        # The same partition reached from different parents has the same objective.
        _, uniq = np.unique(_as_void(key[first]), return_index=True)
        top = uniq[np.argsort(-obj[uniq], kind='stable')[:width]]
        labels, blocks = lab[first[top]], blocks[p_idx[first[top]]] + 1
    return np.maximum.accumulate(lb)


class _Pruner:
    def __init__(self, select, alive):
        self.select, self.alive = select, alive


def _make_pruner(ref, tau, count1, best_val):
    """
    Branch-and-bound filter for exact mode.

    Upper bound. Let D be a descendant of parent p with blocks(D) = blocks(p) + k.
    Every feature f compatible with D but not with p yields a refinement p∧f
    lying in the lattice interval [p, D]; p∧f splits each block of p in at
    most two, and there are at most 2^k - 1 such refinements besides p. So,
    grouping features by the refinement they induce (weight = summed column
    counts, cost = extra blocks),
        obj(D) <= w(features compatible with p)
                  + sum of the top (2^k - 1) group weights with cost <= k,
    and for descendants that also refine a given group g,
        obj(D) <= w(compat p) + min(C[m], C[m-1] + w_g),   m = 2^k - 1,
    where C is the cumulative sum of those sorted weights. With k = 1 this is
    exactly obj(p∧g).

    A candidate is dropped only if, for every reachable block count b, its
    bound is <= the incumbent best_val[b] (descendants come later in the
    search order than the incumbent and later partitions only win when
    strictly better) or strictly below a proven lower bound lb[b]. Results,
    including tie-breaking, are therefore identical to the unpruned search.

    Lower bounds come from the exact objectives of one-step refinements seen
    so far. (Seeding them with a beam run was measured to be slower on every
    input tried: these bounds become tight about as quickly, and the beam
    pays to score every candidate it trims.)

    Each kept candidate carries its bound for every block count b (-1 where
    b is unreachable); duplicates reached from several parents keep the
    elementwise minimum. When the candidate's turn to be expanded comes, the
    incumbent and lower bounds have usually improved, and ``alive`` skips it
    entirely -- no refinement, no canonicalisation, no recording -- if its
    stored bound no longer beats them for any b. The same ordering argument
    keeps this exact.
    """
    F = ref.F
    lb = _greedy_lower_bounds(ref, tau, count1, _DIVE_WIDTH) if _DIVE_WIDTH else np.zeros(tau + 1, dtype=np.int64)

    def select(compat, pblocks, cbm, flat, keys):
        nonlocal lb
        n = compat.shape[0]
        p_idx, f_idx = np.divmod(flat, F)
        wc = compat.astype(np.int64) @ count1                     # (n,)
        cost = cbm.ravel()[flat] - pblocks[p_idx]                  # extra blocks

        # Group candidates of the same parent that induce the same refinement.
        tag = np.concatenate(
            [p_idx.astype('<u4').view(np.uint8).reshape(-1, 4),
             keys.view(np.uint8).reshape(keys.shape[0], -1)], axis=1)
        _, first, inv = np.unique(_as_void(tag), return_index=True, return_inverse=True)
        inv = inv.ravel()
        gw = np.bincount(inv, weights=count1[f_idx]).astype(np.int64)
        cand_gw = gw[inv]

        # Exact objectives of one-step refinements are proven lower bounds.
        one = cost == 1
        if one.any():
            b1 = pblocks[p_idx[one]] + 1
            obj1 = wc[p_idx[one]] + cand_gw[one]
            np.maximum.at(lb, b1, obj1)
            lb = np.maximum.accumulate(lb)

        # Dense per-parent view: group weight / cost at the group's first feature.
        W = np.zeros((n, F), dtype=np.int64)
        Cost = np.full((n, F), tau + 1, dtype=np.int64)
        W[p_idx[first], f_idx[first]] = gw
        Cost[p_idx[first], f_idx[first]] = cost[first]

        bounds = np.full((flat.shape[0], tau + 1), -1, dtype=np.int64)
        rows = np.arange(flat.shape[0])
        cand_pb = pblocks[p_idx]
        for k in range(1, tau):
            live = (cost <= k) & (cand_pb + k <= tau)
            if not live.any():
                continue
            m = min(2 ** k - 1, F)
            top = -np.sort(-np.where(Cost <= k, W, 0), axis=1)[:, :m]
            c_m = top.sum(axis=1)
            c_m1 = c_m - top[:, m - 1]
            ub = wc[p_idx] + np.minimum(c_m[p_idx], c_m1[p_idx] + cand_gw)
            bounds[rows[live], (cand_pb + k)[live]] = ub[live]
        keep = alive(bounds, None)
        return keep, bounds[keep]

    def alive(bounds, blocks):
        # bounds[:, b] = -1 marks unreachable b, which never beats best_val >= 0
        # once the trivial partition is recorded.
        return ((bounds > best_val[None, :]) & (bounds >= lb[None, :])).any(axis=1)

    return _Pruner(select, alive)


def MSTPPartition(X, tau, topN=np.inf, prune=True):
    """
    MSTP (Max-Subset τ-Partitioning) algorithm for binary matrix clustering.

    This algorithm finds the maximum subset of columns that can be clustered
    into at most tau clusters for a given binary matrix.

    Args:
        X (np.ndarray): Binary matrix (rows × columns)
        tau (int): Maximum number of clusters allowed
        topN (int or float): Maximum number of partitions to keep at each iteration
                            (default: np.inf for no limit)
        prune (bool): Skip branches that provably cannot beat the best solution
                      found so far (exact mode only; results are unchanged).
    Returns:
        np.ndarray: Solution list (tau × M) where each row indicates which
                   columns to include for that tau value
    """
    tau = int(tau)
    Xf, X_inverse = _preprocess(np.array(X).T.copy())  # features x structures
    _, count1 = np.unique(X_inverse, return_counts=True)
    count1 = count1.astype(np.int64)
    beam = topN != np.inf

    ref = _Refiner(Xf, tau)
    S = ref.S

    # Best partition seen so far for each tau' in 1..tau (first max wins).
    best_val = np.full(tau + 1, -1, dtype=np.int64)
    best_compat = [None] * (tau + 1)

    def record(compat, blocks):
        if compat.shape[0] == 0:
            return
        obj = compat.astype(np.int64) @ count1
        for t in range(1, tau + 1):
            masked = np.where(blocks <= t, obj, -1)
            j = int(np.argmax(masked))
            if masked[j] > best_val[t]:
                best_val[t] = masked[j]
                best_compat[t] = compat[j].copy()

    def visit(compat, blocks):
        record(compat, blocks)

    pruner = _make_pruner(ref, tau, count1, best_val) if (prune and not beam) else None

    # Start from the trivial (single-block) partition.
    labels = np.zeros((1, S), dtype=ref.label_dtype)
    keys = np.full((1, S), max(S - 1, 0), dtype=ref.key_dtype)
    blocks = np.ones(1, dtype=np.int64)
    # Keys of every partition already generated (exact mode). A hash set of
    # the full key bytes: exact membership, no re-sorting of the whole history
    # each iteration.
    seen = set(_as_void(keys).tolist())
    # Upper bounds per partition and block count (pruning only); the root is
    # unbounded.
    bounds = np.full((1, tau + 1), np.iinfo(np.int64).max, dtype=np.int64) if pruner else None

    for _ in range(tau):
        if labels.shape[0] == 0:
            break
        cand_labels, cand_keys, cand_bounds, pruned = ref.expand(labels, blocks, visit, pruner, bounds)

        # Drop candidates already known: exact mode removes every partition
        # seen before (no partition is expanded twice); beam mode keeps the
        # original semantics and removes only those equal to a current parent.
        # A pruned candidate's bound covers its whole subtree, so it counts as
        # seen too: regenerating it from another parent (whose bound may be
        # looser) must not resurrect it.
        if not beam and pruned.shape[0]:
            seen.update(_as_void(pruned).tolist())
        reference = seen if not beam else set(_as_void(keys).tolist())
        # Candidates arrive unique and sorted by key; keep that order.
        idx = np.fromiter((i for i, kb in enumerate(_as_void(cand_keys).tolist()) if kb not in reference),
                          dtype=np.intp)
        labels, keys = cand_labels[idx], cand_keys[idx]
        bounds = cand_bounds[idx] if pruner else None
        del cand_labels, cand_keys, cand_bounds
        blocks = labels.max(axis=1).astype(np.int64) + 1 if labels.shape[0] else np.empty(0, np.int64)

        if beam and labels.shape[0] >= topN:
            numMatch = ref.compat(labels, blocks).sum(axis=1).astype(np.int64)
            argTop = np.argsort(numMatch * -1)[:topN]
            labels, keys, blocks = labels[argTop], keys[argTop], blocks[argTop]

        if not beam:
            seen.update(_as_void(keys).tolist())

    # The last batch is never expanded, but still competes for the objective.
    if pruner is not None and labels.shape[0]:
        live = pruner.alive(bounds, blocks)
        labels, blocks = labels[live], blocks[live]
    if labels.shape[0]:
        record(ref.compat(labels, blocks), blocks)

    solutionList = np.zeros((tau, X_inverse.shape[0]), dtype=int)
    for t in range(1, tau + 1):
        solutionList[t - 1] = best_compat[t][X_inverse]
    return solutionList


# Example usage function
def example_usage():
    """Example of how to use the MSTPPartition function."""

    # Create a test matrix
    X = np.array([
        [1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0],
        [1, 0, 1, 0, 1],
    ])

    tau = 3
    topN = np.inf

    print("Input matrix:")
    print(X)
    print(f"Shape: {X.shape}")
    print(f"Tau: {tau}")
    print(f"TopN: {topN}")

    solution = MSTPPartition(X, tau, topN)

    print(f"\nSolution:")
    print(solution)
    print(f"Solution shape: {solution.shape}")

    return solution


if __name__ == "__main__":
    # Run example when script is executed directly
    example_usage()
