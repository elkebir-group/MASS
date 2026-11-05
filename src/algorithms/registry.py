"""
Algorithm registry: the one place that lists what `--algorithm` can run.

Adding an algorithm means writing its module and adding one entry to
ALGORITHMS below; MASS.py has no per-algorithm code.

Most algorithms are *partition solvers*: a function
``solve(matrix, tau, weights) -> (tau x M) 0/1 array`` whose row t-1 marks the
columns kept with at most t groups (the MSTPPartition contract). MASS.py turns
that into clusters, coverage, timing, memory and error handling. An algorithm
with its own result format (the ILP) supplies a ``run`` instead.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Error-message fragments that mark a run as out of memory or out of time.
MEMORY_KEYWORDS = (
    'memory', 'out of memory', 'allocation', 'insufficient',
    'cannot allocate', 'memoryerror', 'oom', 'memory limit',
    'index out of bounds', 'list assignment index out of range',
)
ILP_MEMORY_KEYWORDS = MEMORY_KEYWORDS + ('gurobi memory', 'solver memory', 'optimization memory')
TIMEOUT_KEYWORDS = (
    'timeout', 'time limit', 'timelimit', 'time limit exceeded',
    'gurobi timeout', 'solver timeout', 'optimization timeout',
    'maximum time', 'time exceeded', 'timed out',
)


@dataclass(frozen=True)
class Variant:
    """One configured run of an algorithm; it gets its own CSV rows."""
    label: str                                   # the CSV `algorithm` value, e.g. "MSTP-BEAM-5"
    family: str                                  # internal result name, e.g. "MSTP-BEAM"
    solve: Optional[Callable] = None             # partition solver (matrix, tau, weights) -> (tau x M)
    run: Optional[Callable] = None               # custom runner (analyzer, matrix, tau, weights) -> result dict
    extra: Dict = field(default_factory=dict)    # extra keys stored in the internal result


@dataclass(frozen=True)
class AlgorithmSpec:
    name: str                                    # the `--algorithm` value
    help: str
    variants: Callable[[object], List[Variant]]  # analyzer -> variants to run for each tau
    # One run at the largest tau answers every smaller tau exactly. True only
    # for exact solvers: a heuristic's answer depends on tau.
    reuse_max_tau: bool = False
    # After a failure, skip the remaining taus of this algorithm:
    #   "memory"  -- an ERROR/FAILED run whose message names memory
    #   "timeout" -- an ERROR/FAILED run that timed out (unless --ilp-continue-on-timeout)
    skip_policy: str = "memory"
    summary_name: str = ""                       # label in the log summary
    success_statuses: Tuple[str, ...] = ("SUCCESS",)
    describe: Callable[[object], str] = lambda args: ""   # banner detail from CLI args


def _mstp_exact(matrix, tau, weights):
    from algorithms.mstp_partitioner import MSTPPartition
    return MSTPPartition(matrix, tau, np.inf)


def _mstp_beam(width):
    def solve(matrix, tau, weights):
        from algorithms.mstp_partitioner import MSTPPartition
        return MSTPPartition(matrix, tau, width)
    return solve


def _brute_force(matrix, tau, weights):
    from algorithms.brute_force import brute_force_partition
    return brute_force_partition(matrix, tau)


def _run_ilp(analyzer, matrix, tau, weights):
    from algorithms.ilp_runner import run_ilp
    return run_ilp(analyzer, matrix, tau, weights)


ALGORITHMS: Dict[str, AlgorithmSpec] = {}


def register(spec: AlgorithmSpec) -> AlgorithmSpec:
    if spec.name in ALGORITHMS:
        raise ValueError(f"algorithm {spec.name!r} registered twice")
    ALGORITHMS[spec.name] = spec
    return spec


register(AlgorithmSpec(
    name="ilp",
    help="Integer Linear Programming solver (needs a Gurobi license)",
    variants=lambda analyzer: [Variant("ILP", "ILP", run=_run_ilp)],
    skip_policy="timeout",
    summary_name="ILP",
    success_statuses=("SUCCESS", "PARTIAL_SUCCESS"),
    describe=lambda args: f"  ILP: Weighted={args.use_weighted_ilp}",
))

register(AlgorithmSpec(
    name="mstp",
    help="Max-Subset tau-Partitioning, exact",
    variants=lambda analyzer: [Variant("MSTP", "MSTP", solve=_mstp_exact)],
    reuse_max_tau=True,
    summary_name="MSTP",
    describe=lambda args: f"  MSTP: Pre-aggregation={not args.no_pre_aggregation}",
))

register(AlgorithmSpec(
    name="mstp_beam",
    help="MSTP with beam search, one run per --beam-value",
    variants=lambda analyzer: [
        Variant(f"MSTP-BEAM-{w}", "MSTP-BEAM", solve=_mstp_beam(w), extra={"beam_value": w})
        for w in analyzer.beam_values
    ],
    summary_name="MSTP-BEAM",
    describe=lambda args: f"  MSTP-BEAM: Beam values={args.beam_value}",
))

register(AlgorithmSpec(
    name="brute_force",
    help="Exhaustive search over every partition; reference for small inputs (<= 10 rows)",
    variants=lambda analyzer: [Variant("BRUTE-FORCE", "BRUTE-FORCE", solve=_brute_force)],
    reuse_max_tau=True,
    summary_name="BRUTE-FORCE",
))


def get(name: str) -> AlgorithmSpec:
    return ALGORITHMS[name]


def names() -> List[str]:
    return list(ALGORITHMS)
