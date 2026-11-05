"""
Run the ILP for one tau and convert its result to the analyzer's result dict.

Kept out of MASS.py so that ILP changes stay in the ILP's own files.
`self` is the FastaTauAnalyzer.
"""
import gc
import os
import sys
import time
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.ILP_optimizer import BinaryMatrixILPOptimizer  # noqa: E402
from memory_monitor import MemoryMonitor  # noqa: E402


def run_ilp(self, matrix: np.ndarray, tau: int, weights: Optional[np.ndarray] = None) -> Dict:
    """
    Run ILP algorithm on matrix with given tau.

    Args:
        matrix: Binary matrix
        tau: Maximum number of partitions
        weights: Optional column weights for weighted ILP

    Returns:
        Dictionary with results
    """
    if self.debug:
        print(f"  Running ILP algorithm with tau={tau}")

    start_time = time.time()

    # Initialize memory monitoring
    memory_monitor = None
    if self.track_memory:
        memory_monitor = MemoryMonitor()
        memory_monitor.start_monitoring()

    try:
        # Use the original BinaryMatrixILPOptimizer
        optimizer = BinaryMatrixILPOptimizer(
            time_limit=self.time_limit,
            debug=self.debug,
            use_pre_aggregation=False,  # Pre-aggregation now done at matrix creation level
        )
        result = optimizer.solve(matrix, tau=tau, weights=weights)
        runtime = time.time() - start_time

        # Stop memory monitoring and get peak memory
        peak_memory_mb = 0
        if memory_monitor:
            memory_monitor.stop_monitoring()
            peak_memory_mb = memory_monitor.get_peak_memory_mb()

        # Calculate pattern coverage if we have a solution
        selected_columns = result.get('selected_columns', [])
        cluster_map = result.get('cluster_map', {})
        if len(selected_columns) > 0 and len(cluster_map) > 0:
            pattern_coverage_result = self._calculate_pattern_coverage(matrix, selected_columns, cluster_map, weights)
            result['pattern_info'] = pattern_coverage_result.get('pattern_info', {})
        else:
            result['pattern_info'] = {}

        # Add runtime and algorithm info
        # Handle status: OPTIMAL = SUCCESS, TIME_LIMIT/INTERRUPTED/etc with solution = PARTIAL_SUCCESS, others = FAILED
        original_status = result.get('status', 'UNKNOWN')
        has_solution = result.get('has_incumbent', False) or len(result.get('selected_columns', [])) > 0

        if original_status == 'OPTIMAL':
            status = 'SUCCESS'
        elif original_status in ['TIME_LIMIT', 'INTERRUPTED', 'SOLUTION_LIMIT', 'USER_OBJ_LIMIT'] and has_solution:
            # Non-optimal but has a valid solution
            status = 'PARTIAL_SUCCESS'
            # Preserve original status for reference
            result['gurobi_status'] = original_status
            result['is_optimal'] = False
        else:
            status = 'FAILED'

        result.update({
            'algorithm': 'ILP',
            'runtime': runtime,
            'status': status
        })

        # Add memory usage if tracked (always include for timeouts)
        if self.track_memory or status == 'TIME_LIMIT':
            result['peak_memory_mb'] = peak_memory_mb

        # If the algorithm failed (no solution), capture the reason
        if status == 'FAILED':
            result['error_message'] = result.get('error_message', f"Algorithm failed with status: {original_status}")

        # Clean up memory after algorithm completion
        del optimizer
        gc.collect()

        return result

    except Exception as e:
        runtime = time.time() - start_time

        # Stop memory monitoring and get peak memory
        peak_memory_mb = 0
        if memory_monitor:
            memory_monitor.stop_monitoring()
            peak_memory_mb = memory_monitor.get_peak_memory_mb()

        import traceback
        error_details = f"Exception: {str(e)}\nTraceback: {traceback.format_exc()}"
        algorithm_name = 'ILP'
        result = {
            'algorithm': algorithm_name,
            'runtime': runtime,
            'status': 'ERROR',
            'error_message': error_details,
            'selected_columns': [],
            'num_clusters': 0,
            'objective_value': 0,
            'cluster_map': {}
        }

        # Add memory usage if tracked (always include for timeouts)
        if self.track_memory or status == 'TIME_LIMIT':
            result['peak_memory_mb'] = peak_memory_mb

        return result
