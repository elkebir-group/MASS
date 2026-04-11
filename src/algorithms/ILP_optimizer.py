import numpy as np
import gurobipy as gp
from gurobipy import GRB, quicksum
from typing import Dict, List, Tuple, Optional, Union
import time
from contextlib import redirect_stdout
import io
import os
import sys

# Add parent directory to path to import gurobi_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from gurobi_config import GUROBI_WLS_CONFIG, USE_LOCAL_LICENSE
except ImportError:
    # Fallback if config file doesn't exist
    GUROBI_WLS_CONFIG = None
    USE_LOCAL_LICENSE = True


class BinaryMatrixILPOptimizer:
    """
    A reusable ILP optimizer for binary matrix partitioning problems.
    
    This class provides a clean interface for solving the binary matrix partitioning
    problem using Gurobi optimization solver with configurable parameters.
    """
    
    def __init__(self, 
                 connection_params: Optional[Dict] = None,
                 time_limit: Optional[int] = None,
                 output_flag: bool = False,
                 debug: bool = False,
                 use_pre_aggregation: bool = True,
                 objn_reltol: Optional[float] = None,
                 objn_abstol: Optional[float] = None):
        """
        Initialize the ILP optimizer.
        
        Args:
            connection_params: Gurobi connection parameters (WLS, Cloud, etc.)
            time_limit: Maximum solve time in seconds (None for no limit)
            output_flag: Whether to show solver output
            debug: Whether to enable debug output
            use_pre_aggregation: Whether to pre-aggregate identical rows
            objn_reltol: Optional multi-objective relative tolerance for primary objective locking
            objn_abstol: Optional multi-objective absolute tolerance for primary objective locking
        """
        # Get connection params: use provided, or load from config, or None (local license)
        if connection_params is None:
            connection_params = self._get_default_connection_params()
        self.connection_params = connection_params
        self.time_limit = time_limit
        self.output_flag = output_flag
        self.debug = debug
        self.use_pre_aggregation = use_pre_aggregation
        self.objn_reltol = objn_reltol
        self.objn_abstol = objn_abstol
    
    def _get_default_connection_params(self) -> Optional[Dict]:
        """
        Get default connection parameters for Gurobi.
        
        Loads from gurobi_config.py if available, otherwise returns None
        to use local license file.
        """
        if USE_LOCAL_LICENSE:
            return None  # Use local license file
        
        if GUROBI_WLS_CONFIG is not None:
            return GUROBI_WLS_CONFIG
        
        # Fallback: return None to use local license
        return None
    
    def _pre_aggregate_rows(self, matrix: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Pre-aggregate identical rows to reduce problem size.
        
        Args:
            matrix: Original binary matrix
            
        Returns:
            Tuple of (aggregated_matrix, row_mapping)
        """
        if not self.use_pre_aggregation:
            return matrix, {i: [i] for i in range(matrix.shape[0])}
        
        # Find unique rows and their multiplicities
        unique_rows = []
        row_mapping = {}  # Maps aggregated row index to original row indices
        
        seen_rows = {}
        for i, row in enumerate(matrix):
            row_tuple = tuple(row)
            if row_tuple in seen_rows:
                # Add to existing aggregated row
                agg_idx = seen_rows[row_tuple]
                row_mapping[agg_idx].append(i)
            else:
                # Create new aggregated row
                agg_idx = len(unique_rows)
                unique_rows.append(row)
                row_mapping[agg_idx] = [i]
                seen_rows[row_tuple] = agg_idx
        
        aggregated_matrix = np.array(unique_rows)
        
        if self.debug:
            print(f"Pre-aggregation: {matrix.shape[0]} rows → {aggregated_matrix.shape[0]} unique rows")
            print(f"Reduction: {matrix.shape[0] - aggregated_matrix.shape[0]} duplicate rows removed")
        
        return aggregated_matrix, row_mapping
    
    def solve(self, matrix: np.ndarray, tau: int, weights: Optional[np.ndarray] = None) -> Dict:
        """
        Solve the binary matrix partitioning problem.
        
        Args:
            matrix: Binary matrix (m x n)
            tau: Maximum number of representatives
            weights: Optional column weights (n,) - if None, uses unweighted objective
            
        Returns:
            Dictionary containing:
            - selected_columns: List of selected column indices
            - cluster_map: Dictionary mapping representatives to assigned rows
            - objective_value: Objective function value (weighted or unweighted)
            - unweighted_objective: Number of selected columns (always included)
            - runtime: Solve time in seconds
            - status: Solver status
            - error_message: Error message if solve failed
            - weights: Column weights used (None if unweighted)
        """
        start_time = time.time()
        
        # Validate weights if provided
        if weights is not None:
            if matrix.shape[1] != len(weights):
                raise ValueError(f"Matrix columns ({matrix.shape[1]}) must match weights length ({len(weights)})")
            if np.any(weights < 0):
                raise ValueError("All weights must be non-negative")
        
        # Store original matrix shape
        original_matrix_shape = matrix.shape
        
        try:
            # Use connection params if available (WLS), otherwise use default (local license)
            env_params = self.connection_params if self.connection_params is not None else {}
            with gp.Env(params=env_params) as env:
                # Pre-aggregate identical rows
                aggregated_matrix, row_mapping = self._pre_aggregate_rows(matrix)
                pre_aggregation_shape = aggregated_matrix.shape
                
                result = self._solve_with_env(aggregated_matrix, tau, env, weights)
                
                # Expand cluster assignments to original rows
                if self.use_pre_aggregation and 'cluster_map' in result:
                    expanded_cluster_map = {}
                    for rep, agg_rows in result['cluster_map'].items():
                        expanded_cluster_map[rep] = []
                        for agg_row in agg_rows:
                            expanded_cluster_map[rep].extend(row_mapping[agg_row])
                    result['cluster_map'] = expanded_cluster_map
                
                # Add matrix metadata
                result['original_matrix_shape'] = original_matrix_shape
                result['pre_aggregation_shape'] = pre_aggregation_shape
                result['row_mapping'] = row_mapping
                result['duplicates_removed'] = original_matrix_shape[0] - pre_aggregation_shape[0]
                result['runtime'] = time.time() - start_time
                result['weights'] = weights
                return result
                
        except Exception as e:
            return {
                'selected_columns': [],
                'cluster_map': {},
                'objective_value': 0,
                'unweighted_objective': 0,
                'runtime': time.time() - start_time,
                'status': 'ERROR',
                'error_message': str(e),
                'original_matrix_shape': original_matrix_shape,
                'pre_aggregation_shape': original_matrix_shape,  # No aggregation on error
                'row_mapping': {i: [i] for i in range(original_matrix_shape[0])},
                'duplicates_removed': 0,
                'weights': weights
            }
    
    def _solve_with_env(self, matrix: np.ndarray, tau: int, env: gp.Env, weights: Optional[np.ndarray] = None) -> Dict:
        """Solve the problem with a given Gurobi environment."""
        m, n = matrix.shape
        
        # Capture output if needed
        output_capture = io.StringIO()
        redirect_context = redirect_stdout(output_capture) if not self.output_flag and not self.debug else None
        
        try:
            if redirect_context:
                redirect_context.__enter__()
                
            with gp.Model(env=env) as model:
                # Set solver parameters
                model.Params.OutputFlag = 1 if self.output_flag else 0
                if self.time_limit is not None:
                    model.Params.TimeLimit = self.time_limit
                # Note: ObjNRelTol and ObjNAbsTol are not available in Gurobi 12.0.3
                # Available multi-objective parameters: MultiObjMethod, MultiObjPre, ObjNumber, ObjScale, ZeroObjNodes
                # if self.objn_reltol is not None:
                #     model.setParam(GRB.Param.ObjNRelTol, self.objn_reltol)
                # if self.objn_abstol is not None:
                #     model.setParam(GRB.Param.ObjNAbsTol, self.objn_abstol)
                
                # Decision variables
                x = model.addVars(n, vtype=GRB.BINARY, name="x")
                y = model.addVars(m, vtype=GRB.BINARY, name="y")
                z = model.addVars(m, m, vtype=GRB.BINARY, name="z")
                w = model.addVars(m, m, vtype=GRB.CONTINUOUS, name="w")

                # Objective: column selection only
                column_objective = quicksum((weights[j] if weights is not None else 1.0) * x[j] for j in range(n))
                model.setObjective(column_objective, GRB.MAXIMIZE)

                # At most tau representatives
                model.addConstr(quicksum(y[i] for i in range(m)) <= tau, "MaxTau")

                # Mismatch witness constraints
                for i in range(m):
                    for i2 in range(m):
                        for j in range(n):
                            a_ij, a_i2j = matrix[i, j], matrix[i2, j]
                            model.addConstr(w[i, i2] >= (a_ij - a_i2j) * x[j])
                            model.addConstr(w[i, i2] >= (a_i2j - a_ij) * x[j])

                # Each row assigned to exactly one representative
                for i2 in range(m):
                    model.addConstr(quicksum(z[i, i2] for i in range(m)) == 1)

                # Consistency constraints
                for i in range(m):
                    for i2 in range(m):
                        model.addConstr(z[i, i2] + y[i] - 1 <= 1 - w[i, i2])
                        if i != i2:
                            model.addConstr(z[i, i2] <= 1 - w[i, i2])
                        model.addConstr(z[i, i2] <= y[i])
                        model.addConstr(z[i, i2] >= y[i] - w[i, i2])
                
                # Solve the model
                model.optimize()
                
                # Extract solution - try to get incumbent solution even if not optimal
                # Check if we have any solution available (including non-optimal/time-limited solutions)
                has_solution = model.SolCount > 0
                
                if self.debug:
                    print(f"DEBUG: Model status = {model.status} (GRB.TIME_LIMIT = {GRB.TIME_LIMIT}, GRB.OPTIMAL = {GRB.OPTIMAL})")
                    print(f"DEBUG: SolCount = {model.SolCount}, has_solution = {has_solution}")
                
                # Determine if status indicates we should try to extract solution
                # These statuses may have feasible solutions:
                # - GRB.OPTIMAL: Optimal solution found
                # - GRB.TIME_LIMIT: Time limit reached (has incumbent)
                # - GRB.INTERRUPTED: Optimization interrupted (may have incumbent)
                # - GRB.SOLUTION_LIMIT: Solution limit reached (has solution)
                # - GRB.USER_OBJ_LIMIT: User objective limit reached (has solution)
                statuses_with_solution = {
                    GRB.OPTIMAL,
                    GRB.TIME_LIMIT,
                    GRB.INTERRUPTED,
                    GRB.SOLUTION_LIMIT,
                    GRB.USER_OBJ_LIMIT,
                }
                
                # Check if we should extract solution
                should_extract = model.status in statuses_with_solution
                
                if should_extract and has_solution:
                    # Extract incumbent solution (may be optimal or non-optimal)
                    # Gurobi stores the best found solution as the incumbent when time limit is reached
                    selected_columns = [j for j in range(n) if x[j].X > 0.5]
                    
                    # Get objective value from incumbent solution
                    objective_value = model.ObjVal if has_solution else 0.0
                    unweighted_objective = sum(1 for j in selected_columns)
                    
                    # Cluster assignment via z[i, i']
                    cluster_map = {}
                    if has_solution:
                        for i2 in range(m):
                            # find the representative assigned to i2
                            rep_found = None
                            for i in range(m):
                                z_val = model.getVarByName(f"z[{i},{i2}]").X
                                if z_val > 0.5:
                                    rep_found = i
                                    break
                            if rep_found is not None:
                                cluster_map.setdefault(rep_found, []).append(i2)
                    
                    # Determine status string
                    if model.status == GRB.OPTIMAL:
                        status_str = 'OPTIMAL'
                    elif model.status == GRB.TIME_LIMIT:
                        status_str = 'TIME_LIMIT'
                    elif model.status == GRB.INTERRUPTED:
                        status_str = 'INTERRUPTED'
                    elif model.status == GRB.SOLUTION_LIMIT:
                        status_str = 'SOLUTION_LIMIT'
                    elif model.status == GRB.USER_OBJ_LIMIT:
                        status_str = 'USER_OBJ_LIMIT'
                    else:
                        status_str = f'STATUS_{model.status}'
                    
                    result = {
                        'selected_columns': selected_columns,
                        'cluster_map': cluster_map,
                        'objective_value': objective_value,
                        'unweighted_objective': unweighted_objective,
                        'status': status_str,
                        'gurobi_status': model.status,
                        'error_message': None,
                        'is_optimal': (model.status == GRB.OPTIMAL),
                        'has_incumbent': has_solution
                    }
                    
                    if self.debug and has_solution:
                        optimality_str = "OPTIMAL" if model.status == GRB.OPTIMAL else "NON-OPTIMAL (incumbent)"
                        print(f"\nSolution status: {optimality_str}")
                        print(f"Selected columns: {selected_columns}")
                        if weights is not None:
                            print(f"Weighted primary objective: {objective_value:.6f}")
                        else:
                            print(f"Primary objective (|S|): {unweighted_objective}")
                        print("Clusters (rep → rows):")
                        for rep, members in result['cluster_map'].items():
                            print(f"  {rep}: {members}")
                    
                    return result
                
                # No feasible solution available
                # Check if we have a solution but status doesn't match expected ones
                if has_solution:
                    # Try to extract anyway - sometimes solutions exist even with unexpected statuses
                    try:
                        selected_columns = [j for j in range(n) if x[j].X > 0.5]
                        objective_value = model.ObjVal if has_solution else 0.0
                        return {
                            'selected_columns': selected_columns,
                            'cluster_map': {},  # May not be able to extract cluster map
                            'objective_value': objective_value,
                            'unweighted_objective': len(selected_columns),
                            'status': f'STATUS_{model.status}',
                            'gurobi_status': model.status,
                            'error_message': f"Unexpected status {model.status} but solution found",
                            'is_optimal': False,
                            'has_incumbent': True
                        }
                    except:
                        pass  # Fall through to error case
                
                # No solution available - but still report status correctly
                # Determine status string even when no solution
                if model.status == GRB.TIME_LIMIT:
                    status_str = 'TIME_LIMIT'
                    error_msg = "Time limit reached (no solution found yet)"
                elif model.status == GRB.INTERRUPTED:
                    status_str = 'INTERRUPTED'
                    error_msg = "Optimization interrupted (no solution found)"
                elif model.status == GRB.SOLUTION_LIMIT:
                    status_str = 'SOLUTION_LIMIT'
                    error_msg = "Solution limit reached (no solution found)"
                else:
                    status_str = f'STATUS_{model.status}'
                    error_msg = f"Solver status: {model.status} (no solution found)"
                
                return {
                    'selected_columns': [],
                    'cluster_map': {},
                    'objective_value': 0.0,
                    'unweighted_objective': 0,
                    'status': status_str,
                    'gurobi_status': model.status,
                    'error_message': error_msg,
                    'is_optimal': False,
                    'has_incumbent': False
                }
                    
        finally:
            if redirect_context:
                redirect_context.__exit__(None, None, None)


# Legacy function wrapper for backward compatibility
def implement_ilp_gurobi_remote(matrix: np.ndarray, tau: int, env: gp.Env, weights: Optional[np.ndarray] = None) -> Tuple[List[int], Dict]:
    """
    Legacy function wrapper for backward compatibility.
    
    Args:
        matrix: Binary matrix
        tau: Maximum number of representatives
        env: Gurobi environment
        weights: Optional column weights
        
    Returns:
        Tuple of (selected_columns, cluster_map)
    """
    optimizer = BinaryMatrixILPOptimizer(output_flag=True, debug=True)
    result = optimizer._solve_with_env(matrix, tau, env, weights)
    return result['selected_columns'], result['cluster_map']


def main():
    """Example usage of the BinaryMatrixILPOptimizer with column-only objective."""
    # Example matrix
    np.random.seed(1)
    matrix = np.random.randint(0, 2, size=(6, 8))  # m=6 rows, n=8 columns
    weights = np.linspace(0.1, 1.0, 8)  # Different weights for each column
    tau = 2

    # Create optimizer (column-only objective)
    optimizer = BinaryMatrixILPOptimizer(
        debug=True,
        output_flag=True,
        use_pre_aggregation=True,
    )
    if optimizer.debug:
        print("Using column-only objective")
    print("=== COLUMN-ONLY ILP ===")
    result = optimizer.solve(matrix, tau, weights=weights)
    print(f"Status: {result['status']}")
    print(f"Runtime: {result['runtime']:.3f} seconds")
    print(f"Selected columns: {result['selected_columns']}")
    print(f"Primary objective (columns): {result['objective_value']:.6f}")
    print(f"Unweighted objective: {result['unweighted_objective']}")
    
    if result['error_message']:
        print(f"Error: {result['error_message']}")


if __name__ == "__main__":
    main()