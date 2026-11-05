#!/usr/bin/env python3
"""
Script to run MSTP algorithm, ILP, or MSTP-BEAM on structure files with tau values or ranges,
and save results to CSV files.

This script allows users to:
1. Process single structure files or directories of structure files (supports FASTA, text, JSON, CSV formats)
2. Run algorithms on single tau values or ranges of tau values
3. Compare MSTP algorithm vs ILP results
4. Save comprehensive results to CSV files

Usage:
    # Single structure file, single tau
    python src/MASS.py --input input.fasta --tau 3 --output results.csv
    
    # Single structure file, tau range
    python src/MASS.py --input input.fasta --tau-range 2 5 --output results.csv
    
    # Directory of structure files, single tau
    python src/MASS.py --input-dir /path/to/structure/files --tau 3 --output results.csv
    
    # Directory of structure files, tau range
    python src/MASS.py --input-dir /path/to/structure/files --tau-range 2 8 --output results.csv
"""

import argparse
import numpy as np
import pandas as pd
import time
import sys
import os
import gc
import csv
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import glob
import psutil

# Import our algorithms
from algorithms import registry as algorithm_registry
from memory_monitor import MemoryMonitor
from matrix_input import load_matrix
from algorithms.parse_structures import parse_fasta_file, universal_extract_base_pairs, create_structure_matrix, universal_load_structure_units_matrix_from_fasta, universal_load_structure_units_matrix_with_weights_from_fasta, universal_load_base_pair_matrix_from_fasta


class FastaTauAnalyzer:
    """
    Analyzer class for running algorithms on FASTA files with different tau values.
    """
    
    def __init__(self, debug: bool = False, time_limit: Optional[int] = None, memory_monitoring: bool = False,
                 pre_aggregation: bool = True,
                 use_structure_units: bool = True, use_weighted_ilp: bool = True,
                 track_memory: bool = False,
                 beam_value: int = 5, beam_values: Optional[List[int]] = None, skip_invalid_tau: bool = True,
                 ilp_continue_on_timeout: bool = False, mstp_extract_from_max_tau: bool = True,
                 detailed_output: bool = False):
        """
        Initialize the analyzer.
        
        Args:
            debug: Enable debug output
            time_limit: Time limit for ILP solver (seconds)
            memory_monitoring: Enable memory monitoring for MSTP algorithm
            pre_aggregation: Merge identical rows before running (default: True); results
                             are the same either way, only the problem size changes
            use_structure_units: Use structure units matrix instead of base pairs matrix (default: True)
            use_weighted_ilp: Use weighted ILP optimization based on nucleotide counts (default: True)
            track_memory: Track peak memory usage for all algorithms (default: False)
            beam_value: Beam width for MSTP-BEAM algorithm (default: 5)
            skip_invalid_tau: Skip tau values greater than the number of unique structures (default: True)
            ilp_continue_on_timeout: If True, do not skip subsequent taus when ILP hits time limit
            mstp_extract_from_max_tau: If True, run MSTP algorithm once for max tau and extract solutions for all tau values (default: True)
        """
        self.debug = debug
        self.time_limit = time_limit
        self.memory_monitoring = memory_monitoring
        self.pre_aggregation = pre_aggregation
        self.use_structure_units = use_structure_units
        self.use_weighted_ilp = use_weighted_ilp
        self.track_memory = track_memory
        self.beam_value = beam_value  # Default/primary beam value (for backward compatibility)
        self.beam_values = beam_values if beam_values is not None else [beam_value]  # All beam values to run
        self.skip_invalid_tau = skip_invalid_tau
        self.results = []
        self.ilp_continue_on_timeout = ilp_continue_on_timeout
        self.mstp_extract_from_max_tau = mstp_extract_from_max_tau
        self.detailed_output = detailed_output
        # 'structures' (RNA inputs) or 'matrix' (--matrix: a plain 0/1 table)
        self.input_kind = 'structures'
        self.weights_file = None
        
    def load_matrix_from_fasta(self, fasta_file: str, max_structures: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Load structure matrix from input file. Supports multiple formats:
        - FASTA: >id\nseq\nstructure
        - Text: >id\nstructure (no sequence)
        - JSON: {id: structure}
        - CSV: id,structure columns
        
        Args:
            fasta_file: Path to input file (any supported format)
            max_structures: Maximum number of structures to process
            
        Returns:
            Tuple of (matrix, metadata)
        """
        if self.debug:
            print(f"Loading structure file: {fasta_file}")
            
        try:
            # Use universal_load_base_pair_matrix_from_fasta for consistency with structure units mode
            matrix, unique_pairs, valid_structures, id_to_row_mapping, original_matrix_shape, pre_aggregation_shape, row_mapping, duplicates_removed, gapped_mappings = universal_load_base_pair_matrix_from_fasta(
                fasta_file, max_structures=max_structures, use_pre_aggregation=self.pre_aggregation)
            
            if self.debug:
                print(f"  Loaded {len(valid_structures)} structures")
            
            metadata = {
                'input_file': fasta_file,
                'num_sequences': len(valid_structures),
                'num_structures': len(valid_structures),
                'matrix_shape': matrix.shape,
                'sparsity': 1 - np.sum(matrix) / matrix.size if matrix.size > 0 else 0,
                'gapped_mappings': gapped_mappings
            }
            
            # Add detailed metadata only if detailed output is enabled
            if self.detailed_output:
                metadata.update({
                    'column_mapping': unique_pairs,  # Column index to base pair mapping
                    'id_to_row_mapping': id_to_row_mapping,  # Sequence ID to row index mapping
                    'original_matrix_shape': original_matrix_shape,
                    'pre_aggregation_shape': pre_aggregation_shape,
                    'row_mapping': row_mapping,
                    'duplicates_removed': duplicates_removed
                })
            else:
                # Still need these for internal processing, but don't add to metadata
                metadata['_column_mapping'] = unique_pairs
                metadata['_id_to_row_mapping'] = id_to_row_mapping
                metadata['_original_matrix_shape'] = original_matrix_shape
                metadata['_pre_aggregation_shape'] = pre_aggregation_shape
                metadata['_row_mapping'] = row_mapping
                metadata['_duplicates_removed'] = duplicates_removed
            
            if self.debug:
                print(f"  Matrix shape: {matrix.shape}")
                print(f"  Sparsity: {metadata['sparsity']:.3f}")
                
            return matrix, metadata
            
        except Exception as e:
            print(f"❌ Error loading structure file {fasta_file}: {e}")
            raise
    
    def load_structure_units_matrix_from_fasta(self, fasta_file: str, max_structures: Optional[int] = None, 
                                             include_stem_runs: bool = False) -> Tuple[np.ndarray, Dict]:
        """
        Load structure units matrix from input file. Supports multiple formats:
        - FASTA: >id\nseq\nstructure
        - Text: >id\nstructure (no sequence)
        - JSON: {id: structure}
        - CSV: id,structure columns
        
        Stacking pairs are always included.
        
        Args:
            fasta_file: Path to input file (any supported format)
            max_structures: Maximum number of structures to process
            include_stem_runs: Whether to include stem runs
            
        Returns:
            Tuple of (matrix, metadata)
        """
        if self.debug:
            print(f"Loading structure units matrix from file: {fasta_file}")
            
        try:
            # Use the new function from parse_structures
            matrix, unique_units, valid_structures, id_to_row_mapping, original_matrix_shape, pre_aggregation_shape, row_mapping, duplicates_removed, gapped_mappings = universal_load_structure_units_matrix_from_fasta(
                fasta_file, max_structures=max_structures, 
                include_stem_runs=include_stem_runs, use_pre_aggregation=self.pre_aggregation)
            
            metadata = {
                'input_file': fasta_file,
                'num_sequences': len(valid_structures),
                'num_structures': len(valid_structures),
                'matrix_shape': matrix.shape,
                'sparsity': 1 - np.sum(matrix) / matrix.size if matrix.size > 0 else 0,
                'gapped_mappings': gapped_mappings
            }
            
            # Add detailed metadata only if detailed output is enabled
            if self.detailed_output:
                metadata.update({
                    'include_stacking': True,  # Always True now
                    'include_stem_runs': include_stem_runs,
                    'column_mapping': unique_units,  # Column index to structure unit mapping
                    'id_to_row_mapping': id_to_row_mapping,  # Sequence ID to row index mapping
                    'original_matrix_shape': original_matrix_shape,
                    'pre_aggregation_shape': pre_aggregation_shape,
                    'row_mapping': row_mapping,
                    'duplicates_removed': duplicates_removed
                })
            else:
                # Still need these for internal processing, but don't add to metadata
                metadata['_include_stacking'] = True  # Always True now
                metadata['_include_stem_runs'] = include_stem_runs
                metadata['_column_mapping'] = unique_units
                metadata['_id_to_row_mapping'] = id_to_row_mapping
                metadata['_original_matrix_shape'] = original_matrix_shape
                metadata['_pre_aggregation_shape'] = pre_aggregation_shape
                metadata['_row_mapping'] = row_mapping
                metadata['_duplicates_removed'] = duplicates_removed
            
            if self.debug:
                print(f"  Matrix shape: {matrix.shape}")
                print(f"  Sparsity: {metadata['sparsity']:.3f}")
                print(f"  Unique structure units: {len(unique_units)}")
                
            return matrix, metadata
            
        except Exception as e:
            print(f"❌ Error loading structure units matrix from file {fasta_file}: {e}")
            raise
    
    @staticmethod
    def _clusters_for_tau(solution_matrix: np.ndarray, matrix: np.ndarray, tau: int) -> Tuple[List[int], Dict]:
        """Selected columns for `tau` (row tau-1 of the solution) and the clusters
        they induce: cluster_id -> row indices sharing one pattern on those columns."""
        best_solution = solution_matrix[tau - 1]  # tau-1 because 0-indexed
        selected_columns = np.where(best_solution == 1)[0].tolist()
        cluster_map = {}
        if len(selected_columns) > 0:
            submatrix = matrix[:, selected_columns]
            unique_patterns, pattern_indices = np.unique(submatrix, axis=0, return_inverse=True)
            for row_idx, pattern_idx in enumerate(pattern_indices):
                if pattern_idx not in cluster_map:
                    cluster_map[pattern_idx] = []
                cluster_map[pattern_idx].append(row_idx)
        return selected_columns, cluster_map

    def _extract_partition_solution_for_tau(self, solution_matrix: np.ndarray, matrix: np.ndarray, 
                                            tau: int, weights: Optional[np.ndarray] = None) -> Dict:
        """
        Extract partition solution for a specific tau value from the full solution matrix.
        
        Args:
            solution_matrix: Full solution matrix from MSTPPartition (tau_max x M)
            matrix: Original binary matrix
            tau: Specific tau value to extract
            weights: Optional column weights for weighted coverage calculation
            
        Returns:
            Dictionary with results for the specific tau value
        """
        selected_columns = []
        cluster_map = {}
        
        # Get the solution for this specific tau (tau-1 because 0-indexed)
        if solution_matrix.shape[0] >= tau:
            selected_columns, cluster_map = self._clusters_for_tau(solution_matrix, matrix, tau)
        
        # Calculate objective value (number of selected columns)
        objective_value = len(selected_columns)
        
        # Calculate pattern coverage for each unique pattern
        pattern_coverage_result = self._calculate_pattern_coverage(matrix, selected_columns, cluster_map, weights)
        pattern_info = pattern_coverage_result.get('pattern_info', {})
        
        return {
            'algorithm': 'Partition',
            'selected_columns': selected_columns,
            'num_clusters': len(cluster_map),
            'objective_value': objective_value,
            'cluster_map': cluster_map,
            'pattern_info': pattern_info
        }
    
    def _calculate_pattern_coverage(self, matrix: np.ndarray, selected_columns: List[int], 
                                     cluster_map: Dict, weights: Optional[np.ndarray] = None) -> Dict:
        """
        Calculate weighted coverage for each unique pattern.
        
        Args:
            matrix: Binary matrix
            selected_columns: List of selected column indices
            cluster_map: Dictionary mapping cluster_id -> list of row indices
            weights: Optional column weights
            
        Returns:
            Dictionary with pattern information:
            - pattern_info: Dict mapping pattern_str -> {
                'pattern': list, 'count': int, 'weighted_coverage_percent': float
              }
        """
        pattern_info = {}
        
        if len(selected_columns) == 0 or len(cluster_map) == 0:
            return {'pattern_info': pattern_info}
        
        # Create submatrix with selected columns
        submatrix = matrix[:, selected_columns]
        
        # Calculate total weight if weights are available
        total_weight = 0.0
        if weights is not None:
            weights_array = np.asarray(weights)
            total_weight = float(weights_array.sum()) if weights_array.size > 0 else 0.0
        
        # For each cluster (pattern), calculate coverage
        for cluster_id, row_indices in cluster_map.items():
            if len(row_indices) == 0:
                continue
            
            # Get the pattern for this cluster (all rows in cluster have same pattern)
            pattern = submatrix[row_indices[0]].tolist()
            pattern_str = str(pattern)  # Convert to string for dictionary key
            
            # Count structures in this pattern
            count = len(row_indices)
            
            # Calculate weighted coverage for this pattern
            weighted_coverage_percent = 0.0
            if weights is not None and total_weight > 0:
                try:
                    selected_indices = np.asarray(selected_columns)
                    selected_weights = weights_array[selected_indices]
                    selected_weight_sum = float(selected_weights.sum())
                    weighted_coverage_percent = 100.0 * selected_weight_sum / total_weight
                except (IndexError, ValueError):
                    weighted_coverage_percent = 0.0
            
            pattern_info[pattern_str] = {
                'pattern': pattern,
                'count': count,
                'weighted_coverage_percent': weighted_coverage_percent
            }
        
        return {'pattern_info': pattern_info}
    
    def run_partition_solver(self, variant, matrix: np.ndarray, tau: int, weights: Optional[np.ndarray] = None,
                             return_solution_matrix: bool = False) -> Union[Dict, Tuple[Dict, np.ndarray]]:
        """
        Run a partition solver (see algorithms/registry.py) on matrix with given tau.

        Args:
            variant: registry Variant whose `solve(matrix, tau, weights)` returns a
                     (tau x M) 0/1 array, row t-1 marking the columns kept with <= t groups
            matrix: Binary matrix
            tau: Maximum number of partitions
            weights: Optional column weights for weighted coverage calculation
            return_solution_matrix: If True, return tuple of (result_dict, solution_matrix)

        Returns:
            Dictionary with results, or tuple of (result_dict, solution_matrix) if
            return_solution_matrix=True and the run succeeded
        """
        if self.debug:
            print(f"  Running {variant.label} algorithm with tau={tau}")

        start_time = time.time()

        # Initialize memory monitoring
        memory_monitor = None
        if self.track_memory:
            memory_monitor = MemoryMonitor()
            memory_monitor.start_monitoring()

        try:
            solution = variant.solve(matrix, tau, weights)
            runtime = time.time() - start_time

            # Check if we exceeded time limit - if so, record time_limit as runtime
            if self.time_limit and runtime > self.time_limit:
                runtime = float(self.time_limit)  # Record the time limit as the runtime

            # Stop memory monitoring and get peak memory
            peak_memory_mb = 0
            if memory_monitor:
                memory_monitor.stop_monitoring()
                peak_memory_mb = memory_monitor.get_peak_memory_mb()

            # solution is tau x M; row tau-1 is the best with at most tau groups
            selected_columns = []
            cluster_map = {}
            if solution.shape[0] >= tau:
                selected_columns, cluster_map = self._clusters_for_tau(solution, matrix, tau)

            # Calculate objective value (number of selected columns)
            objective_value = len(selected_columns)

            # Calculate pattern coverage for each unique pattern
            pattern_coverage_result = self._calculate_pattern_coverage(matrix, selected_columns, cluster_map, weights)
            pattern_info = pattern_coverage_result.get('pattern_info', {})

            # Check if we exceeded time limit - record time_limit as runtime and mark status accordingly
            if self.time_limit and runtime >= self.time_limit:
                runtime = float(self.time_limit)  # Record the time limit as the runtime
                status = 'TIME_LIMIT'
                # Ensure memory is captured even if track_memory was False
                if not memory_monitor and peak_memory_mb == 0:
                    try:
                        process = psutil.Process()
                        peak_memory_mb = process.memory_info().rss / 1024 / 1024
                    except Exception:
                        pass
            else:
                status = 'SUCCESS'

            result = {
                'algorithm': variant.family,
                'runtime': runtime,
                'status': status,
                'selected_columns': selected_columns,
                'num_clusters': len(cluster_map),
                'objective_value': objective_value,
                'cluster_map': cluster_map,
                'pattern_info': pattern_info,
                **variant.extra,
                'solution_matrix_shape': solution.shape
            }

            # Add memory usage if tracked (always include for timeouts)
            if self.track_memory or status == 'TIME_LIMIT':
                result['peak_memory_mb'] = peak_memory_mb

            # Return solution matrix if requested
            if return_solution_matrix:
                return result, solution
            # Clean up memory after algorithm completion
            del solution
            gc.collect()
            return result

        except Exception as e:
            runtime = time.time() - start_time

            # Check if this is a timeout error - if so, record time_limit as runtime
            is_timeout = 'time limit' in str(e).lower() or 'timeout' in str(e).lower()
            if is_timeout and self.time_limit and runtime > self.time_limit:
                runtime = float(self.time_limit)  # Record the time limit as the runtime
                status = 'TIME_LIMIT'
            else:
                status = 'ERROR'

            # Stop memory monitoring and get peak memory (always record, even on timeout)
            peak_memory_mb = 0
            if memory_monitor:
                memory_monitor.stop_monitoring()
                peak_memory_mb = memory_monitor.get_peak_memory_mb()
            elif self.track_memory or is_timeout:
                # Still try to get memory even if monitor wasn't initialized (always for timeouts)
                try:
                    process = psutil.Process()
                    peak_memory_mb = process.memory_info().rss / 1024 / 1024
                except Exception:
                    pass

            import traceback
            error_details = f"Exception: {str(e)}\nTraceback: {traceback.format_exc()}"
            result = {
                'algorithm': variant.family,
                'runtime': runtime,
                'status': status,
                'error_message': error_details,
                'selected_columns': [],
                'num_clusters': 0,
                'objective_value': 0,
                'cluster_map': {},
                **variant.extra,
            }

            # Add memory usage if tracked (always include for timeouts)
            if self.track_memory or is_timeout:
                result['peak_memory_mb'] = peak_memory_mb

            return result

    def _write_result_to_csv(self, result: Dict, csv_writer_ref: List = None, csv_file = None, header_written_ref: List = None) -> None:
        """
        Write a single result row to CSV file incrementally.

        The header is the first row's columns. A later row with a column the
        header lacks (an error or warning on a failed run, or another file's
        metadata) widens the header: the file is rewritten with every row so
        far, so no row and no column is ever dropped.

        Args:
            result: Result dictionary to write
            csv_writer_ref: List containing CSV DictWriter object (using list for mutable reference)
            csv_file: CSV file object (for flushing)
            header_written_ref: List containing boolean flag for whether header has been written (using list for mutable reference)
        """
        if csv_writer_ref is None or csv_file is None:
            return

        try:
            rewrite = False
            if header_written_ref is not None and not header_written_ref[0]:
                self._csv_fieldnames = list(result.keys())
                self._csv_rows = []
                header_written_ref[0] = True
                rewrite = True
            new_columns = [k for k in result if k not in self._csv_fieldnames]
            if new_columns:
                self._csv_fieldnames.extend(new_columns)
                rewrite = True
            self._csv_rows.append(result)

            if rewrite:
                csv_file.seek(0)
                csv_file.truncate()
                csv_writer = csv.DictWriter(csv_file, fieldnames=self._csv_fieldnames, restval='')
                csv_writer.writeheader()
                csv_writer.writerows(self._csv_rows)
                csv_writer_ref[0] = csv_writer
            else:
                csv_writer_ref[0].writerow(result)
            csv_file.flush()  # Ensure data is written immediately
        except Exception as e:
            print(f"⚠️  Warning: Error writing result to CSV: {e}")

    def _run_variant(self, variant, matrix: np.ndarray, tau: int, weights: Optional[np.ndarray],
                     return_solution_matrix: bool = False):
        """Run one registry variant: its own runner, or the shared partition-solver runner."""
        if variant.run is not None:
            return variant.run(self, matrix, tau, weights)
        return self.run_partition_solver(variant, matrix, tau, weights, return_solution_matrix=return_solution_matrix)

    def _is_skip_failure(self, spec, result: Dict, tau: int) -> bool:
        """Whether this failed run means the algorithm's remaining runs are skipped."""
        if not result or result.get('status') not in ['ERROR', 'FAILED']:
            return False
        error_msg = result.get('error_message', '').lower()
        if spec.skip_policy == 'memory':
            if any(keyword in error_msg for keyword in algorithm_registry.MEMORY_KEYWORDS):
                if self.debug:
                    print(f"  🚨 {spec.summary_name} algorithm memory failure detected at tau={tau}")
                    print(f"      Error: {result.get('error_message', 'Unknown error')}")
                    print(f"      Will skip {spec.summary_name} algorithm for remaining tau values")
                return True
            return False
        # timeout policy (ILP): a timeout keyword, or a runtime more than 5% over the limit
        runtime = result.get('runtime', 0)
        time_limit_exceeded = bool(self.time_limit and runtime > self.time_limit * 1.05)
        timed_out = time_limit_exceeded or any(keyword in error_msg for keyword in algorithm_registry.TIMEOUT_KEYWORDS)
        if timed_out and self.debug:
            print(f"  🚨 {spec.summary_name} algorithm timeout failure detected at tau={tau}")
            print(f"      Error: {result.get('error_message', 'Unknown error')}")
            print(f"      Runtime: {runtime:.1f}s (limit: {self.time_limit}s)")
            if self.ilp_continue_on_timeout:
                print(f"      Continuing {spec.summary_name} for remaining tau values (per flag)")
            else:
                print(f"      Will skip {spec.summary_name} algorithm for remaining tau values")
        if self.debug and any(keyword in error_msg for keyword in algorithm_registry.ILP_MEMORY_KEYWORDS):
            print(f"  🚨 {spec.summary_name} algorithm memory failure detected at tau={tau}")
            print(f"      Error: {result.get('error_message', 'Unknown error')}")
            print(f"      Runtime: {runtime:.1f}s")
            print(f"      Note: Continuing with remaining tau values")
        return timed_out and not self.ilp_continue_on_timeout

    @staticmethod
    def _skipped_result(spec, variant) -> Dict:
        """Result recorded for a run skipped after an earlier failure."""
        if spec.skip_policy == 'timeout':
            status, message = 'SKIPPED_TIMEOUT', 'Skipped due to timeout failure in previous tau'
        else:
            status, message = 'SKIPPED_MEMORY', 'Skipped due to memory failure in previous tau'
        return {
            'algorithm': variant.family,
            'runtime': 0.0,
            'status': status,
            'error_message': message,
            'selected_columns': [],
            'num_clusters': 0,
            'objective_value': 0,
            'cluster_map': {},
            **variant.extra,
        }

    @staticmethod
    def _convert_cluster_map_to_ids(cluster_map: Dict, row_mapping: Dict, id_to_row_mapping: Dict) -> Dict:
        """
        Convert cluster_map from row indices to structure IDs, including duplicates.

        Args:
            cluster_map: Dictionary mapping cluster_id -> list of aggregated row indices
            row_mapping: Dictionary mapping aggregated row index -> list of original row indices
            id_to_row_mapping: Dictionary mapping structure ID -> original row index

        Returns:
            Dictionary mapping cluster_id -> list of structure IDs (including duplicates)
        """
        if not cluster_map:
            return {}

        # Create reverse mapping: original row index -> structure ID
        row_to_id = {v: k for k, v in id_to_row_mapping.items()}

        cluster_map_with_ids = {}
        for cluster_id, aggregated_row_indices in cluster_map.items():
            structure_ids = []
            for agg_row_idx in aggregated_row_indices:
                # Expand aggregated row to original row indices
                if agg_row_idx in row_mapping:
                    original_row_indices = row_mapping[agg_row_idx]
                else:
                    # If not in row_mapping, assume it's already an original row index
                    original_row_indices = [agg_row_idx]

                # Convert original row indices to structure IDs
                for orig_row_idx in original_row_indices:
                    if orig_row_idx in row_to_id:
                        structure_ids.append(row_to_id[orig_row_idx])
                    else:
                        # Fallback: use row index as string if ID not found
                        structure_ids.append(str(orig_row_idx))

            cluster_map_with_ids[cluster_id] = sorted(set(structure_ids))  # Remove duplicates and sort

        return cluster_map_with_ids

    @staticmethod
    def _convert_selected_columns_to_units(selected_columns: List[int], column_mapping: Optional[Dict] = None, unique_units: Optional[List] = None) -> List:
        """
        Convert selected column indices to actual structure units.

        Args:
            selected_columns: List of column indices
            column_mapping: Dictionary mapping column index -> structure unit (if available)
            unique_units: List of unique structure units (if available)

        Returns:
            List of structure units (or column indices if mapping not available)
        """
        if not selected_columns:
            return []

        if column_mapping:
            # Use column_mapping if available
            units = [column_mapping.get(col_idx, f"col_{col_idx}") for col_idx in selected_columns]
        elif unique_units:
            # Use unique_units list if available
            units = [unique_units[col_idx] if col_idx < len(unique_units) else f"col_{col_idx}" for col_idx in selected_columns]
        else:
            # Fallback: return column indices as strings
            units = [f"col_{col_idx}" for col_idx in selected_columns]

        return units

    def _create_algorithm_result(self, algorithm_name: str, result: Dict, metadata: Dict,
                                 tau: int, matrix: np.ndarray, weights: Optional[np.ndarray]) -> Dict:
        """Create a standardized result row for a single algorithm."""
        if not result:
            return None

        # Helper to safely parse dict from string or return dict
        def safe_parse_dict(value):
            if value is None:
                return {}
            if isinstance(value, dict):
                return value
            if isinstance(value, str):
                try:
                    import ast
                    return ast.literal_eval(value)
                except:
                    return {}
            return {}

        # Get mappings for conversion
        row_mapping = {}
        if 'row_mapping' in metadata:
            row_mapping = safe_parse_dict(metadata['row_mapping'])
        elif '_row_mapping' in metadata:
            row_mapping = safe_parse_dict(metadata['_row_mapping'])

        id_to_row_mapping = {}
        if 'id_to_row_mapping' in metadata:
            id_to_row_mapping = safe_parse_dict(metadata['id_to_row_mapping'])
        elif '_id_to_row_mapping' in metadata:
            id_to_row_mapping = safe_parse_dict(metadata['_id_to_row_mapping'])

        # Get column mapping for selected columns
        column_mapping = None
        unique_units = None
        if 'column_mapping' in metadata:
            col_map_val = metadata['column_mapping']
            if isinstance(col_map_val, dict):
                column_mapping = col_map_val
            elif isinstance(col_map_val, list):
                unique_units = col_map_val
        elif '_column_mapping' in metadata:
            col_map_val = metadata['_column_mapping']
            if isinstance(col_map_val, dict):
                column_mapping = col_map_val
            elif isinstance(col_map_val, list):
                unique_units = col_map_val
        elif 'unique_units' in metadata:
            unique_units = metadata['unique_units'] if isinstance(metadata['unique_units'], list) else safe_parse_dict(metadata['unique_units'])
        elif 'unique_base_pairs' in metadata:
            unique_units = metadata['unique_base_pairs'] if isinstance(metadata['unique_base_pairs'], list) else safe_parse_dict(metadata['unique_base_pairs'])

        # Convert cluster_map to use IDs
        cluster_map_with_ids = {}
        if result.get('cluster_map'):
            cluster_map_with_ids = self._convert_cluster_map_to_ids(
                result.get('cluster_map', {}),
                row_mapping or {},
                id_to_row_mapping or {}
            )

        # Convert selected_columns to structure units
        selected_columns = result.get('selected_columns', [])
        selected_units = self._convert_selected_columns_to_units(
            selected_columns,
            column_mapping,
            unique_units
        )

        algo_result = {
            **{k: v for k, v in metadata.items() if k not in ['gapped_mappings', 'column_mapping', '_column_mapping', 'unique_units', 'unique_base_pairs']},
            'tau': tau,
            'algorithm': algorithm_name,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'runtime': result.get('runtime', 0.0),
            'status': result.get('status', 'UNKNOWN'),
            'selected_k': len(selected_columns),
            'num_clusters': result.get('num_clusters', 0),
            'objective_value': result.get('objective_value', 0),
            'selected_units': str(selected_units),  # List of selected structure units
            'cluster_map': str(cluster_map_with_ids),  # Now uses IDs
        }

        # Add selected_columns_list (column indices) to detailed columns
        if self.detailed_output:
            algo_result['selected_columns_list'] = str(selected_columns)
            algo_result['pattern_info'] = str(result.get('pattern_info', {}))

        # Add memory usage if tracked
        if self.track_memory and 'peak_memory_mb' in result:
            algo_result['peak_memory_mb'] = result['peak_memory_mb']

        # Add error message if status indicates failure
        if result.get('status') in ['ERROR', 'FAILED', 'SKIPPED_MEMORY', 'SKIPPED_TIMEOUT']:
            algo_result['error'] = result.get('error_message', result.get('status', 'Unknown error'))

        # Add warning if partial success
        if result.get('status') == 'PARTIAL_SUCCESS':
            gurobi_status = result.get('gurobi_status', 'UNKNOWN')
            algo_result['warning'] = f"Non-optimal solution (status: {gurobi_status})"

        # Note: algorithm_variant removed - only ILP, MSTP, MSTP-BEAM allowed
        # beam_value is now part of algorithm name (MSTP-BEAM-{beam_value})
        # Removed algorithm-specific optional columns: solution_matrix_shape, unweighted_objective,
        # structure_coverage, coverage_ratio, gurobi_status, is_optimal, has_incumbent, active_clusters, prototypes

        # Add detailed metadata only if detailed output is enabled
        if self.detailed_output:
            if 'original_matrix_shape' in result:
                algo_result['original_matrix_shape'] = str(result.get('original_matrix_shape', ()))
            elif 'original_matrix_shape' in metadata:
                algo_result['original_matrix_shape'] = str(metadata.get('original_matrix_shape', ()))
            elif '_original_matrix_shape' in metadata:
                algo_result['original_matrix_shape'] = str(metadata.get('_original_matrix_shape', ()))

            if 'pre_aggregation_shape' in result:
                algo_result['pre_aggregation_shape'] = str(result.get('pre_aggregation_shape', ()))
            elif 'pre_aggregation_shape' in metadata:
                algo_result['pre_aggregation_shape'] = str(metadata.get('pre_aggregation_shape', ()))
            elif '_pre_aggregation_shape' in metadata:
                algo_result['pre_aggregation_shape'] = str(metadata.get('_pre_aggregation_shape', ()))

            if 'row_mapping' in result:
                algo_result['row_mapping'] = str(result.get('row_mapping', {}))
            elif 'row_mapping' in metadata:
                algo_result['row_mapping'] = str(metadata.get('row_mapping', {}))
            elif '_row_mapping' in metadata:
                algo_result['row_mapping'] = str(metadata.get('_row_mapping', {}))

            if 'duplicates_removed' in result:
                algo_result['duplicates_removed'] = result.get('duplicates_removed', 0)
            elif 'duplicates_removed' in metadata:
                algo_result['duplicates_removed'] = metadata.get('duplicates_removed', 0)
            elif '_duplicates_removed' in metadata:
                algo_result['duplicates_removed'] = metadata.get('_duplicates_removed', 0)

        # Calculate feature coverage
        total_features = matrix.shape[1]
        if result.get('status') in ['SUCCESS', 'PARTIAL_SUCCESS']:
            selected_features = len(selected_columns)
            feature_coverage = (selected_features / total_features * 100) if total_features > 0 else 0
            algo_result['feature_coverage_percent'] = feature_coverage

            # Calculate weighted coverage if weights are available
            if weights is not None:
                weights_array = np.asarray(weights)
                total_weight = float(weights_array.sum()) if weights_array.size > 0 else 0.0
                if selected_features > 0 and total_weight > 0:
                    try:
                        selected_indices = np.asarray(selected_columns)
                        selected_weights = weights_array[selected_indices]
                        selected_weight_sum = float(selected_weights.sum())
                        algo_result['feature_coverage_percent_weighted'] = 100.0 * selected_weight_sum / total_weight
                    except (IndexError, ValueError):
                        algo_result['feature_coverage_percent_weighted'] = 0.0
                else:
                    algo_result['feature_coverage_percent_weighted'] = 0.0
            else:
                algo_result['feature_coverage_percent_weighted'] = np.nan
        else:
            algo_result['feature_coverage_percent'] = 0.0
            algo_result['feature_coverage_percent_weighted'] = 0.0 if weights is not None else np.nan

        # Add total features
        algo_result['total_features'] = total_features

        return algo_result


    def analyze_fasta_file(self, fasta_file: str, tau_values: List[int], max_structures: Optional[int] = None, algorithm: str = 'mstp', csv_writer_ref: List = None, csv_file = None, header_written_ref: List = None) -> List[Dict]:
        """
        Analyze a single structure file with multiple tau values.
        Supports multiple input formats: FASTA, text, JSON, CSV.
        
        Args:
            fasta_file: Path to input file (FASTA, text, JSON, or CSV format)
            tau_values: List of tau values to test
            max_structures: Maximum number of structures to process
            
        Returns:
            List of result dictionaries
        """
        file_results = []
        
        try:
            # Load the matrix: a plain 0/1 table, or features of RNA structures
            # (structure units or base pairs)
            if self.input_kind == 'matrix':
                matrix, weights, metadata = load_matrix(
                    fasta_file, self.weights_file, max_structures, self.detailed_output, self.pre_aggregation)
            elif self.use_structure_units:
                if self.use_weighted_ilp:
                    matrix, weights, unique_units, valid_structures, id_to_row_mapping, original_matrix_shape, pre_aggregation_shape, row_mapping, duplicates_removed, gapped_mappings = universal_load_structure_units_matrix_with_weights_from_fasta(
                        fasta_file, max_structures, use_pre_aggregation=self.pre_aggregation)
                    metadata = {
                        'unique_units': unique_units,
                        'valid_structures': valid_structures,
                        'id_to_row_mapping': id_to_row_mapping,
                        'original_matrix_shape': original_matrix_shape,
                        'pre_aggregation_shape': pre_aggregation_shape,
                        'row_mapping': row_mapping,
                        'duplicates_removed': duplicates_removed,
                        'gapped_mappings': gapped_mappings
                    }
                else:
                    matrix, metadata = self.load_structure_units_matrix_from_fasta(fasta_file, max_structures)
                    weights = None
            else:
                matrix, metadata = self.load_matrix_from_fasta(fasta_file, max_structures)
                # For base pairs mode, use uniform weights (1.0 per base pair)
                # This allows weighted coverage to be calculated (equals unweighted coverage)
                weights = np.ones(matrix.shape[1], dtype=float) if matrix.shape[1] > 0 else None
            
            if matrix.shape[1] == 0 and self.input_kind == 'matrix':
                raise ValueError(f"{fasta_file} has no columns")
            if matrix.shape[1] == 0:
                raise ValueError(
                    f"No structural features found in {fasta_file} "
                    f"({len(metadata.get('valid_structures', [])) or matrix.shape[0]} structures parsed; "
                    "all are unpaired, empty, or failed to parse)")

            # Validate tau values
            max_possible_tau = matrix.shape[0]
            
            if self.skip_invalid_tau:
                valid_tau_values = [tau for tau in tau_values if tau <= max_possible_tau]
                
                if len(valid_tau_values) != len(tau_values):
                    invalid_tau = [tau for tau in tau_values if tau > max_possible_tau]
                    print(f"⚠️  Warning: Skipping tau values {invalid_tau} (exceeds matrix rows {max_possible_tau})")
            else:
                valid_tau_values = tau_values
                # Check for invalid tau values but don't skip them
                invalid_tau = [tau for tau in tau_values if tau > max_possible_tau]
                if invalid_tau:
                    print(f"⚠️  Warning: Tau values {invalid_tau} exceed matrix rows {max_possible_tau}, but will be attempted anyway")
            
            # Run the selected algorithm for each tau value (see algorithms/registry.py)
            spec = algorithm_registry.get(algorithm)
            variants = spec.variants(self)
            # Set by a failure under spec.skip_policy; the remaining runs of this
            # algorithm are then recorded as skipped.
            skip_remaining = False

            # An exact algorithm runs once at the largest tau and every smaller tau
            # is read off that run (the answers are the same either way). If that
            # run fails, each tau runs on its own instead, so every row reports
            # its own outcome: smaller taus may well fit where the largest did not.
            reused_runs = {}  # variant label -> (max-tau result, solution matrix)
            if (self.mstp_extract_from_max_tau and
                spec.reuse_max_tau and
                len(valid_tau_values) > 1):
                max_tau = max(valid_tau_values)
                for variant in variants:
                    if self.debug:
                        print(f"\nRunning {variant.label} algorithm once for max tau={max_tau} (will extract solutions for all tau values)")
                    outcome = self._run_variant(variant, matrix, max_tau, weights, return_solution_matrix=True)
                    if isinstance(outcome, tuple):
                        reused_runs[variant.label] = outcome
                    else:
                        # A failed run returns only its error result.
                        print(f"  ⚠️  {variant.label} failed at max tau={max_tau} ({outcome.get('status')}); "
                              f"running each tau separately")
                        if self.debug:
                            print(f"      Error: {outcome.get('error_message', 'Unknown error')}")

            for tau in valid_tau_values:
                if self.debug:
                    print(f"\nAnalyzing tau={tau}")
                else:
                    print(f"tau={tau}:", end=" ")

                run_results = []  # (variant, result)
                for variant in variants:
                    if skip_remaining:
                        reason = 'timeout' if spec.skip_policy == 'timeout' else 'memory'
                        print(f"  ⚠️  Skipping {variant.label} algorithm (tau={tau}) due to previous {reason} failure")
                        result = self._skipped_result(spec, variant)
                    elif variant.label in reused_runs:
                        max_tau_result, solution_matrix = reused_runs[variant.label]
                        if self.debug:
                            print(f"  Extracting {variant.label} solution for tau={tau} from pre-computed solution matrix")
                        extracted_result = self._extract_partition_solution_for_tau(
                            solution_matrix, matrix, tau, weights)
                        # Use runtime and memory from the max tau run
                        result = {
                            **extracted_result,
                            'runtime': max_tau_result['runtime'],
                            'status': max_tau_result['status'],
                            'solution_matrix_shape': solution_matrix.shape
                        }
                        if 'peak_memory_mb' in max_tau_result:
                            result['peak_memory_mb'] = max_tau_result['peak_memory_mb']
                        if max_tau_result['status'] in ['ERROR', 'FAILED']:
                            result['error_message'] = max_tau_result.get('error_message', 'Unknown error')
                    else:
                        result = self._run_variant(variant, matrix, tau, weights)
                        if self._is_skip_failure(spec, result, tau):
                            skip_remaining = True
                    run_results.append((variant, result))

                # Create one result row per variant
                algorithm_results = []
                for variant, result in run_results:
                    row = self._create_algorithm_result(variant.label, result, metadata, tau, matrix, weights)
                    if row:
                        algorithm_results.append(row)

                # Add sequences and structures columns (if available)
                # Parse the input file to get sequences and structures
                try:
                    if self.input_kind == 'matrix':
                        raise ValueError("plain matrix input has no sequences or structures")
                    from algorithms.parse_structures import parse_structure_file
                    parsed_data = parse_structure_file(fasta_file)
                    
                    # Create dictionaries mapping structure ID to sequence/structure
                    sequences_dict = {}
                    structures_dict = {}
                    
                    for structure_id, sequence, structure in parsed_data:
                        sequences_dict[structure_id] = sequence if sequence else ''
                        structures_dict[structure_id] = structure if structure else ''
                    
                    # Add to all algorithm results
                    for algo_row in algorithm_results:
                        algo_row['sequences'] = str(sequences_dict)
                        algo_row['structures'] = str(structures_dict)
                except Exception as e:
                    # If parsing fails, skip sequences/structures columns
                    if self.debug:
                        print(f"  Warning: Could not parse sequences/structures from input file: {e}")
                
                # Add column and row mappings to all results if detailed output is enabled
                if self.detailed_output:
                    for algo_row in algorithm_results:
                        # Add column_mapping (unique_units) to detailed columns
                        if 'column_mapping' in metadata:
                            algo_row['column_mapping'] = str(metadata['column_mapping'])
                        elif '_column_mapping' in metadata:
                            algo_row['column_mapping'] = str(metadata['_column_mapping'])
                        elif 'unique_units' in metadata:
                            algo_row['column_mapping'] = str(metadata['unique_units'])
                        elif 'unique_base_pairs' in metadata:
                            algo_row['column_mapping'] = str(metadata['unique_base_pairs'])
                        
                        # Add id_to_row_mapping to detailed columns
                        if 'id_to_row_mapping' in metadata:
                            algo_row['id_to_row_mapping'] = str(metadata['id_to_row_mapping'])
                        elif '_id_to_row_mapping' in metadata:
                            algo_row['id_to_row_mapping'] = str(metadata['_id_to_row_mapping'])
                        
                        # Add include_stacking and include_stem_runs to detailed columns (these are feature extraction parameters)
                        if 'include_stacking' in metadata:
                            algo_row['include_stacking'] = metadata['include_stacking']
                        if 'include_stem_runs' in metadata:
                            algo_row['include_stem_runs'] = metadata['include_stem_runs']
                
                # Write each algorithm result as a separate row
                for algo_result in algorithm_results:
                    file_results.append(algo_result)
                    # Write result to CSV incrementally
                    self._write_result_to_csv(algo_result, csv_writer_ref, csv_file, header_written_ref)

                # Always print runtime information for each algorithm
                for variant, result in run_results:
                    status_icon = {'SUCCESS': "✅", 'PARTIAL_SUCCESS': "⚠️"}.get(
                        result['status'], "⏭️" if result['status'].startswith('SKIPPED') else "❌")
                    memory_info = ""
                    if self.track_memory and 'peak_memory_mb' in result:
                        memory_info = f", {result['peak_memory_mb']:.1f}MB peak"
                    status_text = result['status']
                    if result['status'] == 'PARTIAL_SUCCESS':
                        status_text = f"{result['status']} ({result.get('gurobi_status', 'UNKNOWN')})"
                    print(f"    {status_icon} {variant.label} algorithm (tau={tau}): {status_text}, {result['runtime']:.3f}s{memory_info}")

            # Clean up memory between tau values
            reused_runs.clear()
            gc.collect()

            # Simplified runtime summary
            print(f"\nSummary:")
            labels = {variant.label for variant in variants}
            rows = [r for r in file_results if r.get('algorithm') in labels]
            successful = len([r for r in rows if r.get('status') in spec.success_statuses])
            total_time = sum(r.get('runtime', 0) for r in rows)
            print(f"  {spec.summary_name}: {successful}/{len(rows)} successful, {total_time:.2f}s total")

            if skip_remaining:
                skipped = [r for r in rows if str(r.get('status', '')).startswith('SKIPPED')]
                reason = 'timeout' if spec.skip_policy == 'timeout' else 'memory'
                print(f"\n⚠️  {spec.summary_name} algorithm {reason} failure summary:")
                print(f"    Skipped {len(skipped)} {spec.summary_name} algorithm runs due to {reason} constraints")
                if spec.skip_policy == 'timeout':
                    print(f"    Consider increasing time limit or reducing tau range for {spec.summary_name} algorithm")

            memory_failures = [r for r in rows if r.get('status') == 'ERROR' and
                               any(keyword in r.get('error', '').lower() for keyword in algorithm_registry.ILP_MEMORY_KEYWORDS)]
            if memory_failures:
                print(f"\n⚠️  {spec.summary_name} algorithm memory failure summary:")
                print(f"    {len(memory_failures)} runs failed due to memory constraints")
                print(f"    Failed tau values: {[r['tau'] for r in memory_failures]}")

        except Exception as e:
            print(f"❌ Error analyzing {fasta_file}: {e}")
            # stdout is redirected to the log file; make file-level failures visible too
            console = getattr(self, 'original_stdout', None)
            if console is not None and console is not sys.stdout:
                console.write(f"❌ Error analyzing {fasta_file}: {e}\n")
                console.flush()
            import traceback
            error_details = f"File analysis failed: {str(e)}\nTraceback: {traceback.format_exc()}"
            # Add error result
            error_result = {
                'input_file': fasta_file,
                'num_sequences': 0,
                'num_structures': 0,
                'matrix_shape': '(0, 0)',
                'sparsity': 0.0,
                'tau': 0,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'file_analysis_error': error_details
            }
            file_results.append(error_result)
            
            # Write error result to CSV incrementally
            self._write_result_to_csv(error_result, csv_writer_ref, csv_file, header_written_ref)
        
        # Clean up large objects at the end of file analysis
        if 'matrix' in locals():
            del matrix
        if 'weights' in locals():
            del weights
        if 'metadata' in locals():
            del metadata
        gc.collect()
        
        return file_results
    
    def analyze_fasta_directory(self, fasta_dir: str, tau_values: List[int], max_structures: Optional[int] = None, algorithm: str = 'mstp', csv_writer_ref: List = None, csv_file = None, header_written_ref: List = None) -> List[Dict]:
        """
        Analyze all structure files in a directory.
        Supports multiple input formats: FASTA, text, JSON, CSV.
        
        Args:
            fasta_dir: Path to directory containing structure files
            tau_values: List of tau values to test
            max_structures: Maximum number of structures to process per file
            
        Returns:
            List of result dictionaries
        """
        all_results = []
        
        # Find all structure files (FASTA, text, JSON, CSV)
        file_patterns = ['*.fasta', '*.fa', '*.fna', '*.ffn', '*.faa', '*.frn', '*.txt', '*.json', '*.csv']
        structure_files = []
        
        for pattern in file_patterns:
            structure_files.extend(glob.glob(os.path.join(fasta_dir, pattern)))
            structure_files.extend(glob.glob(os.path.join(fasta_dir, '**', pattern), recursive=True))
        
        structure_files = list(set(structure_files))  # Remove duplicates
        structure_files.sort()
        
        if not structure_files:
            print(f"Error: No structure files found in {fasta_dir}")
            return all_results
        
        print(f"Found {len(structure_files)} files")
        
        # Analyze each file
        for i, structure_file in enumerate(structure_files, 1):
            print(f"\n[{i}/{len(structure_files)}] {os.path.basename(structure_file)}")
            
            file_results = self.analyze_fasta_file(structure_file, tau_values, max_structures, algorithm, csv_writer_ref, csv_file, header_written_ref)
            all_results.extend(file_results)
        
        return all_results


def main():
    """Main function to run the analysis based on command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run ILP, MSTP, and MSTP-BEAM algorithms on structure files with tau values/ranges. Supports FASTA, text, JSON, and CSV input formats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single structure file (FASTA format), single tau
  python src/MASS.py --input input.fasta --tau 3 --output results.csv
  
  # Single structure file (text format: >id\nstructure), tau range
  python src/MASS.py --input input.txt --tau-range 2 5 --output results.csv
  
  # Single structure file (JSON format: {id: structure}), single tau
  python src/MASS.py --input input.json --tau 3 --output results.csv
  
  # Single structure file (CSV format: id,structure), tau range
  python src/MASS.py --input input.csv --tau-range 2 5 --output results.csv
  
  # Directory of structure files, single tau
  python src/MASS.py --input-dir /path/to/structure/files --tau 3 --output results.csv
  
  # Directory of structure files, tau range
  python src/MASS.py --input-dir /path/to/structure/files --tau-range 2 8 --output results.csv
  
  # With debug output, time limit, and memory tracking
  python src/MASS.py --input input.fasta --tau-range 2 6 --output results.csv --debug --time-limit 300 --track-memory
  
  # Use default timeout (7200 seconds = 120 minutes)
  python src/MASS.py --input input.fasta --tau-range 2 6 --output results.csv
  
  # Keep identical rows instead of merging them (same results; for debugging)
  python src/MASS.py --input input.fasta --tau-range 2 6 --output results.csv --no-pre-aggregation
  
  # Weighted ILP is enabled by default; disable if needed
  python src/MASS.py --input input.fasta --tau-range 2 6 --output results.csv --no-weighted-ilp
  
  # Run MSTP-BEAM algorithm
  python src/MASS.py --input input.fasta --tau-range 2 6 --output results.csv --algorithm mstp_beam --beam-value 10
  
  # Allow invalid tau values (tau values exceeding unique structures will be attempted)
  python src/MASS.py --input input.fasta --tau-range 2 20 --output results.csv --allow-invalid-tau
  
  # Continue ILP after timeouts
  python src/MASS.py --input input.fasta --tau-range 2 6 --output results.csv --ilp-continue-on-timeout
  
  # Disable MSTP extract from max tau (run separately for each tau)
  python src/MASS.py --input input.fasta --tau-range 2 6 --output results.csv --no-mstp-extract-from-max-tau
        """
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str, help='Path to single structure file (supports FASTA, text, JSON, CSV formats)')
    input_group.add_argument('--input-dir', type=str, help='Path to directory containing structure files (supports FASTA, text, JSON, CSV formats)')
    input_group.add_argument('--matrix', type=str,
                             help='Path to a plain 0/1 matrix, rows = objects, columns = features (.npy, .csv, .tsv, .txt, .tab). '
                                  'A non-numeric first row is read as column names, a non-numeric first column as row names.')
    parser.add_argument('--weights', type=str,
                        help='With --matrix: column weights, one number per column (comma, tab, space or newline separated). Default: 1 each.')
    
    # Tau options
    tau_group = parser.add_mutually_exclusive_group(required=True)
    tau_group.add_argument('--tau', type=int, help='Single tau value to test')
    tau_group.add_argument('--tau-range', nargs=2, type=int, metavar=('MIN', 'MAX'), 
                          help='Range of tau values (inclusive)')
    
    # Algorithm options
    parser.add_argument('--algorithm', type=str, choices=algorithm_registry.names(),
                       default='mstp', help='Algorithm to run (default: mstp). ' + '; '.join(
                           f"{name}: {spec.help}" for name, spec in algorithm_registry.ALGORITHMS.items()))
    
    # Configuration options
    parser.add_argument('--time-limit', type=int, default=7200, help='Time limit for ILP solver (seconds, default: 7200)')
    parser.add_argument('--max-structures', type=int, help='Maximum number of structures to process per FASTA file')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--detailed', action='store_true', help='Include detailed columns in output (column mappings, gapped sequences, pre-aggregation metadata, etc.)')
    parser.add_argument('--track-memory', action='store_true', default=False, help='Enable memory tracking (default: memory tracking is disabled)')
    parser.add_argument('--use-base-pairs', action='store_true', 
                       help='Use base pairs matrix instead of structure units matrix (default: structure units)')
    
    # ILP algorithm options
    parser.add_argument('--no-pre-aggregation', '--mstp-no-pre-aggregation', '--ilp-no-pre-aggregation',
                       dest='no_pre_aggregation', action='store_true',
                       help='Do not merge identical rows before running (results are the same; only the problem size changes). '
                            '--mstp-no-pre-aggregation and --ilp-no-pre-aggregation are older names for this flag.')
    parser.add_argument('--ilp-continue-on-timeout', action='store_true',
                       help='Do not skip subsequent tau values when ILP hits time limit')
    
    # Weighted ILP options
    parser.add_argument('--no-weighted-ilp', dest='use_weighted_ilp', action='store_false', default=True,
                       help='Disable weighted ILP optimization (default: weighted ILP is enabled)')
    
    # MSTP-BEAM options
    parser.add_argument('--beam-value', type=int, nargs='+', default=[5],
                       help='Beam width(s) for MSTP-BEAM algorithm (default: 5). Only used when --algorithm mstp_beam. Can specify multiple values, each will create a separate row with algorithm name MSTP-BEAM-{beam_value})')
    
    # Tau validation options
    parser.add_argument('--allow-invalid-tau', action='store_true',
                       help='Allow tau values greater than the number of unique structures (default: skip invalid tau values)')
    
    # MSTP algorithm optimization options
    parser.add_argument('--no-mstp-extract-from-max-tau', dest='mstp_extract_from_max_tau', action='store_false', default=True,
                       help='Disable running MSTP algorithm once for max tau and extracting solutions for all tau values (default: enabled, only applies to tau ranges; MSTP-BEAM always runs each tau separately because beam results depend on tau)')
    
    # Output options
    parser.add_argument('--output', '-o', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--log', type=str, default=None, help='Log file path (default: {output_basename}.log in same directory as output file)')
    
    args = parser.parse_args()
    if args.weights and not args.matrix:
        parser.error('--weights applies only to --matrix input')
    
    # Set up logging
    log_file = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    if args.log is not None:
        log_file = args.log
    else:
        # Default: use output filename with .log extension
        output_path = Path(args.output)
        log_file = str(output_path.parent / f"{output_path.stem}.log")
    
    # Open log file and redirect stdout/stderr
    log_fp = open(log_file, 'w', encoding='utf-8')
    sys.stdout = log_fp
    sys.stderr = log_fp
    
    # Print to console (before redirect)
    original_stdout.write(f"Logging to: {log_file}\n")
    original_stdout.write(f"Running analysis...\n")
    original_stdout.flush()
    
    # Determine tau values
    if args.tau:
        tau_values = [args.tau]
    else:
        min_tau, max_tau = args.tau_range
        tau_values = list(range(min_tau, max_tau + 1))
    
    print(f"Tau values to test: {tau_values}")
    
    # Simplified algorithm information
    print(f"Algorithm: {args.algorithm.upper()}")
    algorithm_detail = algorithm_registry.get(args.algorithm).describe(args)
    if algorithm_detail:
        print(algorithm_detail)
    
    # Initialize analyzer
    analyzer = FastaTauAnalyzer(
        debug=args.debug, 
        time_limit=args.time_limit, 
        memory_monitoring=False,  # Removed flag, always False
        pre_aggregation=not args.no_pre_aggregation,
        use_structure_units=not args.use_base_pairs,
        use_weighted_ilp=args.use_weighted_ilp,
        track_memory=args.track_memory,
        beam_value=args.beam_value[0] if isinstance(args.beam_value, list) and len(args.beam_value) > 0 else (args.beam_value if isinstance(args.beam_value, int) else 5),
        beam_values=args.beam_value if isinstance(args.beam_value, list) else [args.beam_value] if isinstance(args.beam_value, int) else [5],
        skip_invalid_tau=not args.allow_invalid_tau,
        ilp_continue_on_timeout=args.ilp_continue_on_timeout,
        mstp_extract_from_max_tau=args.mstp_extract_from_max_tau,
        detailed_output=args.detailed
    )
    
    # Store original_stdout for use in other functions
    analyzer.original_stdout = original_stdout
    
    csv_file = None
    results = []
    try:
        # Open CSV file for incremental writing
        csv_file = open(args.output, 'w', newline='')
        csv_writer_ref = [None]  # Use list to allow mutable reference
        header_written_ref = [False]  # Track if header has been written
        
        # Run analysis with incremental CSV writing
        if args.matrix:
            print(f"Analyzing matrix: {os.path.basename(args.matrix)}")
            analyzer.input_kind = 'matrix'
            analyzer.weights_file = args.weights
            results = analyzer.analyze_fasta_file(args.matrix, tau_values, args.max_structures, args.algorithm, csv_writer_ref, csv_file, header_written_ref)
        elif args.input:
            print(f"Analyzing: {os.path.basename(args.input)}")
            results = analyzer.analyze_fasta_file(args.input, tau_values, args.max_structures, args.algorithm, csv_writer_ref, csv_file, header_written_ref)
        else:
            print(f"Analyzing directory: {args.input_dir}")
            results = analyzer.analyze_fasta_directory(args.input_dir, tau_values, args.max_structures, args.algorithm, csv_writer_ref, csv_file, header_written_ref)
        
    finally:
        # Close CSV file
        if csv_file:
            csv_file.close()
    
    # Print summary statistics (similar to save_results_to_csv but without saving again)
    try:
        if results:
            # Print to stdout
            original_stdout.write(f"\n✅ Results saved incrementally to: {args.output}\n")
            original_stdout.flush()
            # Also print to log
            print(f"\n✅ Results saved incrementally to: {args.output}")
            print(f"   Total experiments: {len(results)}")
            
            # Show summary statistics using pandas
            summary_lines = []  # Collect summary lines to print to stdout
            try:
                df = pd.DataFrame(results)
                
                if 'tau' in df.columns:
                    tau_line = f"   Tau range: {df['tau'].min()} - {df['tau'].max()}"
                    print(tau_line)
                    summary_lines.append(tau_line)
                    unique_tau_line = f"   Unique tau values: {df['tau'].nunique()}"
                    print(unique_tau_line)
                    summary_lines.append(unique_tau_line)
                
                if 'input_file' in df.columns:
                    files_line = f"   Input files analyzed: {df['input_file'].nunique()}"
                    print(files_line)
                    summary_lines.append(files_line)
                
                if 'total_features' in df.columns:
                    avg_features = df['total_features'].mean()
                    features_line = f"   Average total features per file: {avg_features:.1f}"
                    print(features_line)
                    summary_lines.append(features_line)
                
                # Show statistics by algorithm (using new column structure)
                if 'algorithm' in df.columns:
                    for algo_name in df['algorithm'].dropna().unique():
                        algo_df = df[df['algorithm'] == algo_name]
                        success_count = len(algo_df[algo_df['status'] == 'SUCCESS'])
                        partial_success_count = len(algo_df[algo_df['status'] == 'PARTIAL_SUCCESS'])
                        total_success_count = success_count + partial_success_count
                        attempted_count = len(algo_df[algo_df['status'].isin(['SUCCESS', 'PARTIAL_SUCCESS', 'ERROR', 'FAILED'])])

                        if attempted_count > 0:
                            algo_line = f"   {algo_name} algorithm success rate: {total_success_count}/{attempted_count} ({total_success_count/attempted_count*100:.1f}%)"
                            if partial_success_count:
                                algo_line += f" - {success_count} optimal, {partial_success_count} non-optimal"
                            print(algo_line)
                            summary_lines.append(algo_line)

                            # Show average feature coverage for successful runs
                            successful_algo = algo_df[algo_df['status'].isin(['SUCCESS', 'PARTIAL_SUCCESS'])]
                            if len(successful_algo) > 0 and 'feature_coverage_percent' in successful_algo.columns:
                                avg_coverage = successful_algo['feature_coverage_percent'].mean()
                                coverage_line = f"   {algo_name} average feature coverage: {avg_coverage:.1f}%"
                                print(coverage_line)
                                summary_lines.append(coverage_line)
                        else:
                            algo_line = f"   {algo_name} algorithm success rate: N/A (no attempts)"
                            print(algo_line)
                            summary_lines.append(algo_line)
            except Exception as e:
                error_msg = f"⚠️  Warning: Error generating summary statistics: {e}"
                print(error_msg)
                summary_lines.append(error_msg)
            
            # Print summary to stdout
            if summary_lines:
                original_stdout.write("\n".join(summary_lines) + "\n")
                original_stdout.flush()
            
            # Print completion message to stdout
            original_stdout.write(f"\n✅ Analysis completed successfully!\n")
            original_stdout.flush()
            # Also print to log
            print(f"\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        # Restore stdout/stderr before exiting
        if log_file:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_fp.close()
        sys.exit(1)
    finally:
        # Always restore stdout/stderr and close log file
        if log_file:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            log_fp.close()
            original_stdout.write(f"✓ Log saved to: {log_file}\n")


if __name__ == "__main__":
    main()
