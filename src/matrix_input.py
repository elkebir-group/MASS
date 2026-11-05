"""
Plain matrix input (`--matrix`): any 0/1 objects x features table, so MASS
runs on data that is not RNA structures (mutation matrices, SNP panels,
presence/absence tables, ...).

Formats, chosen by extension:
- `.npy`: a 2-D array; rows and columns are named by index.
- `.csv`: comma-separated; `.tsv` / `.txt` / `.tab`: tab-separated.
  A first row with any non-numeric cell after the first column is a header
  of column names. A first column with any non-numeric cell is row names.

Column weights (`--weights`) are one number per column, separated by commas,
tabs, spaces or newlines, in column order. Without them every column weighs 1.

Identical rows are aggregated, as the RNA loaders do (unless
`--no-pre-aggregation`); `row_mapping` records which original rows each
aggregated row stands for.
"""
import csv
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from algorithms.parse_structures import _pre_aggregate_rows

_DELIMITERS = {'.csv': ',', '.tsv': '\t', '.txt': '\t', '.tab': '\t'}


def _is_number(cell: str) -> bool:
    try:
        float(cell)
        return True
    except ValueError:
        return False


def _read_table(path: str) -> Tuple[np.ndarray, Optional[List[str]], Optional[List[str]]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        values = np.load(path, allow_pickle=False)
        if values.ndim != 2:
            raise ValueError(f"{path}: expected a 2-D array, got shape {values.shape}")
        return values, None, None
    if ext not in _DELIMITERS:
        raise ValueError(f"{path}: unsupported matrix format {ext!r} (use .npy, .csv, .tsv, .txt or .tab)")
    with open(path, newline='', encoding='utf-8') as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f, delimiter=_DELIMITERS[ext]) if any(c.strip() for c in r)]
    if not rows:
        raise ValueError(f"{path}: empty matrix file")

    has_row_names = any(not _is_number(r[0]) for r in rows[1:]) if len(rows) > 1 else not _is_number(rows[0][0])
    first = 1 if has_row_names else 0
    has_header = any(not _is_number(c) for c in rows[0][first:])
    column_names = rows[0][first:] if has_header else None
    body = rows[1:] if has_header else rows
    row_names = [r[0] for r in body] if has_row_names else None

    width = len(body[0]) - first if body else 0
    for i, r in enumerate(body):
        if len(r) - first != width:
            raise ValueError(f"{path}: row {i + 1} has {len(r) - first} values, expected {width}")
    values = np.array([[float(c) for c in r[first:]] for r in body], dtype=float).reshape(len(body), width)
    return values, column_names, row_names


def read_weights(path: str, n_columns: int) -> np.ndarray:
    with open(path, encoding='utf-8') as f:
        tokens = [t for t in re.split(r'[,\s]+', f.read()) if t]
    try:
        weights = np.array([float(t) for t in tokens], dtype=float)
    except ValueError as e:
        raise ValueError(f"{path}: weights must be numbers ({e})") from None
    if weights.shape[0] != n_columns:
        raise ValueError(f"{path}: {weights.shape[0]} weights for {n_columns} columns")
    if (weights < 0).any():
        raise ValueError(f"{path}: weights must be non-negative")
    return weights


def load_matrix(path: str, weights_path: Optional[str] = None, max_rows: Optional[int] = None,
                detailed_output: bool = False, pre_aggregation: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Return (matrix, weights, metadata) in the shape the RNA loaders give MASS."""
    values, column_names, row_names = _read_table(path)
    if max_rows is not None:
        values = values[:max_rows]
        row_names = row_names[:max_rows] if row_names else None
    bad = ~np.isin(values, (0, 1))
    if bad.any():
        r, c = map(int, np.argwhere(bad)[0])
        raise ValueError(f"{path}: values must be 0 or 1; row {r + 1}, column {c + 1} is {values[r, c]:g}")
    full = values.astype(int)
    n_rows, n_cols = full.shape

    column_names = column_names or [f"col_{j}" for j in range(n_cols)]
    row_names = row_names or [str(i) for i in range(n_rows)]
    if len(set(row_names)) != n_rows:
        raise ValueError(f"{path}: row names must be unique")
    weights = read_weights(weights_path, n_cols) if weights_path else np.ones(n_cols, dtype=float)

    if pre_aggregation and n_rows:
        matrix, row_mapping = _pre_aggregate_rows(full)
    else:
        matrix, row_mapping = full, {i: [i] for i in range(n_rows)}
    matrix = matrix.reshape(-1, n_cols)
    id_to_row_mapping = {name: i for i, name in enumerate(row_names)}

    metadata = {
        'input_file': path,
        'num_sequences': 0,
        'num_structures': n_rows,
        'matrix_shape': matrix.shape,
        'sparsity': 1 - np.sum(matrix) / matrix.size if matrix.size > 0 else 0,
    }
    details = {
        'column_mapping': column_names,
        'id_to_row_mapping': id_to_row_mapping,
        'original_matrix_shape': full.shape,
        'pre_aggregation_shape': full.shape,
        'row_mapping': row_mapping,
        'duplicates_removed': n_rows - matrix.shape[0],
    }
    if detailed_output:
        metadata.update(details)
    else:
        metadata.update({f"_{k}": v for k, v in details.items()})
    return matrix, weights, metadata
