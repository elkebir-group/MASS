import numpy as np
from collections import defaultdict
import re
import json
import csv
import os
from typing import List, Tuple, Dict, Set, Optional


def calculate_structure_unit_weights(structure_units: List, rna_length: int) -> np.ndarray:
    """
    Calculate weights for structure units based on nucleotide counts.
    
    Args:
        structure_units: List of structure units in format ({base pairs}, {unpaired nucleotides})
        or legacy formats (for backward compatibility)
        rna_length: Length of the RNA sequence (not used anymore, kept for compatibility)
        
    Returns:
        Array of weights for each structure unit (integer nucleotide counts)
        
    Weight calculation rules:
    - Weight = 2 * num_base_pairs + num_unpaired_nucleotides
    - Each base pair contributes 2 nucleotides
    - Each unpaired nucleotide contributes 1 nucleotide
    """
    weights = []
    
    for unit in structure_units:
        # Handle string representations of tuples
        if isinstance(unit, str):
            try:
                # Try to evaluate the string as a tuple
                import ast
                unit = ast.literal_eval(unit)
            except:
                try:
                    unit = eval(unit)
                except:
                    # If evaluation fails, treat as unknown format
                    weight = 1
                    print(f"Warning: Could not parse structure unit string: {unit}")
                    weights.append(weight)
                    continue
        
        if isinstance(unit, tuple):
            # Check for new format: ({base pairs}, {unpaired nucleotides})
            if len(unit) == 2 and isinstance(unit[0], set) and isinstance(unit[1], set):
                # New format: ({base pairs}, {unpaired nucleotides})
                num_pairs = len(unit[0])
                num_unpaired = len(unit[1])
                weight = 2 * num_pairs + num_unpaired
            # Legacy format handling for backward compatibility
            elif len(unit) == 2 and isinstance(unit[0], tuple) and isinstance(unit[1], tuple):
                # Stacking pair: ((i, j), (i+1, j-1))
                # Weight = 4 nucleotides (2 pairs × 2 nucleotides each)
                weight = 4
            elif len(unit) == 2 and isinstance(unit[0], tuple) and isinstance(unit[1], (list, tuple)):
                # Hairpin: ((i, j), [list of nucleotide indices])
                # Weight = enclosing pair (2 nucleotides) + unpaired nucleotides
                loop_length = len(unit[1])
                weight = 2 + loop_length
            elif len(unit) == 4 and isinstance(unit[0], tuple) and isinstance(unit[1], tuple):
                # Interior/bulge: ((i, j), (l, k), [left-side nucleotides], [right-side nucleotides])
                # Weight = outer pair (2 nucleotides) + inner pair (2 nucleotides) + left_nucleotides + right_nucleotides
                left_nucleotides = len(unit[2])
                right_nucleotides = len(unit[3])
                total_nucleotides = 2 + 2 + left_nucleotides + right_nucleotides
                weight = total_nucleotides
            elif len(unit) == 3 and unit[1] == 'multiloop':
                # Multiloop: (pairs, 'multiloop', unpaired_indices)
                # Weight = 2 * num_pairs + num_unpaired_nucleotides
                pairs = unit[0]
                unpaired_indices = unit[2]
                num_pairs = len(pairs)
                num_unpaired = len(unpaired_indices)
                total_nucleotides = 2 * num_pairs + num_unpaired
                weight = total_nucleotides
            elif len(unit) == 2 and unit[1] == 'stem_run':
                # Stem run: (pairs, 'stem_run')
                # Weight = 2 * num_pairs
                pairs = unit[0]
                num_pairs = len(pairs)
                weight = 2 * num_pairs
            else:
                # Unknown tuple format, default weight
                weight = 1
                print(f"Warning: Unknown structure unit format: {unit} (len={len(unit)})")
        else:
            # Non-tuple unit, default weight
            weight = 1
            print(f"Warning: Non-tuple structure unit: {unit}")
        
        weights.append(weight)
    
    return np.array(weights)

def parseBracketString_pk(s):
    """
    Parse a dot-bracket structure string and return list of base pairs.
    Returns None if the structure is invalid.
    """
    stack_paren = []  # Stack for parentheses '()'
    stack_square = []  # Stack for square brackets '[]'
    stack_curl = []  # Stack for curly brackets '{}'
    structure = []

    for i, c in enumerate(s):
        if c == '(':
            stack_paren.append(i)
        elif c == '[':
            stack_square.append(i)
        elif c == '{':
            stack_curl.append(i)
        elif c == ')':
            if not stack_paren:
                return None
            idx_open = stack_paren.pop(-1)
            structure.append((idx_open, i))
        elif c == ']':
            if not stack_square:
                return None
            idx_open = stack_square.pop(-1)
            structure.append((idx_open, i))
        elif c == '}':
            if not stack_curl:
                return None
            idx_open = stack_curl.pop(-1)
            structure.append((idx_open, i))
    
    # Check for unmatched opening brackets
    if stack_paren or stack_square or stack_curl:
        return None
    
    return sorted(structure)

def create_nucleotide_mapping(gapped_sequence):
    """
    Create a mapping from gapped positions to ungapped positions.
    
    Args:
        gapped_sequence: Sequence with gaps (e.g., "G-U-AC")
        
    Returns:
        Tuple of (gapped_to_ungapped, ungapped_to_gapped, ungapped_sequence)
        - gapped_to_ungapped: dict mapping gapped position (1-based) -> ungapped position (1-based)
        - ungapped_to_gapped: dict mapping ungapped position (1-based) -> gapped position (1-based)
        - ungapped_sequence: sequence without gaps (e.g., "GUAC")
    """
    gapped_to_ungapped = {}
    ungapped_to_gapped = {}
    ungapped_sequence = ""
    
    ungapped_pos = 1  # Start from 1 for 1-based indexing
    for gapped_pos, char in enumerate(gapped_sequence, 1):  # Start from 1 for 1-based indexing
        if char != '-':  # Not a gap
            gapped_to_ungapped[gapped_pos] = ungapped_pos
            ungapped_to_gapped[ungapped_pos] = gapped_pos
            ungapped_sequence += char
            ungapped_pos += 1
    
    return gapped_to_ungapped, ungapped_to_gapped, ungapped_sequence

def map_structure_units_to_gapped(structure_units, ungapped_to_gapped):
    """
    Map structure units from ungapped positions to gapped positions.
    
    Args:
        structure_units: List of structure units in ungapped coordinates
        ungapped_to_gapped: Dictionary mapping ungapped -> gapped positions
        
    Returns:
        List of structure units in gapped coordinates
    """
    gapped_units = []
    
    for unit in structure_units:
        if isinstance(unit, tuple):
            # Check for new format: ({base pairs}, {unpaired nucleotides})
            if len(unit) == 2 and isinstance(unit[0], set) and isinstance(unit[1], set):
                # New format: ({base pairs}, {unpaired nucleotides})
                gapped_pairs = set()
                for pair in unit[0]:
                    if isinstance(pair, tuple) and len(pair) >= 2:
                        gapped_pair = (ungapped_to_gapped.get(pair[0], pair[0]), 
                                      ungapped_to_gapped.get(pair[1], pair[1]))
                        gapped_pairs.add(gapped_pair)
                # Filter out positions that don't exist in gapped coordinates
                gapped_unpaired = {ungapped_to_gapped[pos] for pos in unit[1] if pos in ungapped_to_gapped}
                gapped_units.append((gapped_pairs, gapped_unpaired))
            # Legacy format handling for backward compatibility
            elif len(unit) == 2 and isinstance(unit[0], tuple) and isinstance(unit[1], tuple):
                # Stacking pair: ((i, j), (i+1, j-1))
                outer_pair = (ungapped_to_gapped.get(unit[0][0], unit[0][0]), 
                             ungapped_to_gapped.get(unit[0][1], unit[0][1]))
                inner_pair = (ungapped_to_gapped.get(unit[1][0], unit[1][0]), 
                             ungapped_to_gapped.get(unit[1][1], unit[1][1]))
                gapped_units.append((outer_pair, inner_pair))
                
            elif len(unit) == 2 and isinstance(unit[0], tuple) and isinstance(unit[1], list):
                # Hairpin: ((i, j), [unpaired_indices])
                outer_pair = (ungapped_to_gapped.get(unit[0][0], unit[0][0]), 
                             ungapped_to_gapped.get(unit[0][1], unit[0][1]))
                # Filter out positions that don't exist in gapped coordinates
                unpaired_indices = [ungapped_to_gapped[pos] for pos in unit[1] if pos in ungapped_to_gapped]
                gapped_units.append((outer_pair, unpaired_indices))
                
            elif len(unit) == 4 and isinstance(unit[0], tuple) and isinstance(unit[1], tuple):
                # Interior/bulge: ((i, j), (l, k), [left_nucleotides], [right_nucleotides])
                outer_pair = (ungapped_to_gapped.get(unit[0][0], unit[0][0]), 
                             ungapped_to_gapped.get(unit[0][1], unit[0][1]))
                inner_pair = (ungapped_to_gapped.get(unit[1][0], unit[1][0]), 
                             ungapped_to_gapped.get(unit[1][1], unit[1][1]))
                # Filter out positions that don't exist in gapped coordinates
                left_nucleotides = [ungapped_to_gapped[pos] for pos in unit[2] if pos in ungapped_to_gapped]
                right_nucleotides = [ungapped_to_gapped[pos] for pos in unit[3] if pos in ungapped_to_gapped]
                gapped_units.append((outer_pair, inner_pair, left_nucleotides, right_nucleotides))
                
            elif len(unit) == 3 and unit[1] == 'multiloop':
                # Multiloop: (pairs, 'multiloop', unpaired_indices)
                gapped_pairs = []
                for pair in unit[0]:
                    if isinstance(pair, tuple) and len(pair) >= 2:
                        gapped_pair = (ungapped_to_gapped.get(pair[0], pair[0]), 
                                     ungapped_to_gapped.get(pair[1], pair[1]))
                        gapped_pairs.append(gapped_pair)
                # Filter out positions that don't exist in gapped coordinates
                unpaired_indices = [ungapped_to_gapped[pos] for pos in unit[2] if pos in ungapped_to_gapped]
                gapped_units.append((gapped_pairs, 'multiloop', unpaired_indices))
                
            elif len(unit) == 2 and unit[1] == 'stem_run':
                # Stem run: (pairs, 'stem_run')
                gapped_pairs = []
                for pair in unit[0]:
                    if isinstance(pair, tuple) and len(pair) >= 2:
                        gapped_pair = (ungapped_to_gapped.get(pair[0], pair[0]), 
                                     ungapped_to_gapped.get(pair[1], pair[1]))
                        gapped_pairs.append(gapped_pair)
                gapped_units.append((gapped_pairs, 'stem_run'))
            else:
                # Unknown format - keep as is
                gapped_units.append(unit)
        else:
            # Non-tuple unit - keep as is
            gapped_units.append(unit)
    
    return gapped_units

def map_structure_units_to_ungapped(structure_units, gapped_to_ungapped):
    """
    Map structure units from gapped positions to ungapped positions.
    
    Args:
        structure_units: List of structure units in gapped coordinates
        gapped_to_ungapped: Dictionary mapping gapped -> ungapped positions
        
    Returns:
        List of structure units in ungapped coordinates
    """
    ungapped_units = []
    
    for unit in structure_units:
        if isinstance(unit, tuple):
            # Check for new format: ({base pairs}, {unpaired nucleotides})
            if len(unit) == 2 and isinstance(unit[0], set) and isinstance(unit[1], set):
                # New format: ({base pairs}, {unpaired nucleotides})
                ungapped_pairs = set()
                for pair in unit[0]:
                    if isinstance(pair, tuple) and len(pair) >= 2:
                        ungapped_pair = (gapped_to_ungapped.get(pair[0], pair[0]), 
                                       gapped_to_ungapped.get(pair[1], pair[1]))
                        ungapped_pairs.add(ungapped_pair)
                # Filter out gap positions and map to ungapped
                ungapped_unpaired = {gapped_to_ungapped[pos] for pos in unit[1] if pos in gapped_to_ungapped}
                ungapped_units.append((ungapped_pairs, ungapped_unpaired))
            # Legacy format handling for backward compatibility
            elif len(unit) == 2 and isinstance(unit[0], tuple) and isinstance(unit[1], tuple):
                # Stacking pair: ((i, j), (i+1, j-1))
                outer_pair = (gapped_to_ungapped.get(unit[0][0], unit[0][0]), 
                             gapped_to_ungapped.get(unit[0][1], unit[0][1]))
                inner_pair = (gapped_to_ungapped.get(unit[1][0], unit[1][0]), 
                             gapped_to_ungapped.get(unit[1][1], unit[1][1]))
                ungapped_units.append((outer_pair, inner_pair))
                
            elif len(unit) == 2 and isinstance(unit[0], tuple) and isinstance(unit[1], list):
                # Hairpin: ((i, j), [unpaired_indices])
                outer_pair = (gapped_to_ungapped.get(unit[0][0], unit[0][0]), 
                             gapped_to_ungapped.get(unit[0][1], unit[0][1]))
                # Filter out gap positions and map to ungapped
                unpaired_indices = [gapped_to_ungapped[pos] for pos in unit[1] if pos in gapped_to_ungapped]
                ungapped_units.append((outer_pair, unpaired_indices))
                
            elif len(unit) == 4 and isinstance(unit[0], tuple) and isinstance(unit[1], tuple):
                # Interior/bulge: ((i, j), (l, k), [left_nucleotides], [right_nucleotides])
                outer_pair = (gapped_to_ungapped.get(unit[0][0], unit[0][0]), 
                             gapped_to_ungapped.get(unit[0][1], unit[0][1]))
                inner_pair = (gapped_to_ungapped.get(unit[1][0], unit[1][0]), 
                             gapped_to_ungapped.get(unit[1][1], unit[1][1]))
                # Filter out gap positions and map to ungapped
                left_nucleotides = [gapped_to_ungapped[pos] for pos in unit[2] if pos in gapped_to_ungapped]
                right_nucleotides = [gapped_to_ungapped[pos] for pos in unit[3] if pos in gapped_to_ungapped]
                ungapped_units.append((outer_pair, inner_pair, left_nucleotides, right_nucleotides))
                
            elif len(unit) == 3 and unit[1] == 'multiloop':
                # Multiloop: (pairs, 'multiloop', unpaired_indices)
                ungapped_pairs = []
                for pair in unit[0]:
                    if isinstance(pair, tuple) and len(pair) >= 2:
                        ungapped_pair = (gapped_to_ungapped.get(pair[0], pair[0]), 
                                       gapped_to_ungapped.get(pair[1], pair[1]))
                        ungapped_pairs.append(ungapped_pair)
                # Filter out gap positions and map to ungapped
                ungapped_unpaired = [gapped_to_ungapped[pos] for pos in unit[2] if pos in gapped_to_ungapped]
                ungapped_units.append((ungapped_pairs, 'multiloop', ungapped_unpaired))
            else:
                # Unknown format, keep as is
                ungapped_units.append(unit)
        else:
            # Non-tuple unit, keep as is
            ungapped_units.append(unit)
    
    return ungapped_units

def parse_gapped_structure(gapped_sequence, gapped_structure):
    """
    Parse a gapped sequence and structure, returning both gapped and ungapped structure units.
    
    Args:
        gapped_sequence: Sequence with gaps (e.g., "G-U-AC")
        gapped_structure: Structure with gaps (e.g., "(....)")
        
    Returns:
        Tuple of (gapped_units, ungapped_units, gapped_to_ungapped, ungapped_to_gapped, ungapped_sequence)
    """
    # Create nucleotide mapping
    gapped_to_ungapped, ungapped_to_gapped, ungapped_sequence = create_nucleotide_mapping(gapped_sequence)
    
    # Extract structure units from ungapped structure (after removing gaps)
    # This ensures valid structure units without gaps breaking relationships
    ungapped_structure_str = ''.join(c for c in gapped_structure if c != '-')
    ungapped_units = rna_units_from_dotbracket(ungapped_structure_str, include_stem_runs=False, sequence=ungapped_sequence)
    
    # Map ungapped units back to gapped coordinates
    gapped_units = map_structure_units_to_gapped(ungapped_units, ungapped_to_gapped)
    
    return gapped_units, ungapped_units, gapped_to_ungapped, ungapped_to_gapped, ungapped_sequence

def parse_structure_file(filename: str) -> List[Tuple[str, Optional[str], str]]:
    """
    Universal parser for structure files. Supports multiple formats:
    1. FASTA format: >id\nseq\nstructure
    2. Text format: >id\nstructure (no sequence)
    3. JSON format: {id: structure}
    4. CSV format: id,structure columns
    
    Args:
        filename: Path to input file
        
    Returns:
        List of tuples: (sequence_id, sequence, structure_string)
        where sequence can be None if not provided
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    
    # Detect file format based on extension and content
    file_ext = os.path.splitext(filename)[1].lower()
    
    # Try JSON first (most specific)
    if file_ext == '.json':
        return _parse_json_file(filename)
    
    # Try CSV
    if file_ext == '.csv':
        return _parse_csv_file(filename)
    
    # For .txt, .fasta, .fa, or no extension, try to detect format
    # Read first few lines to detect format
    with open(filename, 'r') as f:
        first_line = f.readline().strip()
        f.seek(0)  # Reset to beginning
        
        # Check if it's JSON
        if first_line.startswith('{'):
            try:
                return _parse_json_file(filename)
            except:
                pass  # Not valid JSON, continue
        
        # Check if it's CSV (has comma and no '>')
        if ',' in first_line and not first_line.startswith('>'):
            try:
                return _parse_csv_file(filename)
            except:
                pass  # Not valid CSV, continue
        
        # Default to FASTA/text format
        return _parse_fasta_or_text_file(filename)


def _parse_json_file(filename: str) -> List[Tuple[str, Optional[str], str]]:
    """Parse JSON file with format {id: structure}"""
    structures = []
    with open(filename, 'r') as f:
        data = json.load(f)
        for seq_id, structure_str in data.items():
            # Extract structure string (handle if it's a dict with structure field)
            if isinstance(structure_str, dict):
                structure_str = structure_str.get('structure', '')
            elif not isinstance(structure_str, str):
                structure_str = str(structure_str)
            structures.append((seq_id, None, structure_str))
    return structures


def _parse_csv_file(filename: str) -> List[Tuple[str, Optional[str], str]]:
    """Parse CSV file with id,structure columns"""
    structures = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        # Check if columns exist
        if 'id' not in reader.fieldnames or 'structure' not in reader.fieldnames:
            # Try case-insensitive matching
            fieldnames_lower = {f.lower(): f for f in reader.fieldnames}
            id_col = fieldnames_lower.get('id', None)
            structure_col = fieldnames_lower.get('structure', None)
            
            if id_col is None or structure_col is None:
                raise ValueError(f"CSV file must have 'id' and 'structure' columns. Found: {reader.fieldnames}")
        else:
            id_col = 'id'
            structure_col = 'structure'
        
        for row in reader:
            seq_id = row[id_col].strip()
            structure_str = row[structure_col].strip()
            if seq_id and structure_str:
                structures.append((seq_id, None, structure_str))
    return structures


def _parse_fasta_or_text_file(filename: str) -> List[Tuple[str, Optional[str], str]]:
    """
    Parse FASTA or text file. Handles both:
    - FASTA: >id\nseq\nstructure
    - Text: >id\nstructure (no sequence)
    """
    structures = []
    current_id = None
    current_seq = None
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            if line.startswith('>'):
                # When we see a new ID, check if we need to save the previous entry
                # (This handles the case where the previous entry was structure-only and already saved)
                # If current_id exists but current_seq is None, it means we never saw a structure line
                # after the ID, which is an incomplete entry - skip it
                if current_id and current_seq is None:
                    # Previous entry was incomplete (ID but no structure) - skip it
                    pass
                
                # Start new entry
                current_id = line[1:].strip()  # Remove '>' prefix
                current_seq = None
            elif current_id is None:
                # Skip lines before first ID
                continue
            elif current_seq is None:
                # This could be either sequence or structure
                # Check if it looks like a structure (contains brackets)
                if any(c in line for c in '().[]{}'):
                    # It's a structure (no sequence provided)
                    structure_part = line.split()[0]  # Take only the bracket part
                    structures.append((current_id, None, structure_part))
                    current_id = None  # Reset for next entry
                else:
                    # It's a sequence line
                    parts = line.split()
                    current_seq = parts[0]  # Take only the sequence part
            else:
                # We have sequence, this must be structure
                structure_part = line.split()[0]  # Take only the bracket part
                structures.append((current_id, current_seq, structure_part))
                current_id = None
                current_seq = None
        
        # Handle last entry if file ends without a new '>'
        # If we have an ID and sequence but no structure yet, the entry is incomplete - skip it
        # If we have an ID but no sequence, we should have already saved it when we saw the structure
        # (unless the structure line was never encountered, in which case it's incomplete)
        if current_id and current_seq:
            # We have ID and sequence but no structure - incomplete entry, skip it
            pass
    
    return structures


def parse_fasta_file(filename):
    """
    Parse the FASTA file containing RNA sequences and their dot-bracket structures.
    Handles both formats:
    1. One sequence per entry with one structure
    2. One sequence with multiple structure predictions
    
    Returns a list of tuples: (sequence_id, sequence, structure_string)
    """
    structures = []
    current_id = None
    current_seq = None
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Start new entry
                current_id = line[1:]  # Remove '>' prefix
                current_seq = None
            elif current_seq is None:
                # This is the sequence line
                # Handle format: "sequence -25.60 3.00" or just "sequence"
                parts = line.split()
                current_seq = parts[0]  # Take only the sequence part
            else:
                # This is a structure line
                # Structure line format: "(((...))) (-31.90)" or just "(((...)))"
                structure_part = line.split()[0]  # Take only the bracket part
                
                # Add the structure and reset for next entry
                structures.append((current_id, current_seq, structure_part))
                current_seq = None  # Reset to prepare for next sequence
    
    return structures

def parse_fasta_file_robust(filename):
    """
    Robust version of parse_fasta_file that handles empty lines and malformed entries.
    
    Returns a list of tuples: (sequence_id, sequence, structure_string)
    """
    structures = []
    current_id = None
    current_seq = None
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            if line.startswith('>'):
                # Start new entry
                current_id = line[1:]  # Remove '>' prefix
                current_seq = None
            elif current_seq is None:
                # This is the sequence line
                # Handle format: "sequence -25.60 3.00" or just "sequence"
                parts = line.split()
                if parts:  # Check if line has content
                    current_seq = parts[0]  # Take only the sequence part
            else:
                # This is a structure line
                # Structure line format: "(((...))) (-31.90)" or just "(((...)))"
                parts = line.split()
                if parts:  # Check if line has content
                    structure_part = parts[0]  # Take only the bracket part
                    
                    # Add the structure and reset for next entry
                    structures.append((current_id, current_seq, structure_part))
                    current_seq = None  # Reset to prepare for next sequence
    
    return structures

def parse_fasta_file_multiple_structures(filename):
    """
    Parse FASTA file with one sequence and multiple structure predictions.
    Format:
    >ID
    sequence
    structure1
    structure2
    structure3
    ...
    
    Returns a list of tuples: (sequence_id, sequence, structure_string)
    """
    structures = []
    current_id = None
    current_seq = None
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Start new entry
                current_id = line[1:]  # Remove '>' prefix
                current_seq = None
            elif current_seq is None:
                # This is the sequence line
                # Handle format: "sequence -25.60 3.00" or just "sequence"
                parts = line.split()
                current_seq = parts[0]  # Take only the sequence part
            else:
                # This is a structure line
                # Structure line format: "(((...))) (-24.10)" or just "(((...)))"
                structure_part = line.split()[0]  # Take only the bracket part
                
                # Add the structure (keep the same sequence for all structures)
                structures.append((current_id, current_seq, structure_part))
    
    return structures

def extract_base_pairs(structures_data):
    """
    Extract all base pairs from the structures and create a mapping.
    Returns:
    - all_base_pairs: set of all unique base pairs
    - structure_base_pairs: list of base pairs for each structure
    """
    all_base_pairs = set()
    structure_base_pairs = []
    valid_structures = []
    
    for seq_id, sequence, structure_str in structures_data:
        base_pairs = parseBracketString_pk(structure_str)
        if base_pairs is not None:
            # Convert to 1-indexed base pairs (as requested)
            indexed_pairs = [(i+1, j+1) for i, j in base_pairs]
            structure_base_pairs.append(indexed_pairs)
            all_base_pairs.update(indexed_pairs)
            valid_structures.append((seq_id, sequence, structure_str))
        else:
            print(f"Warning: Invalid structure for {seq_id}")
    
    return all_base_pairs, structure_base_pairs, valid_structures

def _pre_aggregate_rows(matrix: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Pre-aggregate identical rows to reduce problem size.
    
    Args:
        matrix: Original binary matrix
        
    Returns:
        Tuple of (aggregated_matrix, row_mapping)
    """
    # Find unique rows and their multiplicities
    unique_rows = []
    row_mapping = {}  # Maps aggregated row index to original row indices
    
    seen_rows = {}
    for i, row in enumerate(matrix):
        row_tuple = tuple(row)
        if row_tuple in seen_rows:
            # This row is a duplicate
            agg_row_idx = seen_rows[row_tuple]
            row_mapping[agg_row_idx].append(i)
        else:
            # This is a new unique row
            agg_row_idx = len(unique_rows)
            unique_rows.append(row)
            seen_rows[row_tuple] = agg_row_idx
            row_mapping[agg_row_idx] = [i]
    
    aggregated_matrix = np.array(unique_rows, dtype=int)
    return aggregated_matrix, row_mapping

def create_structure_matrix(all_base_pairs, structure_base_pairs, valid_structures=None, use_pre_aggregation=True):
    """
    Create a binary matrix where:
    - Rows represent RNA structures
    - Columns represent unique base pairs
    - Cell value is 1 if the structure contains that base pair, 0 otherwise
    
    Args:
        all_base_pairs: Set of all unique base pairs
        structure_base_pairs: List of base pairs for each structure
        valid_structures: List of (seq_id, sequence, structure_str) tuples
        use_pre_aggregation: Whether to pre-aggregate identical rows
    
    Returns:
        - matrix: Binary matrix (aggregated if use_pre_aggregation=True)
        - unique_pairs: List of unique base pairs (column mapping)
        - id_to_row_mapping: Dictionary mapping sequence ID to row index
        - original_matrix_shape: Original matrix shape before aggregation
        - pre_aggregation_shape: Matrix shape after aggregation
        - row_mapping: Dictionary mapping aggregated row index to original row indices
        - duplicates_removed: Number of duplicate rows removed
    """
    # Convert set to sorted list for consistent column ordering
    unique_pairs = sorted(all_base_pairs)
    
    # Create the original matrix
    num_structures = len(structure_base_pairs)
    num_pairs = len(unique_pairs)
    original_matrix = np.zeros((num_structures, num_pairs), dtype=int)
    
    # Create ID to row mapping for original matrix
    original_id_to_row_mapping = {}
    if valid_structures:
        for i, (seq_id, sequence, structure_str) in enumerate(valid_structures):
            original_id_to_row_mapping[seq_id] = i
    
    # Fill the original matrix
    for i, structure_pairs in enumerate(structure_base_pairs):
        for pair in structure_pairs:
            if pair in unique_pairs:
                col_idx = unique_pairs.index(pair)
                original_matrix[i, col_idx] = 1
    
    # Pre-aggregate rows if requested
    if use_pre_aggregation:
        matrix, row_mapping = _pre_aggregate_rows(original_matrix)
        pre_aggregation_shape = matrix.shape
        duplicates_removed = original_matrix.shape[0] - pre_aggregation_shape[0]
        
        # Keep the original ID to row mapping (maps to original row indices)
        # This is needed for CSV generation where we use representative_idx (original row index)
        id_to_row_mapping = original_id_to_row_mapping
    else:
        matrix = original_matrix
        row_mapping = {i: [i] for i in range(original_matrix.shape[0])}
        pre_aggregation_shape = original_matrix.shape
        duplicates_removed = 0
        id_to_row_mapping = original_id_to_row_mapping
    
    return matrix, unique_pairs, id_to_row_mapping, original_matrix.shape, pre_aggregation_shape, row_mapping, duplicates_removed

def extract_structure_units(structures_data, include_stem_runs=False):
    """
    Extract all structure units from the structures using structures.py.
    Stacking pairs are always included.
    
    Returns:
    - all_structure_units: set of all unique structure units
    - structure_units_list: list of structure units for each structure
    - valid_structures: list of valid (seq_id, sequence, structure_str) tuples
    """
    all_structure_units = set()
    structure_units_list = []
    valid_structures = []
    
    for seq_id, sequence, structure_str in structures_data:
        try:
            # Extract structure units using structures.py
            units = rna_units_from_dotbracket(structure_str, 
                                            include_stem_runs=include_stem_runs,
                                            sequence=sequence)
            
            # Convert units to hashable format for set operations
            hashable_units = []
            for unit in units:
                if isinstance(unit, tuple):
                    # Convert tuple to string representation for hashing
                    hashable_unit = str(unit)
                else:
                    hashable_unit = str(unit)
                hashable_units.append(hashable_unit)
                all_structure_units.add(hashable_unit)
            
            structure_units_list.append(hashable_units)
            valid_structures.append((seq_id, sequence, structure_str))
            
        except Exception as e:
            print(f"Warning: Error processing structure for {seq_id}: {e}")
            # Add empty list for failed structures
            structure_units_list.append([])
    
    return all_structure_units, structure_units_list, valid_structures

def create_structure_units_matrix(all_structure_units, structure_units_list, valid_structures=None, use_pre_aggregation=True):
    """
    Create a binary matrix where:
    - Rows represent RNA structures
    - Columns represent unique structure units
    - Cell value is 1 if the structure contains that structure unit, 0 otherwise
    
    Args:
        all_structure_units: Set of all unique structure units
        structure_units_list: List of structure units for each structure
        valid_structures: List of (seq_id, sequence, structure_str) tuples
        use_pre_aggregation: Whether to pre-aggregate identical rows
    
    Returns:
        - matrix: Binary matrix (aggregated if use_pre_aggregation=True)
        - unique_units: List of unique structure units (column mapping)
        - id_to_row_mapping: Dictionary mapping sequence ID to row index
        - original_matrix_shape: Original matrix shape before aggregation
        - pre_aggregation_shape: Matrix shape after aggregation
        - row_mapping: Dictionary mapping aggregated row index to original row indices
        - duplicates_removed: Number of duplicate rows removed
    """
    # Convert set to sorted list for consistent column ordering
    unique_units = sorted(all_structure_units)
    
    # Create the original matrix
    num_structures = len(structure_units_list)
    num_units = len(unique_units)
    original_matrix = np.zeros((num_structures, num_units), dtype=int)
    
    # Create ID to row mapping for original matrix
    original_id_to_row_mapping = {}
    if valid_structures:
        for i, (seq_id, sequence, structure_str) in enumerate(valid_structures):
            original_id_to_row_mapping[seq_id] = i
    
    # Fill the original matrix
    for i, structure_units in enumerate(structure_units_list):
        for unit in structure_units:
            if unit in unique_units:
                col_idx = unique_units.index(unit)
                original_matrix[i, col_idx] = 1
    
    # Pre-aggregate rows if requested
    if use_pre_aggregation:
        matrix, row_mapping = _pre_aggregate_rows(original_matrix)
        pre_aggregation_shape = matrix.shape
        duplicates_removed = original_matrix.shape[0] - pre_aggregation_shape[0]
        
        # Keep the original ID to row mapping (maps to original row indices)
        # This is needed for CSV generation where we use representative_idx (original row index)
        id_to_row_mapping = original_id_to_row_mapping
    else:
        matrix = original_matrix
        row_mapping = {i: [i] for i in range(original_matrix.shape[0])}
        pre_aggregation_shape = original_matrix.shape
        duplicates_removed = 0
        id_to_row_mapping = original_id_to_row_mapping
    
    return matrix, unique_units, id_to_row_mapping, original_matrix.shape, pre_aggregation_shape, row_mapping, duplicates_removed

def create_structure_units_matrix_with_weights(all_structure_units, structure_units_list, valid_structures=None, use_pre_aggregation=True):
    """
    Create a binary matrix with weights where:
    - Rows represent RNA structures
    - Columns represent unique structure units
    - Cell value is 1 if the structure contains that structure unit, 0 otherwise
    - Weights are calculated based on nucleotide counts in each structure unit
    
    Args:
        all_structure_units: Set of all unique structure units
        structure_units_list: List of structure units for each structure
        valid_structures: List of (seq_id, sequence, structure_str) tuples
        use_pre_aggregation: Whether to pre-aggregate identical rows
    
    Returns:
        - matrix: Binary matrix (aggregated if use_pre_aggregation=True)
        - weights: Array of weights for each column (structure unit)
        - unique_units: List of unique structure units (column mapping)
        - id_to_row_mapping: Dictionary mapping sequence ID to row index
        - original_matrix_shape: Original matrix shape before aggregation
        - pre_aggregation_shape: Matrix shape after aggregation
        - row_mapping: Dictionary mapping aggregated row index to original row indices
        - duplicates_removed: Number of duplicate rows removed
    """
    # Convert set to sorted list for consistent column ordering
    unique_units = sorted(all_structure_units)
    
    # Calculate weights for each structure unit
    # Use the first valid structure to determine RNA length
    rna_length = 100  # Default length
    if valid_structures:
        # Use the length of the first sequence as reference
        rna_length = len(valid_structures[0][1])  # sequence length
    
    # Calculate weights using the weight calculation function
    weights = calculate_structure_unit_weights(unique_units, rna_length)
    
    # Create the original matrix
    num_structures = len(structure_units_list)
    num_units = len(unique_units)
    original_matrix = np.zeros((num_structures, num_units), dtype=int)
    
    # Create ID to row mapping for original matrix
    original_id_to_row_mapping = {}
    if valid_structures:
        for i, (seq_id, sequence, structure_str) in enumerate(valid_structures):
            original_id_to_row_mapping[seq_id] = i
    
    # Fill the original matrix
    for i, structure_units in enumerate(structure_units_list):
        for unit in structure_units:
            if unit in unique_units:
                col_idx = unique_units.index(unit)
                original_matrix[i, col_idx] = 1
    
    # Pre-aggregate rows if requested
    if use_pre_aggregation:
        matrix, row_mapping = _pre_aggregate_rows(original_matrix)
        pre_aggregation_shape = matrix.shape
        duplicates_removed = original_matrix.shape[0] - pre_aggregation_shape[0]
        
        # Keep the original ID to row mapping (maps to original row indices)
        id_to_row_mapping = original_id_to_row_mapping
    else:
        matrix = original_matrix
        row_mapping = {i: [i] for i in range(original_matrix.shape[0])}
        pre_aggregation_shape = original_matrix.shape
        duplicates_removed = 0
        id_to_row_mapping = original_id_to_row_mapping
    
    return matrix, weights, unique_units, id_to_row_mapping, original_matrix.shape, pre_aggregation_shape, row_mapping, duplicates_removed

def load_structure_units_matrix_with_weights_from_fasta(fasta_file: str, max_structures=None, 
                                                      include_stem_runs=False, use_pre_aggregation=True):
    """
    Load structure units matrix with weights from FASTA file.
    
    Stacking pairs are always included.
    
    Args:
        fasta_file: Path to FASTA file
        max_structures: Maximum number of structures to process
        include_stem_runs: Whether to include stem runs
        use_pre_aggregation: Whether to pre-aggregate identical rows
        
    Returns:
        Tuple of (matrix, weights, unique_units, valid_structures, id_to_row_mapping, 
                 original_matrix_shape, pre_aggregation_shape, row_mapping, duplicates_removed, gapped_mappings)
    """
    # Parse structure file (supports FASTA, text, JSON, CSV formats)
    try:
        structures_data = parse_structure_file(fasta_file)
    except Exception as e:
        # Fallback to old FASTA parser for backward compatibility
        try:
            structures_data = parse_fasta_file(fasta_file)
        except IndexError as idx_err:
            if "list index out of range" in str(idx_err):
                # Handle empty lines in file
                print(f"⚠️  Warning: Empty lines detected in file. Attempting to clean file...")
                structures_data = parse_fasta_file_robust(fasta_file)
            else:
                raise
        except Exception:
            raise e
    
    if max_structures:
        structures_data = structures_data[:max_structures]
    
    # Check for gapped sequences and extract structure units
    all_structure_units = set()
    structure_units_list = []
    valid_structures = []
    gapped_mappings = []  # Store gapped to ungapped mappings for each structure
    
    for seq_id, sequence, structure_str in structures_data:
        try:
            # Check if sequence has gaps
            has_gaps = '-' in sequence
            
            if has_gaps:
                # Parse gapped structure
                gapped_units, ungapped_units, gapped_to_ungapped, ungapped_to_gapped, ungapped_sequence = parse_gapped_structure(sequence, structure_str)
                
                # Use gapped units for matrix (for ILP/partition)
                units_for_matrix = gapped_units
                
                # Store mapping for CSV generation
                gapped_mappings.append({
                    'gapped_to_ungapped': gapped_to_ungapped,
                    'ungapped_to_gapped': ungapped_to_gapped,
                    'gapped_units': gapped_units,
                    'ungapped_units': ungapped_units,
                    'gapped_sequence': sequence,
                    'ungapped_sequence': ungapped_sequence,
                    'gapped_structure': structure_str
                })
            else:
                # No gaps - use regular parsing
                units_for_matrix = rna_units_from_dotbracket(structure_str, 
                                                          include_stem_runs=include_stem_runs,
                                                          sequence=sequence)
                
                # For ungapped sequences, mapping is identity
                gapped_mappings.append({
                    'gapped_to_ungapped': {i: i for i in range(1, len(sequence) + 1)},
                    'ungapped_to_gapped': {i: i for i in range(1, len(sequence) + 1)},
                    'gapped_units': units_for_matrix,
                    'ungapped_units': units_for_matrix,
                    'gapped_sequence': sequence,
                    'ungapped_sequence': sequence,
                    'gapped_structure': structure_str
                })
            
            # Convert units to hashable format for set operations
            hashable_units = []
            for unit in units_for_matrix:
                if isinstance(unit, tuple):
                    hashable_unit = str(unit)
                else:
                    hashable_unit = str(unit)
                hashable_units.append(hashable_unit)
                all_structure_units.add(hashable_unit)
            
            structure_units_list.append(hashable_units)
            valid_structures.append((seq_id, sequence, structure_str))
            
        except Exception as e:
            print(f"Warning: Error processing structure for {seq_id}: {e}")
            structure_units_list.append([])
            gapped_mappings.append(None)
    
    matrix, weights, unique_units, id_to_row_mapping, original_matrix_shape, pre_aggregation_shape, row_mapping, duplicates_removed = create_structure_units_matrix_with_weights(
        all_structure_units, structure_units_list, valid_structures, use_pre_aggregation)
    
    return matrix, weights, unique_units, valid_structures, id_to_row_mapping, original_matrix_shape, pre_aggregation_shape, row_mapping, duplicates_removed, gapped_mappings

def load_structure_units_matrix_from_fasta(fasta_file: str, max_structures=None, 
                                         include_stem_runs=False, use_pre_aggregation=True):
    """
    Load structure units matrix from FASTA file.
    
    Stacking pairs are always included.
    
    Args:
        fasta_file: Path to FASTA file
        max_structures: Maximum number of structures to process
        include_stem_runs: Whether to include stem runs
        use_pre_aggregation: Whether to pre-aggregate identical rows
        
    Returns:
        Tuple of (matrix, unique_units, valid_structures, id_to_row_mapping, 
                 original_matrix_shape, pre_aggregation_shape, row_mapping, duplicates_removed, gapped_mappings)
    """
    # Parse structure file (supports FASTA, text, JSON, CSV formats)
    try:
        structures_data = parse_structure_file(fasta_file)
    except Exception as e:
        # Fallback to old FASTA parser for backward compatibility
        try:
            structures_data = parse_fasta_file(fasta_file)
        except IndexError as idx_err:
            if "list index out of range" in str(idx_err):
                # Handle empty lines in FASTA file
                print(f"⚠️  Warning: Empty lines detected in file. Attempting to clean file...")
                structures_data = parse_fasta_file_robust(fasta_file)
            else:
                raise
        except Exception:
            raise e
    
    if max_structures:
        structures_data = structures_data[:max_structures]
    
    # Check for gapped sequences and extract structure units
    all_structure_units = set()
    structure_units_list = []
    valid_structures = []
    gapped_mappings = []  # Store gapped to ungapped mappings for each structure
    
    for seq_id, sequence, structure_str in structures_data:
        try:
            # Check if sequence has gaps
            has_gaps = '-' in sequence
            
            if has_gaps:
                # Parse gapped structure
                gapped_units, ungapped_units, gapped_to_ungapped, ungapped_to_gapped, ungapped_sequence = parse_gapped_structure(sequence, structure_str)
                
                # Use gapped units for matrix (for ILP/partition)
                units_for_matrix = gapped_units
                
                # Store mapping for CSV generation
                gapped_mappings.append({
                    'gapped_to_ungapped': gapped_to_ungapped,
                    'ungapped_to_gapped': ungapped_to_gapped,
                    'gapped_units': gapped_units,
                    'ungapped_units': ungapped_units,
                    'gapped_sequence': sequence,
                    'ungapped_sequence': ungapped_sequence,
                    'gapped_structure': structure_str
                })
            else:
                # No gaps - use regular parsing
                units_for_matrix = rna_units_from_dotbracket(structure_str, 
                                                          include_stem_runs=include_stem_runs,
                                                          sequence=sequence)
                
                # For ungapped sequences, mapping is identity
                gapped_mappings.append({
                    'gapped_to_ungapped': {i: i for i in range(1, len(sequence) + 1)},
                    'ungapped_to_gapped': {i: i for i in range(1, len(sequence) + 1)},
                    'gapped_units': units_for_matrix,
                    'ungapped_units': units_for_matrix,
                    'gapped_sequence': sequence,
                    'ungapped_sequence': sequence,
                    'gapped_structure': structure_str
                })
            
            # Convert units to hashable format for set operations
            hashable_units = []
            for unit in units_for_matrix:
                if isinstance(unit, tuple):
                    hashable_unit = str(unit)
                else:
                    hashable_unit = str(unit)
                hashable_units.append(hashable_unit)
                all_structure_units.add(hashable_unit)
            
            structure_units_list.append(hashable_units)
            valid_structures.append((seq_id, sequence, structure_str))
            
        except Exception as e:
            print(f"Warning: Error processing structure for {seq_id}: {e}")
            structure_units_list.append([])
            gapped_mappings.append(None)
    
    matrix, unique_units, id_to_row_mapping, original_matrix_shape, pre_aggregation_shape, row_mapping, duplicates_removed = create_structure_units_matrix(
        all_structure_units, structure_units_list, valid_structures, use_pre_aggregation)
    
    return matrix, unique_units, valid_structures, id_to_row_mapping, original_matrix_shape, pre_aggregation_shape, row_mapping, duplicates_removed, gapped_mappings

def parse_pairs_tree(dotbr):
    """
    Parse a dot-bracket structure and return base pairs with hierarchical relationships.
    Only handles () brackets (for backward compatibility).
    """
    stack = []
    pairs = []
    children_of = {}  # key: open index i of (i,j) ; val: list of immediate child pairs

    for pos, ch in enumerate(dotbr, start=1):
        if ch == '(':
            stack.append(pos)
        elif ch == ')':
            i = stack.pop()
            j = pos
            if i > j: i, j = j, i
            pairs.append((i, j))
            if stack:
                parent_open = stack[-1]
                children_of.setdefault(parent_open, []).append((i, j))
            children_of.setdefault(i, [])
        # '.' -> nothing

    for key in children_of:
        children_of[key].sort(key=lambda p: p[0])
    pairs.sort(key=lambda p: p[0])
    return pairs, children_of

def parse_pairs_tree_general(dotbr, bracket_type='()'):
    """
    Parse a dot-bracket structure with any bracket type and return base pairs with hierarchical relationships.
    
    Args:
        dotbr: Dot-bracket structure string
        bracket_type: String like '()', '[]', '{}' specifying the bracket pair to use
        
    Returns:
        Tuple of (pairs, children_of) where pairs is list of (i,j) tuples and 
        children_of maps parent opening position to list of child pairs
    """
    if len(bracket_type) != 2:
        raise ValueError(f"bracket_type must be 2 characters, got: {bracket_type}")
    
    open_char, close_char = bracket_type[0], bracket_type[1]
    
    stack = []
    pairs = []
    children_of = {}  # key: open index i of (i,j) ; val: list of immediate child pairs

    for pos, ch in enumerate(dotbr, start=1):
        if ch == open_char:
            stack.append(pos)
        elif ch == close_char:
            i = stack.pop()
            j = pos
            if i > j: i, j = j, i
            pairs.append((i, j))
            if stack:
                parent_open = stack[-1]
                children_of.setdefault(parent_open, []).append((i, j))
            children_of.setdefault(i, [])
        # '.' -> nothing

    for key in children_of:
        children_of[key].sort(key=lambda p: p[0])
    pairs.sort(key=lambda p: p[0])
    return pairs, children_of

def decompose_pseudoknot_structure(dotbr):
    """
    Decompose a pseudoknotted structure into multiple pseudoknot-free substructures.
    
    Args:
        dotbr: Dot-bracket structure string (e.g., "((..[[..))..]]")
        
    Returns:
        List of tuples: [(substructure_string, bracket_type), ...]
        where bracket_type is '()', '[]', '{}', etc.
    """
    # Find all bracket types present
    bracket_types = set()
    for char in dotbr:
        if char in '([{':
            bracket_types.add(char)
    
    substructures = []
    
    for bracket_char in sorted(bracket_types):
        # Create substructure with only this bracket type and dots
        substructure = ""
        closing_char = {'(': ')', '[': ']', '{': '}'}[bracket_char]
        
        for char in dotbr:
            if char == bracket_char or char == closing_char or char == '.':
                substructure += char
            else:
                substructure += '.'
        
        # Only add if it contains actual brackets
        if bracket_char in substructure and closing_char in substructure:
            substructures.append((substructure, bracket_char + closing_char))
    
    return substructures

def parse_pseudoknot_structure(dotbr, include_stem_runs=False, sequence=None):
    """
    Parse a pseudoknotted structure by decomposing it into substructures.
    
    Stacking pairs are always included.
    
    Args:
        dotbr: Dot-bracket structure string (may contain pseudoknots)
        include_stem_runs: Whether to include stem runs
        sequence: Optional sequence string
        
    Returns:
        Tuple of (all_pairs, all_units, substructure_info)
        - all_pairs: Combined list of all base pairs from all substructures
        - all_units: Combined list of all structure units from all substructures
        - substructure_info: List of info about each substructure processed
    """
    # Decompose into substructures
    substructures = decompose_pseudoknot_structure(dotbr)
    
    all_pairs = []
    all_units = []
    substructure_info = []
    
    for substructure_str, bracket_type in substructures:
        try:
            # Parse base pairs for this substructure using the appropriate bracket type
            pairs, children_of = parse_pairs_tree_general(substructure_str, bracket_type)
            
            # Extract structure units for this substructure
            # We need to create a temporary version of rna_units_from_dotbracket that works with any bracket type
            units = rna_units_from_dotbracket_general(substructure_str, bracket_type,
                                                   include_stem_runs=include_stem_runs,
                                                   sequence=sequence)
            
            # Add to combined results
            all_pairs.extend(pairs)
            all_units.extend(units)
            
            substructure_info.append({
                'substructure': substructure_str,
                'bracket_type': bracket_type,
                'pairs': pairs,
                'units': units,
                'num_pairs': len(pairs),
                'num_units': len(units)
            })
            
        except Exception as e:
            print(f"Warning: Error processing substructure {substructure_str}: {e}")
            substructure_info.append({
                'substructure': substructure_str,
                'bracket_type': bracket_type,
                'pairs': [],
                'units': [],
                'num_pairs': 0,
                'num_units': 0,
                'error': str(e)
            })
    
    return all_pairs, all_units, substructure_info

def classify_interior_or_bulge(outer, inner):
    """
    Classify the loop between outer and inner pairs.
    
    Args:
        outer: (i, j) - outer pair
        inner: (l, k) - inner pair
        
    Returns:
        "stacking" if consecutive pairs (no unpaired nucleotides)
        "interior" if unpaired nucleotides on both sides
        "bulge" if unpaired nucleotides on only one side
    """
    i, j = outer
    l, k = inner
    left_len  = max(0, l - i - 1)
    right_len = max(0, j - k - 1)
    
    # If no unpaired nucleotides on either side, it's a stacking pair
    if left_len == 0 and right_len == 0:
        return "stacking"
    # If unpaired nucleotides on only one side, it's a bulge
    elif (left_len == 0) ^ (right_len == 0):
        return "bulge"
    # If unpaired nucleotides on both sides, it's an interior loop
    else:
        return "interior"

def stacking_pairs(pairs, children_of):
    """
    Return stacking pairs as two pairs: (i,j) and (i+1,j-1).
    Only include pairs that are truly stacking (not hairpin starts or loop boundaries).
    """
    P = set(pairs)
    stacks = []
    for (i, j) in pairs:
        if (i + 1, j - 1) in P:
            # Check if this is a true stacking pair
            # A pair (i,j) is stacking if:
            # 1. (i+1, j-1) exists
            # 2. (i,j) has children (not a hairpin start)
            # 3. (i+1, j-1) is a direct child of (i,j)
            children_outer = children_of.get(i, [])
            
            # Only consider it stacking if the outer pair has children (not a hairpin start)
            # and the inner pair is a direct child of the outer pair
            if len(children_outer) > 0:
                # Check if (i+1, j-1) is a direct child of (i, j)
                if (i + 1, j - 1) in children_outer:
                    stacks.append(((i, j), (i + 1, j - 1)))
        # optional: include the innermost pair too? Uncomment below to include "roots" of stacks.
        # elif (i - 1, j + 1) in P:
        #     stacks.append(((i, j), (i - 1, j + 1)))
    return stacks

def group_stems(pairs):
    """
    Optional: group stacking pairs into contiguous stems (runs).
    Returns list of stems, each as a list of pairs [(i1,j1), (i2,j2), ...] with i increasing.
    """
    P = set(pairs)
    visited = set()
    stems = []

    for (i, j) in sorted(pairs):
        if (i, j) in visited:
            continue
        # extend outward while neighbors exist
        run = [(i, j)]
        visited.add((i, j))

        # extend inward (i+1, j-1)
        ii, jj = i, j
        while (ii + 1, jj - 1) in P and (ii + 1, jj - 1) not in visited:
            ii, jj = ii + 1, jj - 1
            run.append((ii, jj))
            visited.add((ii, jj))

        # extend outward (i-1, j+1)
        ii, jj = i, j
        while (ii - 1, jj + 1) in P and (ii - 1, jj + 1) not in visited:
            ii, jj = ii - 1, jj + 1
            run.insert(0, (ii, jj))
            visited.add((ii, jj))

        stems.append(run)
    return stems

def find_unpaired_in_multiloop(dotbr, closing_pairs, sequence=None):
    """
    Find unpaired nucleotide indices that are directly on the multiloop ring.
    This matches the behavior of forgi.graph.bulge_graph module's mloop_iterator.
    
    Args:
        dotbr: Dot-bracket structure string
        closing_pairs: List of closing pairs in the multiloop
        sequence: Optional sequence string to filter out gap positions
        
    Returns:
        List of unpaired nucleotide indices on the multiloop ring
    """
    unpaired_indices = []
    
    # Sort closing pairs by their opening position
    sorted_pairs = sorted(closing_pairs, key=lambda p: p[0])
    
    # Find the outer closing pair (the one with the largest closing position)
    outer_pair = max(closing_pairs, key=lambda p: p[1])
    outer_closing_pos = outer_pair[1]
    
    # Find unpaired nucleotides between consecutive opening brackets
    # These are the nucleotides directly on the multiloop ring
    for i in range(len(sorted_pairs)):
        current_pair = sorted_pairs[i]
        next_pair = sorted_pairs[(i + 1) % len(sorted_pairs)]
        
        # Get the range between current pair's opening and next pair's opening
        start_pos = current_pair[0] + 1  # After opening bracket
        end_pos = next_pair[0] - 1       # Before next opening bracket
        
        # Only check if there are positions in this range
        if start_pos <= end_pos:
            # Check each position in this range for unpaired nucleotides
            for pos in range(start_pos, end_pos + 1):
                if dotbr[pos - 1] == '.':  # -1 because dotbr is 0-indexed
                    # Check if this position is directly on the ring
                    # by ensuring it's not within any nested structure
                    is_on_ring = True
                    
                    # Count opening brackets before this position within the multiloop range
                    open_count = 0
                    for j in range(start_pos - 1, pos - 1):  # -1 for 0-indexing
                        if dotbr[j] == '(':
                            open_count += 1
                        elif dotbr[j] == ')':
                            open_count -= 1
                    
                    # If there are unmatched opening brackets, this position is in a nested structure
                    if open_count > 0:
                        is_on_ring = False
                    
                    if is_on_ring and is_nucleotide_position(pos, sequence):
                        unpaired_indices.append(pos)
    
    # Also check unpaired nucleotides between the last closing pair's closing bracket
    # and the outer closing pair's closing bracket
    # Find the last closing pair (the one with the largest closing position among non-outer pairs)
    non_outer_pairs = [p for p in sorted_pairs if p != outer_pair]
    if non_outer_pairs:
        last_closing_pos = max(p[1] for p in non_outer_pairs)
        # Check positions between last closing and outer closing
        for pos in range(last_closing_pos + 1, outer_closing_pos):
            if dotbr[pos - 1] == '.' and is_nucleotide_position(pos, sequence):
                unpaired_indices.append(pos)
    
    return sorted(unpaired_indices)

def is_nucleotide_position(pos, sequence):
    """
    Check if a position contains a nucleotide (not a gap).
    
    Args:
        pos: 1-indexed position
        sequence: Sequence string
        
    Returns:
        True if position contains a nucleotide, False if it's a gap
    """
    if sequence is None:
        return True  # No sequence provided, assume all positions are nucleotides
    if pos < 1 or pos > len(sequence):
        return False  # Position out of range
    return sequence[pos - 1] != '-'  # Check if it's not a gap

def rna_units_from_dotbracket_general(dotbr, bracket_type='()', include_stem_runs=False, sequence=None):
    """
    Emit loop units + stacking pairs (and optional stem runs) for any bracket type.
    
    Returns structure units in format: ({base pairs}, {unpaired nucleotides})
    where base pairs is a set of tuples (i, j) and unpaired nucleotides is a set of indices.
    
    Stacking pairs are always included.
    
    Args:
        dotbr: Dot-bracket structure string
        bracket_type: String like '()', '[]', '{}' specifying the bracket pair to use
        include_stem_runs: Whether to include stem runs
        sequence: Optional sequence string to filter out gap positions
    """
    pairs, children_of = parse_pairs_tree_general(dotbr, bracket_type)
    units = []

    # --- stacking pairs (always included) ---
    stacking_units = stacking_pairs(pairs, children_of)  # each as ((i, j), (i+1, j-1))
    # Convert to new format: ({base pairs}, {unpaired nucleotides})
    for unit in stacking_units:
        bp1, bp2 = unit
        units.append(({bp1, bp2}, set()))

    # (optional) stems as contiguous runs
    if include_stem_runs:
        for run in group_stems(pairs):
            # Convert to new format: ({base pairs}, {unpaired nucleotides})
            units.append((set(run), set()))

    # --- loops from pair tree ---
    for (i, j) in pairs:
        children = children_of.get(i, [])
        if len(children) == 0:
            # Hairpin: convert to ({base pairs}, {unpaired nucleotides})
            unpaired_indices_hp = [pos for pos in range(i + 1, j) 
                                 if dotbr[pos - 1] == '.' and is_nucleotide_position(pos, sequence)]
            units.append(({(i, j)}, set(unpaired_indices_hp)))
        elif len(children) == 1:
            (l, k) = children[0]
            # i < l < k < j holds for valid pseudoknot-free structures
            loop_type = classify_interior_or_bulge((i, j), (l, k))
            
            # Only add as a loop unit if it's not a stacking pair
            # (stacking pairs are already handled by stacking_pairs function)
            if loop_type != "stacking":
                # Interior/bulge: convert to ({base pairs}, {unpaired nucleotides})
                left_nucleotides = [pos for pos in range(i + 1, l) 
                                  if dotbr[pos - 1] == '.' and is_nucleotide_position(pos, sequence)]
                right_nucleotides = [pos for pos in range(k + 1, j) 
                                   if dotbr[pos - 1] == '.' and is_nucleotide_position(pos, sequence)]
                units.append(({(i, j), (l, k)}, set(left_nucleotides + right_nucleotides)))
        else:
            cps = [(i, j)] + children[:]          # outer closing pair + all child stems
            cps = [(a, b) if a < b else (b, a) for (a, b) in cps]
            cps.sort(key=lambda p: p[0])
            
            # Find unpaired nucleotides in the multiloop
            unpaired_indices = find_unpaired_in_multiloop(dotbr, cps, sequence)
            
            # Convert to new format: ({base pairs}, {unpaired nucleotides})
            units.append((set(cps), set(unpaired_indices)))

    return units

def rna_units_from_dotbracket(dotbr, include_stem_runs=False, sequence=None):
    """
    Emit loop units + stacking pairs (and optional stem runs).
    
    Returns structure units in format: ({base pairs}, {unpaired nucleotides})
    where base pairs is a set of tuples (i, j) and unpaired nucleotides is a set of indices.
    
    Stacking pairs are always included.
    
    Args:
        dotbr: Dot-bracket structure string
        include_stem_runs: Whether to include stem runs
        sequence: Optional sequence string to filter out gap positions
    """
    pairs, children_of = parse_pairs_tree(dotbr)
    units = []

    # --- stacking pairs (always included) ---
    stacking_units = stacking_pairs(pairs, children_of)  # each as ((i, j), (i+1, j-1))
    # Convert to new format: ({base pairs}, {unpaired nucleotides})
    for unit in stacking_units:
        bp1, bp2 = unit
        units.append(({bp1, bp2}, set()))

    # (optional) stems as contiguous runs
    if include_stem_runs:
        for run in group_stems(pairs):
            # Convert to new format: ({base pairs}, {unpaired nucleotides})
            units.append((set(run), set()))

    # --- loops from pair tree ---
    for (i, j) in pairs:
        children = children_of.get(i, [])
        if len(children) == 0:
            # Hairpin: convert to ({base pairs}, {unpaired nucleotides})
            unpaired_indices_hp = [pos for pos in range(i + 1, j) 
                                 if dotbr[pos - 1] == '.' and is_nucleotide_position(pos, sequence)]
            units.append(({(i, j)}, set(unpaired_indices_hp)))
        elif len(children) == 1:
            (l, k) = children[0]
            # i < l < k < j holds for valid pseudoknot-free structures
            loop_type = classify_interior_or_bulge((i, j), (l, k))
            
            # Only add as a loop unit if it's not a stacking pair
            # (stacking pairs are already handled by stacking_pairs function)
            if loop_type != "stacking":
                # Interior/bulge: convert to ({base pairs}, {unpaired nucleotides})
                left_nucleotides = [pos for pos in range(i + 1, l) 
                                  if dotbr[pos - 1] == '.' and is_nucleotide_position(pos, sequence)]
                right_nucleotides = [pos for pos in range(k + 1, j) 
                                   if dotbr[pos - 1] == '.' and is_nucleotide_position(pos, sequence)]
                units.append(({(i, j), (l, k)}, set(left_nucleotides + right_nucleotides)))
        else:
            cps = [(i, j)] + children[:]          # outer closing pair + all child stems
            cps = [(a, b) if a < b else (b, a) for (a, b) in cps]
            cps.sort(key=lambda p: p[0])
            
            # Find unpaired nucleotides in the multiloop
            unpaired_indices = find_unpaired_in_multiloop(dotbr, cps, sequence)
            
            # Convert to new format: ({base pairs}, {unpaired nucleotides})
            units.append((set(cps), set(unpaired_indices)))

    return units

def extract_base_pairs_from_structure_units(structure_units: List) -> set:
    """
    Extract all base pairs from a list of structure units.
    
    Args:
        structure_units: List of structure units in format ({base pairs}, {unpaired nucleotides})
        or legacy formats (for backward compatibility)
        
    Returns:
        Set of base pairs as tuples (i, j) where i < j
    
    Structure unit formats supported:
    - New format: ({base pairs}, {unpaired nucleotides}) - preferred format
    - Legacy formats (for backward compatibility):
      - Stacking pair: ((i, j), (i+1, j-1))
      - Hairpin: ((i, j), [unpaired_indices])
      - Interior/bulge: ((i, j), (l, k), [left_nucleotides], [right_nucleotides])
      - Multiloop: (pairs, 'multiloop', unpaired_indices)
      - Stem run: (pairs, 'stem_run')
    """
    # Parse string units if needed
    parsed_units = []
    for unit in structure_units:
        if isinstance(unit, str):
            try:
                import ast
                unit = ast.literal_eval(unit)
            except:
                # Try eval as fallback (less safe but sometimes needed)
                try:
                    unit = eval(unit)
                except:
                    continue  # Skip invalid units
        parsed_units.append(unit)
    
    # Extract all base pairs from units
    all_pairs = set()
    
    for unit in parsed_units:
        if not isinstance(unit, tuple):
            continue
        
        # Check for new format: ({base pairs}, {unpaired nucleotides})
        if len(unit) == 2 and isinstance(unit[0], set) and isinstance(unit[1], set):
            # New format: ({base pairs}, {unpaired nucleotides})
            all_pairs.update(unit[0])
        # Legacy format handling for backward compatibility
        elif len(unit) == 2 and isinstance(unit[0], tuple) and isinstance(unit[1], tuple):
            # Stacking pair: ((i, j), (i+1, j-1))
            all_pairs.add(unit[0])
            all_pairs.add(unit[1])
        elif len(unit) == 2 and isinstance(unit[0], tuple) and isinstance(unit[1], (list, tuple)):
            # Hairpin: ((i, j), [unpaired_indices])
            all_pairs.add(unit[0])
        elif len(unit) == 4 and isinstance(unit[0], tuple) and isinstance(unit[1], tuple):
            # Interior/bulge: ((i, j), (l, k), [left_nucleotides], [right_nucleotides])
            all_pairs.add(unit[0])
            all_pairs.add(unit[1])
        elif len(unit) == 3 and unit[1] == 'multiloop':
            # Multiloop: (pairs, 'multiloop', unpaired_indices)
            pairs = unit[0]
            for pair in pairs:
                if isinstance(pair, tuple) and len(pair) >= 2:
                    all_pairs.add((pair[0], pair[1]))
        elif len(unit) == 2 and unit[1] == 'stem_run':
            # Stem run: (pairs, 'stem_run')
            pairs = unit[0]
            for pair in pairs:
                if isinstance(pair, tuple) and len(pair) >= 2:
                    all_pairs.add((pair[0], pair[1]))
    
    return all_pairs

def _detect_crossing_pairs(pairs):
    """
    Detect if pairs cross each other (indicating pseudoknots).
    
    Args:
        pairs: Set or list of base pairs (i, j)
        
    Returns:
        bool: True if any pairs cross each other
    """
    pairs_list = list(pairs)
    for i in range(len(pairs_list)):
        for j in range(i + 1, len(pairs_list)):
            i1, j1 = pairs_list[i]
            i2, j2 = pairs_list[j]
            # Pairs cross if: i1 < i2 < j1 < j2 or i2 < i1 < j2 < j1
            if (i1 < i2 < j1 < j2) or (i2 < i1 < j2 < j1):
                return True
    return False

def _assign_bracket_types_to_pairs(pairs):
    """
    Assign bracket types to pairs based on pseudoknot levels.
    Uses graph coloring to assign different bracket types to crossing pairs.
    
    Args:
        pairs: Set or list of base pairs (i, j)
        
    Returns:
        Dictionary mapping (i, j) -> bracket_type where bracket_type is '()', '[]', '{}', etc.
    """
    pairs_list = list(pairs)
    bracket_types = ['()', '[]', '{}']
    pair_to_bracket = {}
    
    # Build conflict graph: pairs that cross each other
    conflicts = {pair: [] for pair in pairs_list}
    for i in range(len(pairs_list)):
        for j in range(i + 1, len(pairs_list)):
            p1 = pairs_list[i]
            p2 = pairs_list[j]
            i1, j1 = p1
            i2, j2 = p2
            # Check if pairs cross
            if (i1 < i2 < j1 < j2) or (i2 < i1 < j2 < j1):
                conflicts[p1].append(p2)
                conflicts[p2].append(p1)
    
    # Greedy coloring: assign bracket types to minimize conflicts
    for pair in pairs_list:
        # Find the first bracket type not used by conflicting pairs
        used_types = set()
        for conflict_pair in conflicts[pair]:
            if conflict_pair in pair_to_bracket:
                used_types.add(pair_to_bracket[conflict_pair])
        
        # Assign first available bracket type
        for bracket_type in bracket_types:
            if bracket_type not in used_types:
                pair_to_bracket[pair] = bracket_type
                break
        else:
            # If all types used, default to first type (shouldn't happen with 3 types)
            pair_to_bracket[pair] = bracket_types[0]
    
    return pair_to_bracket

def dotbracket_from_structure_units(structure_units: List, length: int = None) -> str:
    """
    Reconstruct a dot-bracket structure from a list of structure units.
    This is the inverse operation of rna_units_from_dotbracket().
    Handles pseudoknots by detecting crossing pairs and assigning different bracket types.
    
    Args:
        structure_units: List of structure units (tuples or strings)
        length: Optional sequence length. If not provided, will be inferred from units.
        
    Returns:
        Dot-bracket structure string (may contain multiple bracket types for pseudoknots)
    
    Structure unit formats:
    - Stacking pair: ((i, j), (i+1, j-1))
    - Hairpin: ((i, j), [unpaired_indices])
    - Interior/bulge: ((i, j), (l, k), [left_nucleotides], [right_nucleotides])
    - Multiloop: (pairs, 'multiloop', unpaired_indices)
    - Stem run: (pairs, 'stem_run')
    """
    # Parse string units if needed
    parsed_units = []
    for unit in structure_units:
        if isinstance(unit, str):
            try:
                import ast
                unit = ast.literal_eval(unit)
            except:
                # Try eval as fallback (less safe but sometimes needed)
                try:
                    unit = eval(unit)
                except:
                    continue  # Skip invalid units
        parsed_units.append(unit)
    
    # Step 1: Extract all base pairs from units
    all_pairs = extract_base_pairs_from_structure_units(structure_units)
    
    if not all_pairs:
        # No pairs found, return all dots
        inferred_length = length if length is not None else 1
        return '.' * inferred_length
    
    # Step 2: Determine structure length
    if length is None:
        # Find maximum index from all pairs
        max_idx = 0
        for i, j in all_pairs:
            max_idx = max(max_idx, i, j)
        # Add some padding if unpaired nucleotides are expected
        inferred_length = max_idx
    else:
        inferred_length = length
    
    # Ensure length is at least as large as the maximum pair index
    max_pair_idx = max([max(i, j) for i, j in all_pairs]) if all_pairs else 0
    inferred_length = max(inferred_length, max_pair_idx)
    
    # Step 3: Collect unpaired positions from units
    unpaired_positions = set()
    for unit in parsed_units:
        if isinstance(unit, tuple):
            if len(unit) == 2 and isinstance(unit[0], tuple) and isinstance(unit[1], (list, tuple)):
                # Hairpin: ((i, j), [unpaired_indices])
                unpaired_positions.update(unit[1])
            elif len(unit) == 4 and isinstance(unit[0], tuple) and isinstance(unit[1], tuple):
                # Interior/bulge: ((i, j), (l, k), [left_nucleotides], [right_nucleotides])
                if isinstance(unit[2], (list, tuple)):
                    unpaired_positions.update(unit[2])
                if isinstance(unit[3], (list, tuple)):
                    unpaired_positions.update(unit[3])
            elif len(unit) == 3 and unit[1] == 'multiloop':
                # Multiloop: (pairs, 'multiloop', unpaired_indices)
                if isinstance(unit[2], (list, tuple)):
                    unpaired_positions.update(unit[2])
    
    # Update length to account for unpaired positions
    if unpaired_positions:
        max_unpaired = max(unpaired_positions) if unpaired_positions else 0
        inferred_length = max(inferred_length, max_unpaired)
    
    # Step 4: Build parent-child relationship tree from pairs
    sorted_pairs = sorted(all_pairs, key=lambda p: p[0])
    structure = ['.'] * inferred_length
    
    # Detect pseudoknots early (before processing unpaired positions)
    has_pseudoknots = _detect_crossing_pairs(all_pairs)
    
    # Validate that unpaired positions don't conflict with pairs
    pair_positions = set()
    for i, j in all_pairs:
        pair_positions.add(i)
        pair_positions.add(j)
    
    conflicting_unpaired = unpaired_positions & pair_positions
    if conflicting_unpaired:
        # Remove conflicting unpaired positions (they're actually paired)
        # Pairs take precedence over unpaired positions
        unpaired_positions = unpaired_positions - pair_positions
        
        # Warn about conflicts, but note if pseudoknots are present (expected behavior)
        import warnings
        if has_pseudoknots:
            # Pseudoknots can cause conflicts due to multiple substructures - this is expected
            warnings.warn(f"Found {len(conflicting_unpaired)} unpaired positions that conflict with base pairs. "
                         f"This is expected with pseudoknots (multiple bracket types). "
                         f"Pair information takes precedence.", UserWarning)
        else:
            warnings.warn(f"Found {len(conflicting_unpaired)} unpaired positions that conflict with base pairs: "
                         f"{sorted(conflicting_unpaired)}. Pair information takes precedence.", UserWarning)
    
    # Initialize unpaired positions as dots (explicitly)
    for pos in unpaired_positions:
        if 0 < pos <= inferred_length:
            structure[pos-1] = '.'
    
    # Use loop structure information to determine unpaired positions more accurately
    # Process units to mark unpaired positions based on loop boundaries
    for unit in parsed_units:
        if isinstance(unit, tuple):
            if len(unit) == 4 and isinstance(unit[0], tuple) and isinstance(unit[1], tuple):
                # Interior/bulge: ((i, j), (l, k), [left_nucleotides], [right_nucleotides])
                outer_i, outer_j = unit[0]
                inner_l, inner_k = unit[1]
                left_unpaired = unit[2] if isinstance(unit[2], (list, tuple)) else []
                right_unpaired = unit[3] if isinstance(unit[3], (list, tuple)) else []
                
                # Mark explicitly unpaired positions from the unit
                for pos in left_unpaired:
                    if 0 < pos <= inferred_length and pos not in pair_positions:
                        structure[pos-1] = '.'
                for pos in right_unpaired:
                    if 0 < pos <= inferred_length and pos not in pair_positions:
                        structure[pos-1] = '.'
                
                # For positions between outer and inner pairs that aren't explicitly marked,
                # infer they are unpaired if they're not part of any pair
                for pos in range(outer_i + 1, inner_l):
                    if pos not in pair_positions and 0 < pos <= inferred_length:
                        # Check if position is not already a bracket character
                        if structure[pos-1] not in '([{)]}':
                            structure[pos-1] = '.'
                for pos in range(inner_k + 1, outer_j):
                    if pos not in pair_positions and 0 < pos <= inferred_length:
                        # Check if position is not already a bracket character
                        if structure[pos-1] not in '([{)]}':
                            structure[pos-1] = '.'
    
    # Fill in remaining positions between pairs as dots if not already set
    # This ensures positions that should be unpaired are marked
    for i, j in sorted_pairs:
        # Mark positions between i and j as dots if not part of any pair
        for pos in range(i + 1, j):
            if pos not in pair_positions and 0 < pos <= inferred_length:
                # Check if position is not already a bracket character
                if structure[pos-1] not in '([{)]}':
                    structure[pos-1] = '.'
    
    # Step 5: Assign bracket types based on pseudoknot detection
    
    if has_pseudoknots:
        # Assign bracket types to handle pseudoknots
        pair_to_bracket = _assign_bracket_types_to_pairs(all_pairs)
    else:
        # No pseudoknots, use standard parentheses for all pairs
        pair_to_bracket = {pair: '()' for pair in all_pairs}
    
    # Map bracket types to opening/closing characters
    bracket_chars = {
        '()': ('(', ')'),
        '[]': ('[', ']'),
        '{}': ('{', '}')
    }
    
    # Build parent-child relationships (only for non-crossing pairs)
    children_of = {}  # Maps pair (i,j) to list of child pairs
    
    # For each pair, find its direct parent (the immediate containing pair)
    for pair in sorted_pairs:
        children_of[pair] = []
        
    # Find direct parent for each pair (the immediate containing pair)
    # Skip parent-child relationships for crossing pairs
    for pair in sorted_pairs:
        p_i, p_j = pair
        best_parent = None
        best_span = float('inf')
        
        for parent in sorted_pairs:
            if parent == pair:
                continue
            par_i, par_j = parent
            # Check if parent contains pair: par_i < p_i < p_j < par_j
            # This automatically ensures pairs don't cross (proper nesting)
            if par_i < p_i and p_j < par_j:
                
                span = par_j - par_i
                # Find the smallest containing pair (immediate parent)
                if span < best_span:
                    # Check if current best_parent is between this parent and the pair
                    # If so, this parent is not the immediate parent
                    is_immediate = True
                    if best_parent is not None:
                        bp_i, bp_j = best_parent
                        # If best_parent is nested inside this parent, then this parent is not immediate
                        if par_i < bp_i < bp_j < par_j:
                            is_immediate = False
                    
                    if is_immediate:
                        best_parent = parent
                        best_span = span
        
        if best_parent is not None:
            children_of[best_parent].append(pair)
    
    # Step 6: Place pairs recursively using tree structure with appropriate bracket types
    def place_pair_tree(pair):
        """Place a pair and recursively place its children."""
        i, j = pair
        if i > 0 and i <= inferred_length and j > 0 and j <= inferred_length:
            # Get bracket type for this pair
            bracket_type = pair_to_bracket.get(pair, '()')
            open_char, close_char = bracket_chars[bracket_type]
            
            # Place the pair
            structure[i-1] = open_char
            structure[j-1] = close_char
            
            # Place children recursively
            for child in sorted(children_of[pair], key=lambda p: p[0]):
                place_pair_tree(child)
    
    # Start with root pairs (pairs with no parent)
    root_pairs = [pair for pair in sorted_pairs if pair not in 
                  [child for children_list in children_of.values() for child in children_list]]
    
    # Place all root pairs and their descendants
    for root_pair in sorted(root_pairs, key=lambda p: p[0]):
        place_pair_tree(root_pair)
    
    # Ensure any remaining pairs are placed (for crossing pairs that couldn't be in tree)
    for i, j in sorted_pairs:
        if 0 < i <= inferred_length and 0 < j <= inferred_length:
            bracket_type = pair_to_bracket.get((i, j), '()')
            open_char, close_char = bracket_chars[bracket_type]
            if structure[i-1] == '.':
                structure[i-1] = open_char
            if structure[j-1] == '.':
                structure[j-1] = close_char
    
    return ''.join(structure)

def dotbracket_from_base_pairs(base_pairs, length: int = None) -> str:
    """
    Reconstruct a dot-bracket structure from a list of base pairs.
    This is the inverse operation of parseBracketString_pk() and universal_parse_base_pairs().
    Handles pseudoknots by detecting crossing pairs and assigning different bracket types.
    
    Args:
        base_pairs: List or set of base pairs as (i, j) tuples (1-indexed)
        length: Optional sequence length. If not provided, will be inferred from pairs.
        
    Returns:
        Dot-bracket structure string (may contain multiple bracket types for pseudoknots)
    
    Example:
        >>> pairs = [(1, 6), (2, 5), (3, 4)]
        >>> dotbracket_from_base_pairs(pairs)
        '(((...)))'
        
        >>> pairs = [(1, 5), (2, 6), (3, 4)]  # Pseudoknot
        >>> dotbracket_from_base_pairs(pairs)
        '((..[[..))..]]'
    """
    # Convert to set and ensure pairs are tuples
    all_pairs = set(tuple(pair) for pair in base_pairs)
    
    if not all_pairs:
        # No pairs found, return all dots
        inferred_length = length if length is not None else 1
        return '.' * inferred_length
    
    # Step 1: Determine structure length
    if length is None:
        # Find maximum index from all pairs
        max_idx = 0
        for i, j in all_pairs:
            max_idx = max(max_idx, i, j)
        inferred_length = max_idx
    else:
        inferred_length = length
    
    # Ensure length is at least as large as the maximum pair index
    max_pair_idx = max([max(i, j) for i, j in all_pairs]) if all_pairs else 0
    inferred_length = max(inferred_length, max_pair_idx)
    
    # Step 2: Initialize structure with dots
    structure = ['.'] * inferred_length
    
    # Step 3: Detect pseudoknots
    has_pseudoknots = _detect_crossing_pairs(all_pairs)
    
    # Step 4: Assign bracket types based on pseudoknot detection
    if has_pseudoknots:
        # Assign bracket types to handle pseudoknots
        pair_to_bracket = _assign_bracket_types_to_pairs(all_pairs)
    else:
        # No pseudoknots, use standard parentheses for all pairs
        pair_to_bracket = {pair: '()' for pair in all_pairs}
    
    # Map bracket types to opening/closing characters
    bracket_chars = {
        '()': ('(', ')'),
        '[]': ('[', ']'),
        '{}': ('{', '}')
    }
    
    # Step 5: Build parent-child relationships (only for non-crossing pairs)
    sorted_pairs = sorted(all_pairs, key=lambda p: p[0])
    children_of = {}  # Maps pair (i,j) to list of child pairs
    
    # Initialize children_of for all pairs
    for pair in sorted_pairs:
        children_of[pair] = []
    
    # Find direct parent for each pair (the immediate containing pair)
    # Skip parent-child relationships for crossing pairs
    for pair in sorted_pairs:
        p_i, p_j = pair
        best_parent = None
        best_span = float('inf')
        
        for parent in sorted_pairs:
            if parent == pair:
                continue
            par_i, par_j = parent
            # Check if parent contains pair: par_i < p_i < p_j < par_j
            # This automatically ensures pairs don't cross (proper nesting)
            if par_i < p_i and p_j < par_j:
                span = par_j - par_i
                # Find the smallest containing pair (immediate parent)
                if span < best_span:
                    # Check if current best_parent is between this parent and the pair
                    # If so, this parent is not the immediate parent
                    is_immediate = True
                    if best_parent is not None:
                        bp_i, bp_j = best_parent
                        # If best_parent is nested inside this parent, then this parent is not immediate
                        if par_i < bp_i < bp_j < par_j:
                            is_immediate = False
                    
                    if is_immediate:
                        best_parent = parent
                        best_span = span
        
        if best_parent is not None:
            children_of[best_parent].append(pair)
    
    # Step 6: Place pairs recursively using tree structure with appropriate bracket types
    def place_pair_tree(pair):
        """Place a pair and recursively place its children."""
        i, j = pair
        if i > 0 and i <= inferred_length and j > 0 and j <= inferred_length:
            # Get bracket type for this pair
            bracket_type = pair_to_bracket.get(pair, '()')
            open_char, close_char = bracket_chars[bracket_type]
            
            # Place the pair
            structure[i-1] = open_char
            structure[j-1] = close_char
            
            # Place children recursively
            for child in sorted(children_of[pair], key=lambda p: p[0]):
                place_pair_tree(child)
    
    # Start with root pairs (pairs with no parent)
    root_pairs = [pair for pair in sorted_pairs if pair not in 
                  [child for children_list in children_of.values() for child in children_list]]
    
    # Place all root pairs and their descendants
    for root_pair in sorted(root_pairs, key=lambda p: p[0]):
        place_pair_tree(root_pair)
    
    # Step 7: Ensure any remaining pairs are placed (for crossing pairs that couldn't be in tree)
    for i, j in sorted_pairs:
        if 0 < i <= inferred_length and 0 < j <= inferred_length:
            bracket_type = pair_to_bracket.get((i, j), '()')
            open_char, close_char = bracket_chars[bracket_type]
            if structure[i-1] == '.':
                structure[i-1] = open_char
            if structure[j-1] == '.':
                structure[j-1] = close_char
    
    # Step 8: Fill in remaining positions between pairs as dots if not already set
    pair_positions = set()
    for i, j in all_pairs:
        pair_positions.add(i)
        pair_positions.add(j)
    
    for i, j in sorted_pairs:
        # Mark positions between i and j as dots if not part of any pair
        for pos in range(i + 1, j):
            if pos not in pair_positions and 0 < pos <= inferred_length:
                # Check if position is not already a bracket character
                if structure[pos-1] not in '([{)]}':
                    structure[pos-1] = '.'
    
    return ''.join(structure)

def test_dotbracket_from_structure_units():
    """
    Test the dotbracket_from_structure_units function with round-trip tests.
    """
    print("Testing dotbracket_from_structure_units...")
    
    test_cases = [
        ("((..))", "Simple hairpin"),
        ("(((...)))", "Hairpin with 3 unpaired"),
        ("((..[[..))..]]", "Pseudoknot (may not work perfectly)"),
        ("((.(..).))", "Interior loop"),
        ("((..))..((..))", "Multiple hairpins"),
        ("(((...((..))...)))", "Nested structures"),
    ]
    
    for original_structure, description in test_cases:
        print(f"\n--- {description}: {original_structure} ---")
        
        try:
            # Extract structure units
            units = rna_units_from_dotbracket(original_structure, include_stem_runs=False)
            print(f"Extracted {len(units)} structure units")
            
            # Reconstruct structure
            reconstructed = dotbracket_from_structure_units(units, length=len(original_structure))
            print(f"Original:     {original_structure}")
            print(f"Reconstructed: {reconstructed}")
            
            # Check if they match
            if original_structure == reconstructed:
                print("✓ Round-trip successful!")
            else:
                print("⚠ Round-trip mismatch (may be acceptable due to multiple valid representations)")
                
                # Compare base pairs
                orig_pairs = set(parseBracketString_pk(original_structure) or [])
                recon_pairs = set(parseBracketString_pk(reconstructed) or [])
                if orig_pairs == recon_pairs:
                    print("  ✓ Base pairs match (structure is equivalent)")
                else:
                    print(f"  ✗ Base pairs differ: orig={len(orig_pairs)}, recon={len(recon_pairs)}")
                    
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

def test_dotbracket_from_base_pairs():
    """
    Test the dotbracket_from_base_pairs function with round-trip tests.
    """
    print("Testing dotbracket_from_base_pairs...")
    
    test_cases = [
        ("((..))", "Simple hairpin"),
        ("(((...)))", "Hairpin with 3 unpaired"),
        ("((..[[..))..]]", "Pseudoknot"),
        ("((.(..).))", "Interior loop"),
        ("((..))..((..))", "Multiple hairpins"),
        ("(((...((..))...)))", "Nested structures"),
    ]
    
    for original_structure, description in test_cases:
        print(f"\n--- {description}: {original_structure} ---")
        
        try:
            # Extract base pairs from original structure (use universal parsing for pseudoknots)
            base_pairs_0indexed = universal_parse_base_pairs(original_structure)
            if not base_pairs_0indexed:
                print("  ⚠ Could not parse base pairs")
                continue
            
            # Convert to 1-indexed for dotbracket_from_base_pairs
            base_pairs_1indexed = [(i+1, j+1) for i, j in base_pairs_0indexed]
            print(f"Extracted {len(base_pairs_1indexed)} base pairs: {base_pairs_1indexed}")
            
            # Reconstruct structure
            reconstructed = dotbracket_from_base_pairs(base_pairs_1indexed, length=len(original_structure))
            print(f"Original:     {original_structure}")
            print(f"Reconstructed: {reconstructed}")
            
            # Check if they match
            if original_structure == reconstructed:
                print("✓ Round-trip successful!")
            else:
                print("⚠ Round-trip mismatch (may be acceptable due to multiple valid representations)")
                
                # Compare base pairs (use universal parsing for pseudoknots)
                orig_pairs = set(universal_parse_base_pairs(original_structure))
                recon_pairs = set(universal_parse_base_pairs(reconstructed))
                if orig_pairs == recon_pairs:
                    print("  ✓ Base pairs match (structure is equivalent)")
                else:
                    print(f"  ✗ Base pairs differ: orig={len(orig_pairs)}, recon={len(recon_pairs)}")
                    print(f"    Original pairs: {sorted(orig_pairs)}")
                    print(f"    Reconstructed pairs: {sorted(recon_pairs)}")
                    
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Test with explicit base pairs
    print("\n--- Testing with explicit base pairs ---")
    test_pairs = [
        ([(1, 6), (2, 5), (3, 4)], "Simple stem"),
        ([(1, 5), (2, 6)], "Pseudoknot"),
        ([(1, 10), (2, 9), (3, 8), (4, 7)], "Long stem"),
        ([], "No pairs"),
    ]
    
    for pairs, desc in test_pairs:
        print(f"\n{desc}: {pairs}")
        result = dotbracket_from_base_pairs(pairs)
        print(f"  Result: {result}")
        if pairs:
            # Verify round-trip (use universal parsing for pseudoknots)
            parsed = universal_parse_base_pairs(result)
            if parsed:
                parsed_1indexed = set((i+1, j+1) for i, j in parsed)
                original_set = set(pairs)
                if parsed_1indexed == original_set:
                    print("  ✓ Round-trip verified")
                else:
                    print(f"  ⚠ Round-trip mismatch: {parsed_1indexed} vs {original_set}")

def test_pseudoknot_decomposition():
    """
    Test the pseudoknot decomposition functionality with example structures.
    """
    print("Testing pseudoknot decomposition...")
    
    # Test cases
    test_structures = [
        "((..[[..))..]]",  # Simple pseudoknot
        "((..[[..))..]]..((..))",  # Pseudoknot + regular structure
        "((..[[..{{..))..]]..}}",  # Three bracket types
        "((..))",  # No pseudoknot
        "....",  # No structure
    ]
    
    for i, structure in enumerate(test_structures):
        print(f"\n--- Test Case {i+1}: {structure} ---")
        
        # Decompose structure
        substructures = decompose_pseudoknot_structure(structure)
        print(f"Decomposed into {len(substructures)} substructures:")
        
        for j, (substruct, bracket_type) in enumerate(substructures):
            print(f"  {j+1}. {substruct} (bracket type: {bracket_type})")
        
        # Parse pseudoknot structure
        all_pairs, all_units, substructure_info = parse_pseudoknot_structure(structure)
        
        print(f"Total base pairs found: {len(all_pairs)}")
        print(f"Total structure units found: {len(all_units)}")
        
        for info in substructure_info:
            print(f"  {info['bracket_type']}: {info['num_pairs']} pairs, {info['num_units']} units")
            if 'error' in info:
                print(f"    Error: {info['error']}")
        
        print(f"Base pairs: {all_pairs}")
        print(f"Structure units: {[str(unit) for unit in all_units[:3]]}{'...' if len(all_units) > 3 else ''}")

def main():
    # Test universal parsing first
    test_universal_parsing()
    
    print("\n" + "="*50)
    print("Running universal FASTA parsing...")
    
    # Parse the FASTA file
    print("Parsing RF00005_bp.fasta...")
    structures_data = parse_fasta_file('RF00005_bp.fasta')
    print(f"Found {len(structures_data)} RNA structures")
    
    # Extract base pairs using universal parsing
    print("Extracting base pairs using universal parsing...")
    all_base_pairs, structure_base_pairs, valid_structures = universal_extract_base_pairs(structures_data)
    print(f"Found {len(valid_structures)} valid structures")
    print(f"Found {len(all_base_pairs)} unique base pairs")
    
    # Create the matrix
    print("Creating structure matrix...")
    matrix, unique_pairs = create_structure_matrix(all_base_pairs, structure_base_pairs)

    # print(matrix)
    # print(matrix.shape)
    # print(matrix[0].sum())
    
    print(f"Matrix shape: {matrix.shape}")
    print(f"Number of structures: {matrix.shape[0]}")
    print(f"Number of unique base pairs: {matrix.shape[1]}")
    
    # Save results
    np.save('structure_matrix.npy', matrix)
    np.save('unique_base_pairs.npy', np.array(unique_pairs))
    
    # Save structure information
    with open('structure_info.txt', 'w') as f:
        f.write("Structure_ID\tSequence\tDot_Bracket_Structure\n")
        for seq_id, sequence, structure_str in valid_structures:
            f.write(f"{seq_id}\t{sequence}\t{structure_str}\n")
    
    # Print some statistics
    print(f"\nMatrix statistics:")
    print(f"Total entries: {matrix.size}")
    print(f"Non-zero entries: {np.sum(matrix)}")
    print(f"Sparsity: {1 - np.sum(matrix) / matrix.size:.4f}")
    
    # Show some example base pairs
    print(f"\nFirst 10 unique base pairs:")
    for i, pair in enumerate(unique_pairs[:10]):
        print(f"  {i+1}: {pair}")
    
    return matrix, unique_pairs, valid_structures

def detect_pseudoknots(dotbr):
    """
    Detect if a dot-bracket structure contains pseudoknots.
    
    Args:
        dotbr: Dot-bracket structure string
        
    Returns:
        bool: True if pseudoknots are detected, False otherwise
    """
    # Count different bracket types
    bracket_counts = {'(': 0, '[': 0, '{': 0}
    
    for char in dotbr:
        if char in bracket_counts:
            bracket_counts[char] += 1
    
    # Check if more than one bracket type is present
    active_bracket_types = sum(1 for count in bracket_counts.values() if count > 0)
    
    return active_bracket_types > 1

def universal_parse_base_pairs(dotbr):
    """
    Universally parse base pairs from any dot-bracket structure.
    Automatically detects and handles pseudoknots.
    
    Args:
        dotbr: Dot-bracket structure string (may contain pseudoknots)
        
    Returns:
        List of base pairs as (i, j) tuples (0-indexed, consistent with parseBracketString_pk)
    """
    if detect_pseudoknots(dotbr):
        # Use pseudoknot parsing
        all_pairs, _, _ = parse_pseudoknot_structure(dotbr)
        # Convert from 1-indexed to 0-indexed for consistency
        return [(i-1, j-1) for i, j in all_pairs]
    else:
        # Use standard parsing - parseBracketString_pk for consistency
        pairs = parseBracketString_pk(dotbr)
        return pairs if pairs is not None else []

def universal_parse_structure_units(dotbr, include_stem_runs=False, sequence=None):
    """
    Universally parse structure units from any dot-bracket structure.
    Automatically detects and handles pseudoknots.
    
    Stacking pairs are always included.
    
    Args:
        dotbr: Dot-bracket structure string (may contain pseudoknots)
        include_stem_runs: Whether to include stem runs
        sequence: Optional sequence string
        
    Returns:
        List of structure units
    """
    if detect_pseudoknots(dotbr):
        # Use pseudoknot parsing
        _, all_units, _ = parse_pseudoknot_structure(dotbr, 
                                                   include_stem_runs=include_stem_runs,
                                                   sequence=sequence)
        return all_units
    else:
        # Use standard parsing
        return rna_units_from_dotbracket(dotbr, 
                                       include_stem_runs=include_stem_runs,
                                       sequence=sequence)

def universal_extract_base_pairs(structures_data):
    """
    Universally extract base pairs from structures data.
    Automatically handles both pseudoknot-free and pseudoknotted structures.
    
    Args:
        structures_data: List of (seq_id, sequence, structure_str) tuples
        
    Returns:
        Tuple of (all_base_pairs, structure_base_pairs, valid_structures)
    """
    all_base_pairs = set()
    structure_base_pairs = []
    valid_structures = []
    
    for seq_id, sequence, structure_str in structures_data:
        try:
            # Use universal parsing
            base_pairs = universal_parse_base_pairs(structure_str)
            
            if base_pairs is not None and len(base_pairs) > 0:
                # Convert to 1-indexed base pairs
                indexed_pairs = [(i+1, j+1) for i, j in base_pairs]
                structure_base_pairs.append(indexed_pairs)
                all_base_pairs.update(indexed_pairs)
                valid_structures.append((seq_id, sequence, structure_str))
            else:
                print(f"Warning: No valid base pairs found for {seq_id}")
                
        except Exception as e:
            print(f"Warning: Error processing structure for {seq_id}: {e}")
    
    return all_base_pairs, structure_base_pairs, valid_structures

def universal_extract_structure_units(structures_data, include_stem_runs=False):
    """
    Universally extract structure units from structures data.
    Automatically handles both pseudoknot-free and pseudoknotted structures.
    
    Stacking pairs are always included.
    
    Args:
        structures_data: List of (seq_id, sequence, structure_str) tuples
        include_stem_runs: Whether to include stem runs
        
    Returns:
        Tuple of (all_structure_units, structure_units_list, valid_structures)
    """
    all_structure_units = set()
    structure_units_list = []
    valid_structures = []
    
    for seq_id, sequence, structure_str in structures_data:
        try:
            # If sequence is None, generate a dummy sequence of 'N' characters
            # with the same length as the structure string
            if sequence is None:
                sequence = 'N' * len(structure_str)
            
            # Use universal parsing
            units = universal_parse_structure_units(structure_str, 
                                                  include_stem_runs=include_stem_runs,
                                                  sequence=sequence)
            
            # Convert units to hashable format for set operations
            hashable_units = []
            for unit in units:
                if isinstance(unit, tuple):
                    hashable_unit = str(unit)
                else:
                    hashable_unit = str(unit)
                hashable_units.append(hashable_unit)
                all_structure_units.add(hashable_unit)
            
            structure_units_list.append(hashable_units)
            valid_structures.append((seq_id, sequence, structure_str))
            
        except Exception as e:
            print(f"Warning: Error processing structure for {seq_id}: {e}")
            structure_units_list.append([])
    
    return all_structure_units, structure_units_list, valid_structures

def test_universal_parsing():
    """
    Test the universal parsing functions with various structure types.
    """
    print("Testing universal parsing functions...")
    
    test_structures = [
        ("((..))", "Simple hairpin"),
        ("((..[[..))..]]", "Simple pseudoknot"),
        ("((..[[..{{..))..]]..}}", "Complex pseudoknot"),
        ("....", "No structure"),
        ("((..))..((..))", "Multiple hairpins"),
    ]
    
    for structure, description in test_structures:
        print(f"\n--- {description}: {structure} ---")
        
        # Test pseudoknot detection
        has_pseudoknots = detect_pseudoknots(structure)
        print(f"Pseudoknots detected: {has_pseudoknots}")
        
        # Test universal base pair parsing
        base_pairs = universal_parse_base_pairs(structure)
        print(f"Base pairs: {base_pairs}")
        
        # Test universal structure unit parsing
        structure_units = universal_parse_structure_units(structure)
        print(f"Structure units: {len(structure_units)}")
        for i, unit in enumerate(structure_units[:2]):  # Show first 2
            print(f"  {i+1}: {unit}")
        if len(structure_units) > 2:
            print(f"  ... and {len(structure_units) - 2} more")

def universal_load_structure_units_matrix_from_fasta(fasta_file: str, max_structures=None, 
                                                   include_stem_runs=False, use_pre_aggregation=True):
    """
    Universally load structure units matrix from FASTA file.
    Automatically handles both pseudoknot-free and pseudoknotted structures.
    
    Stacking pairs are always included.
    
    Args:
        fasta_file: Path to FASTA file
        max_structures: Maximum number of structures to process
        include_stem_runs: Whether to include stem runs
        use_pre_aggregation: Whether to pre-aggregate identical rows
        
    Returns:
        Tuple of (matrix, unique_units, valid_structures, id_to_row_mapping, 
                 original_matrix_shape, pre_aggregation_shape, row_mapping, duplicates_removed, gapped_mappings)
    """
    # Parse structure file (supports FASTA, text, JSON, CSV formats)
    try:
        structures_data = parse_structure_file(fasta_file)
    except Exception as e:
        # Fallback to old FASTA parser for backward compatibility
        try:
            structures_data = parse_fasta_file(fasta_file)
        except IndexError as idx_err:
            if "list index out of range" in str(idx_err):
                # Handle empty lines in FASTA file
                print(f"⚠️  Warning: Empty lines detected in file. Attempting to clean file...")
                structures_data = parse_fasta_file_robust(fasta_file)
            else:
                raise
        except Exception:
            raise e
    
    if max_structures:
        structures_data = structures_data[:max_structures]
    
    # Use universal extraction
    all_structure_units, structure_units_list, valid_structures = universal_extract_structure_units(
        structures_data, include_stem_runs=include_stem_runs)
    
    # Create gapped mappings for compatibility
    gapped_mappings = []
    for seq_id, sequence, structure_str in valid_structures:
        # Check if sequence has gaps
        has_gaps = '-' in sequence
        
        if has_gaps:
            # Parse gapped structure
            gapped_units, ungapped_units, gapped_to_ungapped, ungapped_to_gapped, ungapped_sequence = parse_gapped_structure(sequence, structure_str)
            
            # Store mapping for CSV generation
            gapped_mappings.append({
                'gapped_to_ungapped': gapped_to_ungapped,
                'ungapped_to_gapped': ungapped_to_gapped,
                'gapped_units': gapped_units,
                'ungapped_units': ungapped_units,
                'gapped_sequence': sequence,
                'ungapped_sequence': ungapped_sequence,
                'gapped_structure': structure_str
            })
        else:
            # For ungapped sequences, mapping is identity
            gapped_mappings.append({
                'gapped_to_ungapped': {i: i for i in range(1, len(sequence) + 1)},
                'ungapped_to_gapped': {i: i for i in range(1, len(sequence) + 1)},
                'gapped_units': [],
                'ungapped_units': [],
                'gapped_sequence': sequence,
                'ungapped_sequence': sequence,
                'gapped_structure': structure_str
            })
    
    # Create matrix using universal extraction results
    matrix, unique_units, id_to_row_mapping, original_matrix_shape, pre_aggregation_shape, row_mapping, duplicates_removed = create_structure_units_matrix(
        all_structure_units, structure_units_list, valid_structures, use_pre_aggregation)
    
    return matrix, unique_units, valid_structures, id_to_row_mapping, original_matrix_shape, pre_aggregation_shape, row_mapping, duplicates_removed, gapped_mappings


def universal_load_structure_units_matrix_original_from_fasta(fasta_file: str, max_structures=None, 
                                                               include_stem_runs=False):
    """
    Universally load structure units matrix from FASTA file, always returning the original matrix
    (without pre-aggregation, even if there are duplicate structures).
    Automatically handles both pseudoknot-free and pseudoknotted structures.
    
    Stacking pairs are always included.
    
    Args:
        fasta_file: Path to FASTA file
        max_structures: Maximum number of structures to process
        include_stem_runs: Whether to include stem runs
        
    Returns:
        Tuple of (original_matrix, unique_units, valid_structures, id_to_row_mapping, 
                 original_matrix_shape, row_mapping, gapped_mappings)
        - original_matrix: Original binary matrix (one row per structure, no aggregation)
        - unique_units: List of unique structure units (column mapping)
        - valid_structures: List of (seq_id, sequence, structure_str) tuples
        - id_to_row_mapping: Dictionary mapping sequence ID to row index in original matrix
        - original_matrix_shape: Shape of the original matrix
        - row_mapping: Dictionary mapping row index to list containing itself (for compatibility)
        - gapped_mappings: List of gapped mapping dictionaries for each structure
    """
    # Parse structure file (supports FASTA, text, JSON, CSV formats)
    try:
        structures_data = parse_structure_file(fasta_file)
    except Exception as e:
        # Fallback to old FASTA parser for backward compatibility
        try:
            structures_data = parse_fasta_file(fasta_file)
        except IndexError as idx_err:
            if "list index out of range" in str(idx_err):
                # Handle empty lines in FASTA file
                print(f"⚠️  Warning: Empty lines detected in file. Attempting to clean file...")
                structures_data = parse_fasta_file_robust(fasta_file)
            else:
                raise
        except Exception:
            raise e
    
    if max_structures:
        structures_data = structures_data[:max_structures]
    
    # Use universal extraction
    all_structure_units, structure_units_list, valid_structures = universal_extract_structure_units(
        structures_data, include_stem_runs=include_stem_runs)
    
    # Create gapped mappings for compatibility
    gapped_mappings = []
    for seq_id, sequence, structure_str in valid_structures:
        # Check if sequence has gaps
        has_gaps = '-' in sequence
        
        if has_gaps:
            # Parse gapped structure
            gapped_units, ungapped_units, gapped_to_ungapped, ungapped_to_gapped, ungapped_sequence = parse_gapped_structure(sequence, structure_str)
            
            # Store mapping for CSV generation
            gapped_mappings.append({
                'gapped_to_ungapped': gapped_to_ungapped,
                'ungapped_to_gapped': ungapped_to_gapped,
                'gapped_units': gapped_units,
                'ungapped_units': ungapped_units,
                'gapped_sequence': sequence,
                'ungapped_sequence': ungapped_sequence,
                'gapped_structure': structure_str
            })
        else:
            # For ungapped sequences, mapping is identity
            gapped_mappings.append({
                'gapped_to_ungapped': {i: i for i in range(1, len(sequence) + 1)},
                'ungapped_to_gapped': {i: i for i in range(1, len(sequence) + 1)},
                'gapped_units': [],
                'ungapped_units': [],
                'gapped_sequence': sequence,
                'ungapped_sequence': sequence,
                'gapped_structure': structure_str
            })
    
    # Create matrix using universal extraction results (with use_pre_aggregation=False to get original)
    original_matrix, unique_units, id_to_row_mapping, original_matrix_shape, _, row_mapping, _ = create_structure_units_matrix(
        all_structure_units, structure_units_list, valid_structures, use_pre_aggregation=False)
    
    return original_matrix, unique_units, valid_structures, id_to_row_mapping, original_matrix_shape, row_mapping, gapped_mappings

def universal_load_structure_units_matrix_with_weights_from_fasta(fasta_file: str, max_structures=None, 
                                                                include_stem_runs=False, use_pre_aggregation=True):
    """
    Universally load structure units matrix with weights from FASTA file.
    Automatically handles both pseudoknot-free and pseudoknotted structures.
    
    Stacking pairs are always included.
    
    Args:
        fasta_file: Path to FASTA file
        max_structures: Maximum number of structures to process
        include_stem_runs: Whether to include stem runs
        use_pre_aggregation: Whether to pre-aggregate identical rows
        
    Returns:
        Tuple of (matrix, weights, unique_units, valid_structures, id_to_row_mapping, 
                 original_matrix_shape, pre_aggregation_shape, row_mapping, duplicates_removed, gapped_mappings)
    """
    # Parse structure file (supports FASTA, text, JSON, CSV formats)
    try:
        structures_data = parse_structure_file(fasta_file)
    except Exception as e:
        # Fallback to old FASTA parser for backward compatibility
        try:
            structures_data = parse_fasta_file(fasta_file)
        except IndexError as idx_err:
            if "list index out of range" in str(idx_err):
                # Handle empty lines in FASTA file
                print(f"⚠️  Warning: Empty lines detected in file. Attempting to clean file...")
                structures_data = parse_fasta_file_robust(fasta_file)
            else:
                raise
        except Exception:
            raise e
    
    if max_structures:
        structures_data = structures_data[:max_structures]
    
    # Use universal extraction
    all_structure_units, structure_units_list, valid_structures = universal_extract_structure_units(
        structures_data, include_stem_runs=include_stem_runs)
    
    # Create gapped mappings for compatibility
    gapped_mappings = []
    for seq_id, sequence, structure_str in valid_structures:
        # Check if sequence has gaps
        has_gaps = '-' in sequence
        
        if has_gaps:
            # Parse gapped structure
            gapped_units, ungapped_units, gapped_to_ungapped, ungapped_to_gapped, ungapped_sequence = parse_gapped_structure(sequence, structure_str)
            
            # Store mapping for CSV generation
            gapped_mappings.append({
                'gapped_to_ungapped': gapped_to_ungapped,
                'ungapped_to_gapped': ungapped_to_gapped,
                'gapped_units': gapped_units,
                'ungapped_units': ungapped_units,
                'gapped_sequence': sequence,
                'ungapped_sequence': ungapped_sequence,
                'gapped_structure': structure_str
            })
        else:
            # For ungapped sequences, mapping is identity
            gapped_mappings.append({
                'gapped_to_ungapped': {i: i for i in range(1, len(sequence) + 1)},
                'ungapped_to_gapped': {i: i for i in range(1, len(sequence) + 1)},
                'gapped_units': [],
                'ungapped_units': [],
                'gapped_sequence': sequence,
                'ungapped_sequence': sequence,
                'gapped_structure': structure_str
            })
    
    # Create matrix with weights using universal extraction results
    matrix, weights, unique_units, id_to_row_mapping, original_matrix_shape, pre_aggregation_shape, row_mapping, duplicates_removed = create_structure_units_matrix_with_weights(
        all_structure_units, structure_units_list, valid_structures, use_pre_aggregation)
    
    return matrix, weights, unique_units, valid_structures, id_to_row_mapping, original_matrix_shape, pre_aggregation_shape, row_mapping, duplicates_removed, gapped_mappings

def universal_load_base_pair_matrix_from_fasta(fasta_file: str, max_structures=None, use_pre_aggregation=True):
    """
    Universally load base pair matrix from FASTA file.
    Automatically handles both pseudoknot-free and pseudoknotted structures.
    
    Args:
        fasta_file: Path to FASTA file
        max_structures: Maximum number of structures to process
        use_pre_aggregation: Whether to pre-aggregate identical rows
        
    Returns:
        Tuple of (matrix, unique_pairs, valid_structures, id_to_row_mapping, 
                 original_matrix_shape, pre_aggregation_shape, row_mapping, duplicates_removed, gapped_mappings)
    """
    # Parse structure file (supports FASTA, text, JSON, CSV formats)
    try:
        structures_data = parse_structure_file(fasta_file)
    except Exception as e:
        # Fallback to old FASTA parser for backward compatibility
        try:
            structures_data = parse_fasta_file(fasta_file)
        except IndexError as idx_err:
            if "list index out of range" in str(idx_err):
                # Handle empty lines in FASTA file
                print(f"⚠️  Warning: Empty lines detected in file. Attempting to clean file...")
                structures_data = parse_fasta_file_robust(fasta_file)
            else:
                raise
        except Exception:
            raise e
    
    if max_structures:
        structures_data = structures_data[:max_structures]
    
    # Use universal extraction
    all_base_pairs, structure_base_pairs, valid_structures = universal_extract_base_pairs(structures_data)
    
    # Create gapped mappings for compatibility
    gapped_mappings = []
    for seq_id, sequence, structure_str in valid_structures:
        # Check if sequence has gaps
        has_gaps = '-' in sequence
        
        if has_gaps:
            # Create nucleotide mapping
            gapped_to_ungapped, ungapped_to_gapped, ungapped_sequence = create_nucleotide_mapping(sequence)
            
            # Store mapping for CSV generation
            gapped_mappings.append({
                'gapped_to_ungapped': gapped_to_ungapped,
                'ungapped_to_gapped': ungapped_to_gapped,
                'gapped_units': [],  # Not applicable for base pairs mode
                'ungapped_units': [],  # Not applicable for base pairs mode
                'gapped_sequence': sequence,
                'ungapped_sequence': ungapped_sequence,
                'gapped_structure': structure_str
            })
        else:
            # For ungapped sequences, mapping is identity
            gapped_mappings.append({
                'gapped_to_ungapped': {i: i for i in range(1, len(sequence) + 1)},
                'ungapped_to_gapped': {i: i for i in range(1, len(sequence) + 1)},
                'gapped_units': [],
                'ungapped_units': [],
                'gapped_sequence': sequence,
                'ungapped_sequence': sequence,
                'gapped_structure': structure_str
            })
    
    # Create matrix using universal extraction results
    matrix, unique_pairs, id_to_row_mapping, original_matrix_shape, pre_aggregation_shape, row_mapping, duplicates_removed = create_structure_matrix(
        all_base_pairs, structure_base_pairs, valid_structures, use_pre_aggregation)
    
    return matrix, unique_pairs, valid_structures, id_to_row_mapping, original_matrix_shape, pre_aggregation_shape, row_mapping, duplicates_removed, gapped_mappings

if __name__ == "__main__":
    matrix, unique_pairs, valid_structures = main() 