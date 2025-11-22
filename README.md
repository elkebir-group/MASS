# Maximum Agreement Secondary Structures (MASS)

MASS is a tool that summarizes RNA structure space. MASS accepts inputs in different formats as long as they provide a list/set of structures in dot-bracket format. As a result, MASS returns how structures will be clustered, and the features selected that produce such clusters of structures.

## Installation

```bash
git clone https://github.com/elkebir-group/MASS.git
cd MASS

# Create a new conda environment
conda create -n mass python=3.9 -y
conda activate mass

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:**
- Python 3.7+
- numpy>=1.19.0
- pandas>=1.3.0
- gurobipy>=9.0.0 (requires Gurobi license)
- psutil>=5.8.0

### Gurobi License Configuration

MASS uses Gurobi for the ILP algorithm, which requires a license. You can obtain a free academic license from [Gurobi's website](https://www.gurobi.com/academia/academic-program-and-licenses/).

**Option 1: Web License Service (WLS) - Recommended**

If you have a WLS license, edit `src/gurobi_config.py` and update the `GUROBI_WLS_CONFIG` dictionary with your credentials:

```python
GUROBI_WLS_CONFIG = {
    "WLSACCESSID": "your-access-id",
    "WLSSECRET": "your-secret-key",
    "LICENSEID": your_license_id,
}
USE_LOCAL_LICENSE = False
```

**Option 2: Local License File**

If you have a local Gurobi license file, set `USE_LOCAL_LICENSE = True` in `src/gurobi_config.py`:

```python
USE_LOCAL_LICENSE = True
```

The license file should be located at `~/gurobi.lic` or set via the `GRB_LICENSE_FILE` environment variable.

**Note:** The MSTP and MSTP-BEAM algorithms do not require Gurobi and can be used without a license.


## Usage

### Basic Command

```bash
python src/MASS.py --input <input_file> --tau <tau_value> --output <output_file.csv>
```

or with a range of tau values:

```bash
python src/MASS.py --input <input_file> --tau-range <min_tau> <max_tau> --output <output_file.csv>
```

or with a specific algorithm 

```bash
python src/MASS.py --input <input_file> --algorithm <algorithm> --tau-range <min_tau> <max_tau> --output <output_file.csv>
```

### Input Formats

MASS supports multiple input formats:

1. **FASTA format**: Each entry contains a sequence ID, sequence, and structure
   ```
   > ID1
   SEQUENCE
   STRUCTURE
   ```

2. **Text format**: Each entry contains an ID and structure (one per line)
   ```
   > ID1
   STRUCTURE
   ```

3. **JSON format**: Dictionary mapping IDs to structures
   ```json
   {
     "ID1": "STRUCTURE1",
     "ID2": "STRUCTURE2"
   }
   ```

4. **CSV format**: CSV file with `id` and `structure` columns
   ```csv
   id,structure
   ID1,STRUCTURE1
   ID2,STRUCTURE2
   ```

### Examples

```bash
# Single structure file (FASTA), single tau, all three algorithms (MASS-ILP, MASS-MSTP, MASS-BEAM)
python src/MASS.py --input data/example_input.fasta --tau 3 --output results.csv

# Single structure file (TXT), tau range
python src/MASS.py --input data/example_input.txt --tau-range 2 5 --output results.csv

# Directory of structure files, single tau
python src/MASS.py --input-dir /path/to/structure/files --tau 3 --output results.csv

# With debug output, time limit
python src/MASS.py --input data/example_input.fasta --tau-range 2 6 --output results.csv --debug --time-limit 300

# Run MSTP-BEAM algorithm with beam width 10
python src/MASS.py --input data/example_input.fasta --tau-range 2 6 --output results.csv --algorithm mstp_beam --beam-value 10
```


### Command-Line Arguments

#### Required Arguments

- `--input <file>` or `--input-dir <directory>`: Path to input file or directory containing structure files
- `--tau <value>` or `--tau-range <min> <max>`: Single tau value or range of tau values (inclusive)
- `--output <file>` or `-o <file>`: Output CSV file path

#### Algorithm Selection

- `--algorithm {ilp,mstp,mstp_beam}`: Algorithm to run (default: `mstp`)
  - `ilp`: Integer Linear Programming solver
  - `mstp`: Max-Subset τ-Partitioning algorithm
  - `mstp_beam`: MSTP with beam search

#### Configuration Options

- `--time-limit <seconds>`: Time limit for ILP solver in seconds (default: 7200)
- `--max-structures <n>`: Maximum number of structures to process per file
- `--debug`: Enable debug output
- `--detailed`: Include detailed columns in output (column mappings, gapped sequences, pre-aggregation metadata, etc.)
- `--track-memory`: Enable memory tracking (default: disabled)
- `--log <file>`: Log file path. If not specified, the log file is automatically created in the same directory as the output file with the same basename and a `.log` extension (e.g., if output is `results.csv`, log will be `results.log`)

#### Matrix Type Options

- `--use-base-pairs`: Use base pairs matrix instead of structure units matrix (default: structure units)

#### MSTP-BEAM Options

- `--beam-value <n> [<n> ...]`: Beam width(s) for MSTP-BEAM algorithm (default: 5). Can specify multiple values, each creates a separate row with algorithm name `MSTP-BEAM-{beam_value}`

#### Tau Validation Options

- `--allow-invalid-tau`: Allow tau values greater than the number of unique structures (default: skip invalid tau values)

## Output

The output is a CSV file containing the following columns:

### Standard Columns

- `tau`: The tau value (maximum number of clusters) used for this run
- `algorithm`: Algorithm used (`ILP`, `MSTP`, or `MSTP-BEAM-{beam_value}`)
- `timestamp`: Timestamp when the analysis was performed
- `runtime`: Runtime in seconds
- `status`: Status of the algorithm run (`SUCCESS`, `PARTIAL_SUCCESS`, `FAILED`, `ERROR`, `SKIPPED_MEMORY`, `SKIPPED_TIMEOUT`)
- `selected_k`: Number of selected features (columns)
- `num_clusters`: Number of clusters created
- `objective_value`: Objective value (number of selected columns)
- `selected_units`: List of selected structure units (features) that define the clusters
- `cluster_map`: Dictionary mapping structure IDs to cluster assignments
- `feature_coverage_percent`: Percentage of total features selected
- `feature_coverage_percent_weighted`: Weighted percentage of total features selected (if weights available)
- `total_features`: Total number of features in the input matrix
- `sequences`: Dictionary mapping structure IDs to sequences (if available in input)
- `structures`: Dictionary mapping structure IDs to structures (if available in input)

## License

BSD 3-Clause License (see LICENSE file for details)
