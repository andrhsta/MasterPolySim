"""
-------------------------------------------------------------------------------------
Loading Genomic Compartment Sequences for Polymer Simulation Input (40 kb Resolution)
-------------------------------------------------------------------------------------
"""

# Import modules
import os
import pandas as pd
import numpy as np



# -------------------------------------
# Loading 40 kb Resolution A/B Sequence 
# -------------------------------------

def load_full_resolution_csv(filename="full_resolution_IMR90_40kb_two_categories.csv", monomer_size=40000):
    """
    Load a CSV file containing genomic compartments and convert it into a sequence array at a given monomer resolution.

    Parameters:
    - filename: str
        Name of the CSV file. Must be located in the same folder as this script.
    - monomer_size: int
        Size of one monomer in base pairs. Default is 40,000 (40 kb).

    Returns:
    - sequence: np.ndarray, array of integers representing compartments (0 for A, 1 for B).
    """
    file_path = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Sequence CSV not found: {file_path}")

    # Read the file line by line
    with open(file_path, 'r') as f:
        lines = [line.strip().strip('"') for line in f]

    # Split by tabs
    split_lines = [line.split('\t') for line in lines]

    # First line is header
    header = split_lines[0]
    data = split_lines[1:]

    # Create DataFrame
    df = pd.DataFrame(data, columns=header)

    # Convert positions to integers
    df['pos_start'] = df['pos_start'].astype(float).astype(int)
    df['pos_end'] = df['pos_end'].astype(float).astype(int)

    # Determine length of sequence in monomers
    max_pos = df['pos_end'].max()
    n_monomers = (max_pos + monomer_size - 1) // monomer_size

    # Initialize sequence array
    sequence = np.zeros(n_monomers, dtype=int)
    mapping = {'A': 0, 'B': 1}

    # Fill sequence based on compartment intervals
    for _, row in df.iterrows():
        start_idx = (row['pos_start'] - 1) // monomer_size
        end_idx = (row['pos_end'] - 1) // monomer_size
        sequence[start_idx:end_idx + 1] = mapping[row['compartment']]

    return sequence