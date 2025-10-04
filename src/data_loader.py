import os
import json
from typing import List, Dict, Optional
from mp_api.client import MPRester
from pymatgen.core.structure import Structure
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def fetch_materials_data(api_key: str, n_samples: int = 100) -> List[Dict]:
    """
    Fetches materials data from the Materials Project API using mp-api.

    Args:
        api_key (str): Your Materials Project API key.
        n_samples (int): Number of materials to fetch.

    Returns:
        List[Dict]: List of material dicts with structure, band_gap, formation_energy_per_atom, material_id.
    """
    with MPRester(api_key) as mpr:
        entries = mpr.materials.search(
            num_chunks=1,
            chunk_size=n_samples,
            fields=["material_id", "structure", "nsites", "elements", "composition", "formula_pretty", "chemsys", "volume", "density"]
        )
        # Convert to list of dicts
        entries = [entry.dict() for entry in entries]
    os.makedirs(DATA_DIR, exist_ok=True)
    out_path = os.path.join(DATA_DIR, "materials.json")
    with open(out_path, "w") as f:
        json.dump(entries, f, default=str)
    return entries

def load_materials_json(path: Optional[str] = None) -> List[Dict]:
    """
    Loads materials data from a JSON file.

    Args:
        path (str): Path to JSON file.

    Returns:
        List[Dict]: List of material dicts.
    """
    if path is None:
        path = os.path.join(DATA_DIR, "materials.json")
    with open(path, "r") as f:
        return json.load(f)

def load_materials_data(path: str) -> pd.DataFrame:
    """
    Load materials data from JSON/CSV and preprocess NaN values.

    Args:
        path (str): Path to the data file.

    Returns:
        pd.DataFrame: Preprocessed materials data.
    """
    if path.endswith(".json"):
        data = pd.read_json(path)
    elif path.endswith(".csv"):
        data = pd.read_csv(path)
    else:
        raise ValueError("Unsupported file format. Use JSON or CSV.")

    # Drop rows where all targets are NaN
    target_columns = ["volume", "band_gap", "formation_energy"]
    data = data.dropna(subset=target_columns, how="all")

    # Normalize non-NaN values
    for col in target_columns:
        if col in data:
            data[col] = (data[col] - data[col].mean()) / data[col].std()

    return data

if __name__ == "__main__":
    # Insert your Materials Project API here
    API_KEY = "j38vnZ06F8XB0CMIIGtXZNTLRIegTSTD"
    fetch_materials_data(API_KEY, n_samples=100)
