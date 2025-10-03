import os
import json
from typing import List, Dict, Optional
from mp_api.client import MPRester
from pymatgen.core.structure import Structure

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

if __name__ == "__main__":
    # Insert your Materials Project API here
    API_KEY = "j38vnZ06F8XB0CMIIGtXZNTLRIegTSTD"
    fetch_materials_data(API_KEY, n_samples=100)
