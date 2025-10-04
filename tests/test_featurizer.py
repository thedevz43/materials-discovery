from pymatgen.core import Structure
from src.featurizers import structure_to_graph

def test_structure_to_graph():
    # Dummy cubic structure
    structure = Structure.from_spacegroup("Pm-3m", ["Na", "Cl"], [[0,0,0],[0.5,0.5,0.5]], 5.64)
    data = structure_to_graph(structure)
    assert hasattr(data, "x")
    assert hasattr(data, "edge_index")
    assert hasattr(data, "edge_attr")
    assert data.x.shape[1] == 3
