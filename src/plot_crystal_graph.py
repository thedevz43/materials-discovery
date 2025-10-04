import os
from pymatgen.core import Structure
import networkx as nx
import matplotlib.pyplot as plt

def plot_crystal_graph(file_path, cutoff=4.5):
    """
    Visualize a crystal structure as a graph.

    Args:
        file_path (str): Path to the CIF or POSCAR file.
        cutoff (float): Cutoff radius for determining neighbors (in Å).
    """
    # Load the structure using pymatgen
    structure = Structure.from_file(file_path)

    # Create a graph
    graph = nx.Graph()

    # Add nodes (atoms) with atomic number as labels
    for i, site in enumerate(structure):
        graph.add_node(i, element=site.species_string, atomic_number=site.specie.number)

    # Add edges (bonds) for neighbors within the cutoff radius
    for i, site in enumerate(structure):
        neighbors = structure.get_neighbors(site, cutoff)
        for neighbor, dist in neighbors:
            j = structure.index(neighbor)
            if not graph.has_edge(i, j):  # Avoid duplicate edges
                graph.add_edge(i, j, distance=dist)

    # Plot the graph
    pos = nx.spring_layout(graph)  # Use spring layout for visualization
    node_colors = [data["atomic_number"] for _, data in graph.nodes(data=True)]
    labels = nx.get_node_attributes(graph, "element")
    edge_labels = nx.get_edge_attributes(graph, "distance")

    plt.figure(figsize=(10, 8))
    nx.draw(
        graph,
        pos,
        with_labels=True,
        labels=labels,
        node_color=node_colors,
        cmap=plt.cm.viridis,
        node_size=500,
        font_size=10,
        font_color="white",
    )
    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels={k: f"{v:.2f} Å" for k, v in edge_labels.items()},
        font_size=8,
    )
    plt.title(f"Crystal Graph: {os.path.basename(file_path)}")
    plt.show()