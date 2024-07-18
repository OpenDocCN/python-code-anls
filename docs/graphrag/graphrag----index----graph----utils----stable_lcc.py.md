# `.\graphrag\graphrag\index\graph\utils\stable_lcc.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""A module for producing a stable largest connected component, i.e. same input graph == same output lcc."""

from typing import Any, cast

import networkx as nx
from graspologic.utils import largest_connected_component

from .normalize_node_names import normalize_node_names


def stable_largest_connected_component(graph: nx.Graph) -> nx.Graph:
    """Return the largest connected component of the graph, with nodes and edges sorted in a stable way."""
    # Make a copy of the input graph to avoid modifying the original
    graph = graph.copy()
    # Extract the largest connected component as a subgraph
    graph = cast(nx.Graph, largest_connected_component(graph))
    # Normalize node names in the graph to ensure consistency
    graph = normalize_node_names(graph)
    # Return the stabilized version of the graph with nodes and edges sorted in a stable way
    return _stabilize_graph(graph)


def _stabilize_graph(graph: nx.Graph) -> nx.Graph:
    """Ensure an undirected graph with the same relationships will always be read the same way."""
    # Create a new empty graph of appropriate type (directed or undirected)
    fixed_graph = nx.DiGraph() if graph.is_directed() else nx.Graph()

    # Retrieve nodes from the input graph along with their data attributes
    sorted_nodes = graph.nodes(data=True)
    # Sort nodes based on their identifiers to maintain a stable node order
    sorted_nodes = sorted(sorted_nodes, key=lambda x: x[0])

    # Add nodes to the fixed graph maintaining the sorted order
    fixed_graph.add_nodes_from(sorted_nodes)

    # Retrieve edges from the input graph along with their data attributes
    edges = list(graph.edges(data=True))

    # If the graph is undirected, ensure edges are consistently sorted by node identifiers
    if not graph.is_directed():

        def _sort_source_target(edge):
            # Ensure consistent ordering of source and target nodes in edges
            source, target, edge_data = edge
            if source > target:
                temp = source
                source = target
                target = temp
            return source, target, edge_data

        # Apply the sorting function to all edges
        edges = [_sort_source_target(edge) for edge in edges]

    # Function to generate a unique key for each edge based on its source and target
    def _get_edge_key(source: Any, target: Any) -> str:
        return f"{source} -> {target}"

    # Sort edges based on the generated key to ensure stable edge order
    edges = sorted(edges, key=lambda x: _get_edge_key(x[0], x[1]))

    # Add sorted edges to the fixed graph
    fixed_graph.add_edges_from(edges)

    # Return the fixed graph with nodes and edges sorted in a stable way
    return fixed_graph
```