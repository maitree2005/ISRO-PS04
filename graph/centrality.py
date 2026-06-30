"""
Module 5 - Criticality Analysis

Computes:
- Betweenness Centrality
- Bridge Edge Detection
- Criticality Score
- Heatmap Score
- Resilience Index
"""

from typing import List, Dict, Any

import networkx as nx
import numpy as np


# -----------------------------------------------------
# Criticality Analysis
# -----------------------------------------------------

def compute_criticality(G: nx.Graph) -> Dict[str, Any]:
    """
    Compute criticality of every node.

    Returns:
        {
            graph,
            ranked_nodes,
            bridge_edges
        }
    """

    if G.number_of_nodes() == 0:
        return {
            "graph": G,
            "ranked_nodes": [],
            "bridge_edges": []
        }

    # -----------------------------------------
    # Betweenness Centrality
    # -----------------------------------------

    bc = nx.betweenness_centrality(
        G,
        normalized=True,
        weight="weight"
    )

    nx.set_node_attributes(
        G,
        bc,
        "betweenness"
    )

    # -----------------------------------------
    # Normalize Scores (0 → 1)
    # -----------------------------------------

    values = np.array(list(bc.values()))

    min_score = values.min()
    max_score = values.max()

    for node, score in bc.items():

        if max_score == min_score:
            normalized = 0.0
        else:
            normalized = (
                score - min_score
            ) / (
                max_score - min_score
            )

        G.nodes[node]["criticality"] = float(normalized)
        G.nodes[node]["heatmap"] = float(normalized)

    # -----------------------------------------
    # Bridge Detection
    # -----------------------------------------

    bridges = list(nx.bridges(G))

    for u, v in bridges:

        G[u][v]["is_bridge"] = True

        # Increase heat for bridges
        edge_score = max(
            G.nodes[u]["criticality"],
            G.nodes[v]["criticality"]
        )

        G[u][v]["criticality"] = edge_score
        G[u][v]["heatmap"] = edge_score

    # -----------------------------------------
    # Normal Edges
    # -----------------------------------------

    for u, v in G.edges():

        if "heatmap" not in G[u][v]:

            score = (
                G.nodes[u]["criticality"]
                +
                G.nodes[v]["criticality"]
            ) / 2

            G[u][v]["criticality"] = score
            G[u][v]["heatmap"] = score
            G[u][v]["is_bridge"] = False

    # -----------------------------------------
    # Ranking
    # -----------------------------------------

    ranked = sorted(
        bc.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return {

        "graph": G,

        "ranked_nodes": ranked,

        "bridge_edges": bridges

    }


# -----------------------------------------------------
# Resilience Index
# -----------------------------------------------------

def compute_resilience_index(
    G: nx.Graph,
    nodes_to_remove: List
) -> List[Dict[str, Any]]:

    if G.number_of_nodes() == 0:
        return []

    largest_component = G.subgraph(
        max(
            nx.connected_components(G),
            key=len
        )
    ).copy()

    try:

        baseline = nx.average_shortest_path_length(
            largest_component,
            weight="weight"
        )

    except Exception:

        baseline = float("inf")

    results = []

    for node in nodes_to_remove:

        G_copy = G.copy()

        if node not in G_copy:

            results.append({

                "node": node,

                "resilience_index": None

            })

            continue

        G_copy.remove_node(node)

        try:

            largest = G_copy.subgraph(
                max(
                    nx.connected_components(G_copy),
                    key=len
                )
            ).copy()

            if largest.number_of_nodes() < 2:

                perturbed = float("inf")

                resilience = 0.0

            else:

                perturbed = nx.average_shortest_path_length(
                    largest,
                    weight="weight"
                )

                resilience = (

                    baseline / perturbed

                    if perturbed != 0

                    else 0

                )

        except Exception:

            perturbed = float("inf")

            resilience = 0.0

        results.append({

            "node": node,

            "resilience_index": float(resilience),

            "baseline_avg_path_length": float(baseline),

            "perturbed_avg_path_length": float(perturbed)

        })

    return results