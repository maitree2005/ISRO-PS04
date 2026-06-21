import networkx as nx
from typing import List, Dict, Any


def compute_criticality(G: nx.Graph) -> Dict:
    bc = nx.betweenness_centrality(G, normalized=True, weight='weight')
    nx.set_node_attributes(G, bc, 'betweenness')
    bridges = list(nx.bridges(G))
    for u, v in bridges:
        G[u][v]['is_bridge'] = True
    ranked = sorted(bc.items(), key=lambda x: x[1], reverse=True)
    return {"graph": G, "ranked_nodes": ranked}


def compute_resilience_index(G: nx.Graph, nodes_to_remove: List) -> List[Dict[str, Any]]:
    if G.number_of_nodes() == 0:
        return []
    lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    try:
        L_base = nx.average_shortest_path_length(lcc, weight='weight')
    except Exception:
        L_base = float('inf')

    results = []
    G_test = G.copy()
    for node in nodes_to_remove:
        if node not in G_test:
            results.append({"node": node, "resilience_index": None})
            continue
        G_test.remove_node(node)
        try:
            lcc_test = G_test.subgraph(max(nx.connected_components(G_test), key=len)).copy()
            if lcc_test.number_of_nodes() < 2:
                R = 0.0
                L_perturbed = float('inf')
            else:
                L_perturbed = nx.average_shortest_path_length(lcc_test, weight='weight')
                R = L_base / L_perturbed if L_perturbed!=0 else 0.0
        except Exception:
            R = 0.0
            L_perturbed = float('inf')
        results.append({
            "node": node,
            "resilience_index": R,
            "baseline_avg_path_length": L_base,
            "perturbed_avg_path_length": L_perturbed
        })
    return results
