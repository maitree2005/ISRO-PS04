from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
import networkx as nx

from graph import centrality

router = APIRouter()


def _build_sample_graph():
    G = nx.Graph()
    G.add_node('n1', pos=(77.5900, 12.9716), name='Node 1')
    G.add_node('n2', pos=(77.5910, 12.9720), name='Node 2')
    G.add_node('n3', pos=(77.5920, 12.9710), name='Node 3')
    G.add_edge('n1', 'n2', weight=100)
    G.add_edge('n2', 'n3', weight=120)
    G.add_edge('n1', 'n3', weight=250)
    return G


@router.get("/criticality")
def get_criticality(city: str = Query('bengaluru'), top_n: int = Query(10)):
    """Return top-N critical nodes (betweenness) for the city graph (sample implementation)."""
    # In production: load precomputed graph for `city` from storage; here we build a sample graph
    G = _build_sample_graph()
    res = centrality.compute_criticality(G)
    ranked = res['ranked_nodes'] if isinstance(res, dict) and 'ranked_nodes' in res else []

    top = []
    for rank, (node, score) in enumerate(ranked[:top_n], start=1):
        pos = G.nodes[node].get('pos', (None, None))
        top.append({
            'node_id': str(node),
            'lat': pos[1],
            'lon': pos[0],
            'betweenness_score': float(score),
            'criticality_rank': rank,
            'label': G.nodes[node].get('name')
        })

    return JSONResponse({
        'city': city,
        'total_nodes': G.number_of_nodes(),
        'top_bottlenecks': top
    })
