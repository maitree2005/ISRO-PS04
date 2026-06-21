from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import networkx as nx

from graph import centrality

router = APIRouter()


class SimRequest(BaseModel):
    node_id: str
    city: str = 'bengaluru'


def _build_sample_graph():
    G = nx.Graph()
    G.add_node('n1', pos=(77.5900, 12.9716), name='Node 1')
    G.add_node('n2', pos=(77.5910, 12.9720), name='Node 2')
    G.add_node('n3', pos=(77.5920, 12.9710), name='Node 3')
    G.add_edge('n1', 'n2', weight=100)
    G.add_edge('n2', 'n3', weight=120)
    G.add_edge('n1', 'n3', weight=250)
    return G


@router.post("/simulate")
def simulate_removal(req: SimRequest):
    """Simulate node removal impact and return resilience metrics (sample implementation)."""
    G = _build_sample_graph()
    if req.node_id not in G:
        raise HTTPException(status_code=404, detail=f"Node {req.node_id} not found")

    results = centrality.compute_resilience_index(G, [req.node_id])
    # Augment response with baseline component sizes
    lcc_before = max(nx.connected_components(G), key=len)
    res = results[0] if results else {}
    response = {
        'node_removed': req.node_id,
        'baseline_lcc_size': len(lcc_before),
        **res
    }
    return JSONResponse(response)
