from fastapi import APIRouter
from fastapi.responses import JSONResponse
import networkx as nx

from graph.export_geojson import graph_to_geojson
from graph import centrality

router = APIRouter()


@router.get("/graph")
def get_sample_graph():
    """Return a small sample graph as GeoJSON for dashboard development."""
    G = nx.Graph()
    # Sample nodes (lon, lat) near Bengaluru
    G.add_node('n1', pos=(77.5900, 12.9716), name='Node 1')
    G.add_node('n2', pos=(77.5910, 12.9720), name='Node 2')
    G.add_node('n3', pos=(77.5920, 12.9710), name='Node 3')

    G.add_edge('n1', 'n2', weight=100)
    G.add_edge('n2', 'n3', weight=120)
    G.add_edge('n1', 'n3', weight=250)

    # Compute basic criticality attributes
    res = centrality.compute_criticality(G)
    G = res['graph'] if isinstance(res, dict) and 'graph' in res else G

    geo = graph_to_geojson(G)
    return JSONResponse(content=geo)
