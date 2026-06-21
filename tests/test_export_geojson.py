import networkx as nx
from graph.export_geojson import graph_to_geojson


def test_graph_to_geojson():
    G = nx.Graph()
    G.add_node('a', pos=(77.59, 12.97), val=1)
    G.add_node('b', pos=(77.60, 12.98), val=2)
    G.add_edge('a', 'b', weight=10)

    geo = graph_to_geojson(G)
    assert isinstance(geo, dict)
    assert geo['type'] == 'FeatureCollection'
    assert len(geo['features']) == 3  # 2 nodes + 1 edge
