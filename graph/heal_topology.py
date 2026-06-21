import networkx as nx
import math
from typing import Tuple


def euclidean_distance(a: Tuple[int,int], b: Tuple[int,int]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])


def angle_between(a: Tuple[int,int], b: Tuple[int,int], c: Tuple[int,int]) -> float:
    # angle at b between ba and bc in degrees
    import math
    bax = a[0]-b[0]
    bay = a[1]-b[1]
    bcx = c[0]-b[0]
    bcy = c[1]-b[1]
    dot = bax*bcx + bay*bcy
    mag1 = math.hypot(bax, bay)
    mag2 = math.hypot(bcx, bcy)
    if mag1==0 or mag2==0:
        return 0.0
    cosang = max(-1.0, min(1.0, dot/(mag1*mag2)))
    return math.degrees(math.acos(cosang))


def compute_angle(ep_a, ep_b, G: nx.Graph) -> float:
    # Try to compute angle based on neighbor direction; fallback to 0
    try:
        # find neighbor of ep_a that's not ep_b
        nbs_a = [n for n in G.neighbors(ep_a) if n != ep_b]
        nbs_b = [n for n in G.neighbors(ep_b) if n != ep_a]
        if len(nbs_a)==0 or len(nbs_b)==0:
            return 0.0
        return angle_between(nbs_a[0], ep_a, ep_b)
    except Exception:
        return 0.0


def heal_topology(G: nx.Graph, max_bridge_distance: int =30, max_angle_deviation: float=45.0) -> nx.Graph:
    components = list(nx.connected_components(G))
    if len(components) <= 1:
        return G

    G_candidates = nx.Graph()
    for i, comp_a in enumerate(components):
        for j, comp_b in enumerate(components):
            if i >= j:
                continue
            endpoints_a = [n for n in comp_a if G.degree(n) == 1]
            endpoints_b = [n for n in comp_b if G.degree(n) == 1]
            for ep_a in endpoints_a:
                for ep_b in endpoints_b:
                    dist = euclidean_distance((ep_a[0], ep_a[1]), (ep_b[0], ep_b[1]))
                    angle = compute_angle(ep_a, ep_b, G)
                    if dist < max_bridge_distance and angle < max_angle_deviation:
                        G_candidates.add_edge(ep_a, ep_b, weight=dist)

    if G_candidates.number_of_edges() > 0:
        mst_bridges = nx.minimum_spanning_tree(G_candidates)
        for u, v, data in mst_bridges.edges(data=True):
            G.add_edge(u, v, weight=data['weight'], edge_type='healed')

    return G
