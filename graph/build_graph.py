import networkx as nx
import numpy as np
from typing import Tuple, Dict, List


def get_neighbours(y: int, x: int, h: int, w: int, skeleton: np.ndarray) -> List[Tuple[int,int]]:
    neighbours = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx_ = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx_ < w and skeleton[ny, nx_] == 1:
                neighbours.append((ny, nx_))
    return neighbours


def skeleton_to_graph(skeleton: np.ndarray) -> nx.Graph:
    """Convert 1px skeleton to NetworkX graph.
    Nodes: intersection (3+ neighbours) or endpoints (1 neighbour).
    Edges: traced paths between nodes.
    Node positions stored as 'pos': (x, y)
    """
    h, w = skeleton.shape
    ys, xs = np.where(skeleton == 1)
    G = nx.Graph()

    node_pixels = set()
    for y, x in zip(ys, xs):
        nbs = get_neighbours(y, x, h, w, skeleton)
        n = len(nbs)
        if n == 1 or n >= 3:
            node_pixels.add((y, x))
            G.add_node((y, x), pos=(x, y), node_type="endpoint" if n == 1 else "intersection")

    visited = set()

    def trace_edge(start_pixel, next_pixel):
        path = [start_pixel, next_pixel]
        cur = next_pixel
        prev = start_pixel
        while True:
            visited.add(cur)
            nbs = [p for p in get_neighbours(cur[0], cur[1], h, w, skeleton) if p != prev]
            if len(nbs) == 0:
                # dead end
                break
            if len(nbs) > 1:
                # reached a branching point
                break
            nxt = nbs[0]
            path.append(nxt)
            prev, cur = cur, nxt
            if cur in node_pixels:
                break
        return path

    # For each node pixel, start tracing along neighbours to find edges
    for node in list(node_pixels):
        y, x = node
        for nb in get_neighbours(y, x, h, w, skeleton):
            if nb in visited:
                continue
            path = trace_edge(node, nb)
            end = path[-1]
            if end not in G.nodes:
                # if end is not a node pixel, create endpoint node
                G.add_node(end, pos=(end[1], end[0]), node_type='endpoint')
                node_pixels.add(end)
            # compute length as number of pixels
            length = len(path)
            G.add_edge(node, end, weight=length, pixels=path)

    return G
