import networkx as nx

_current_graph = None


def set_graph(graph: nx.Graph):
    """
    Store the latest generated graph in memory.
    """
    global _current_graph
    _current_graph = graph.copy()


def get_graph():
    """
    Return the latest graph.
    """
    return _current_graph


def has_graph():
    """
    Check whether a graph is available.
    """
    return _current_graph is not None


def clear_graph():
    """
    Remove stored graph.
    """
    global _current_graph
    _current_graph = None