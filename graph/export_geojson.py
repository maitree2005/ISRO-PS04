from typing import Dict, Any


def graph_to_geojson(G) -> Dict[str, Any]:
    """Convert a NetworkX graph to a GeoJSON FeatureCollection.
    Expects node attribute `pos` = (lon, lat) or (x, y).
    Edge geometry is a LineString from node positions.
    """
    features = []

    # Helper: convert numpy scalars/iterables to native Python types
    def _to_native(val):
        try:
            import numpy as _np
        except Exception:
            _np = None
        if _np is not None and isinstance(val, _np.generic):
            return val.item()
        if isinstance(val, dict):
            return {k: _to_native(v) for k, v in val.items()}
        if isinstance(val, (list, tuple)):
            return [_to_native(v) for v in val]
        return val

    # Nodes as Point features
    for n, data in G.nodes(data=True):
        pos = data.get('pos')
        if pos is None:
            continue
        lon, lat = _to_native(pos[0]), _to_native(pos[1])
        props = {k: v for k, v in data.items() if k != 'pos'}
        props.update({"id": str(_to_native(n))})
        props = {k: _to_native(v) for k, v in props.items()}
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": props
        })

    # Edges as LineString features
    for u, v, data in G.edges(data=True):
        udata = G.nodes.get(u, {})
        vdata = G.nodes.get(v, {})
        upos = udata.get('pos')
        vpos = vdata.get('pos')
        if upos is None or vpos is None:
            continue
        line = [[_to_native(upos[0]), _to_native(upos[1])], [_to_native(vpos[0]), _to_native(vpos[1])]]
        props = {k: v for k, v in data.items()}
        props.update({"u": str(_to_native(u)), "v": str(_to_native(v))})
        props = {k: _to_native(v) for k, v in props.items()}
        features.append({
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": line},
            "properties": props
        })

    return {"type": "FeatureCollection", "features": features}
