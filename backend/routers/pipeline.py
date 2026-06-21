from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import io
import numpy as np
from PIL import Image

import graph.skeletonize as skel_mod
import graph.build_graph as build_graph
import graph.heal_topology as heal_mod
import graph.centrality as centrality_mod
import graph.export_geojson as export_geojson

router = APIRouter()


@router.post("/pipeline")
async def run_pipeline(mask: UploadFile = File(...)):
    """Accept a binary road mask image (PNG/JPEG) and run the graph pipeline, returning GeoJSON."""
    try:
        contents = await mask.read()
        img = Image.open(io.BytesIO(contents)).convert("L")
        arr = np.array(img)
        bin_mask = (arr > 127).astype(np.uint8)

        skeleton = skel_mod.mask_to_skeleton(bin_mask)
        G = build_graph.skeleton_to_graph(skeleton)
        G = heal_mod.heal_topology(G)
        res = centrality_mod.compute_criticality(G)
        if isinstance(res, dict) and 'graph' in res:
            G = res['graph']

        geo = export_geojson.graph_to_geojson(G)
        return JSONResponse(content=geo)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
