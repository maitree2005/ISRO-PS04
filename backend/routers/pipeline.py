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

from backend.services import graph_store

router = APIRouter()


@router.post("/pipeline")
async def run_pipeline(mask: UploadFile = File(...)):
    """
    Complete Road Network Pipeline

    Upload Binary Mask
            ↓
      Skeletonization
            ↓
      Graph Generation
            ↓
      Topology Healing
            ↓
      Criticality Analysis
            ↓
      Store Graph
            ↓
      Export GeoJSON
    """

    try:
        # -----------------------------------------
        # Read uploaded image
        # -----------------------------------------

        contents = await mask.read()

        img = Image.open(io.BytesIO(contents)).convert("L")

        arr = np.array(img)

        bin_mask = (arr > 127).astype(np.uint8)

        # -----------------------------------------
        # Skeletonization
        # -----------------------------------------

        print("Generating Skeleton...")

        skeleton = skel_mod.mask_to_skeleton(bin_mask)

        # -----------------------------------------
        # Build Graph
        # -----------------------------------------

        print("Building Graph...")

        G = build_graph.skeleton_to_graph(skeleton)

        # -----------------------------------------
        # Heal Topology
        # -----------------------------------------

        print("Healing Topology...")

        G = heal_mod.heal_topology(G)

        # -----------------------------------------
        # Compute Criticality
        # -----------------------------------------

        print("Computing Criticality...")

        result = centrality_mod.compute_criticality(G)

        if isinstance(result, dict) and "graph" in result:
            G = result["graph"]

        # -----------------------------------------
        # Store Graph
        # -----------------------------------------

        print("Saving Graph in Memory...")

        print("===== GRAPH NODES =====")
        print(list(G.nodes())[:20])   # first 20 node IDs
        print("=======================")

        graph_store.set_graph(G)

        print("Graph Saved Successfully!")
        print(f"Nodes: {G.number_of_nodes()}")
        print(f"Edges: {G.number_of_edges()}")
        print(f"Graph Exists: {graph_store.has_graph()}")

        # -----------------------------------------
        # Export GeoJSON
        # -----------------------------------------

        print("Exporting GeoJSON...")

        geojson = export_geojson.graph_to_geojson(G)

        return JSONResponse(content=geojson)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )