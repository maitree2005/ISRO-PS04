# ROUTE RESILIENCE — Full Project PRD
## Bharatiya Antariksh Hackathon 2026 | PS-04
### Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis for Urban Mobility

---

> **This document is the single source of truth for building the Route Resilience system.
> Feed this entire file into your coding assistant before writing any code.**

---

## 1. PROJECT SUMMARY

**What is this project?**

Route Resilience is an end-to-end AI/ML pipeline that does two things:

1. **Sees through occlusions** — Uses a Transformer-based deep learning model to extract road networks from satellite images even when roads are hidden under tree canopies, building shadows, or cloud cover. Standard segmentation models fail here because they only classify what they literally see. This model infers what should be there based on spatial context.

2. **Converts pixels to intelligence** — Takes the raw binary road mask output and converts it into a mathematically connected weighted graph. Then runs graph-theoretic analysis to identify which roads and junctions are critical bottlenecks. Simulates disaster scenarios (floods, accidents, construction) by removing nodes and measuring how badly the city's connectivity degrades.

**Primary City:** Bengaluru, Karnataka (explicitly named in ISRO problem statement)

**Who uses this?**
- Disaster response teams (which junction to evacuate first?)
- Urban planners (where is the city most fragile?)
- Traffic simulation systems (what happens if NH44 floods?)

**What is the final deliverable?**
An interactive web dashboard showing:
- Satellite image of Bengaluru overlaid with extracted road network
- Heatmap of road criticality (red = most critical, green = least)
- Click any junction → simulate its removal → see rerouting + travel time impact
- A single Resilience Index score for the entire city network

---

## 2. PROBLEM BEING SOLVED

### 2.1 The Occlusion Problem

Satellite imagery of Indian cities has three major occlusion types:

| Occlusion Type | What Happens | Why It Breaks Models |
|---|---|---|
| Tree canopies | Road spectral signature replaced by vegetation green | Model predicts "no road" |
| Building shadows | Road appears dark/black | Model predicts "shadow", not road |
| Cloud cover | Road entirely invisible | Model has no signal to work with |

Standard U-Net trained on clean data will simply leave gaps in these regions. A gap in a road mask = topologically disconnected network = useless for routing.

### 2.2 The Topology Problem

Even a perfect pixel mask is still just a raster image. It has no concept of:
- Which road connects to which
- Turn restrictions
- Road importance or hierarchy
- What happens to the rest of the network if one road is removed

To make road data useful for disaster response or traffic simulation, you need a **graph** — not a bitmap.

### 2.3 The Criticality Problem

Not all roads are equal. Losing a small lane is different from losing a flyover. But planners currently have no systematic way to quantify: "If this junction is blocked, how badly does the entire city suffer?"

Betweenness Centrality from graph theory answers exactly this — it counts how many shortest paths between all city point-pairs pass through each node. High centrality = bottleneck = gatekeeper.

---

## 3. SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                              │
│  Sentinel-2 (10m) / LISS-IV (5.8m) / Cartosat-3 satellite img  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PHASE 1: PREPROCESSING                        │
│  • Tile large images into 512×512 patches                       │
│  • Normalize pixel values (mean/std per band)                   │
│  • Contrast enhancement (CLAHE)                                 │
│  • Synthetic occlusion augmentation (training only)             │
│  • Band selection: RGB + NIR (4-channel input)                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 2: OCCLUSION-AWARE SEGMENTATION              │
│  • Model: SegFormer-B2 (pretrained on ImageNet)                 │
│  • Fine-tuned on: SpaceNet Roads + DeepGlobe + OSM labels       │
│  • Occlusion training: synthetic masks pasted over roads        │
│  • Loss: Dice + IoU + Boundary-aware + optional Connectivity    │
│  • Output: Binary road mask (0 = no road, 1 = road)             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 3: SKELETONIZATION & GRAPH CONSTRUCTION      │
│  • Morphological thinning → 1-pixel centerlines                 │
│  • Node detection: pixels with 3+ neighbours = intersection     │
│  • Node detection: pixels with 1 neighbour = endpoint           │
│  • Edge construction: skeleton segments between nodes           │
│  • Edge weight: length (pixels → metres via GSD)                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 4: TOPOLOGICAL HEALING                       │
│  • Detect disconnected components (networkx connected_components)│
│  • For each pair of endpoints across components:                │
│    - If Euclidean distance < threshold (30px): candidate bridge  │
│    - If angular deviation < 45°: natural trajectory confirmed    │
│  • Connect via Minimum Spanning Tree (MST)                      │
│  • Verify full connectivity via Union-Find (Disjoint Sets)      │
│  • Output: Single connected weighted graph G(V, E, W)           │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 5: CRITICALITY ANALYSIS                      │
│  • Betweenness Centrality for all nodes                         │
│  • Bridge detection (edges whose removal disconnects graph)     │
│  • Rank nodes by centrality score                               │
│  • Assign Criticality Score to each edge/node                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 6: STRESS TESTING & RESILIENCE INDEX         │
│  • Baseline: compute avg shortest path length L_base            │
│  • Iterative node ablation (remove top-N centrality nodes)      │
│  • After each removal: recompute avg shortest path L_perturbed  │
│  • Resilience Index R = L_base / L_perturbed                    │
│    (R close to 1 = resilient, R << 1 = fragile)                 │
│  • Output: per-node resilience impact score                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 7: INTERACTIVE DASHBOARD                     │
│  • Leaflet.js map centred on Bengaluru                          │
│  • Road network overlaid as GeoJSON                             │
│  • Criticality heatmap layer (colour-coded by score)            │
│  • Click node → remove → recalculate → show rerouting           │
│  • Resilience Index display panel                               │
│  • Export: GeoJSON, PNG heatmap, CSV metrics                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. DETAILED TECH STACK

### 4.1 ML / Deep Learning

| Component | Choice | Reason |
|---|---|---|
| Primary model | **SegFormer-B2** (HuggingFace `nvidia/mit-b2`) | Transformer-based, best balance of speed and accuracy for 30hr constraint. Better long-range context than CNN. |
| Fallback model | **DeepLabV3+** with ResNet-50 backbone | If GPU memory is limited, lighter and faster |
| Framework | **PyTorch 2.x** | Standard, best ecosystem for geospatial ML |
| Training lib | **HuggingFace Transformers + Segmentation Models PyTorch (SMP)** | Pretrained weights + easy fine-tuning |
| Data augmentation | **Albumentations** | Fastest augmentation library for image segmentation |
| Geospatial IO | **Rasterio + GDAL** | Reading/writing GeoTIFF satellite files |
| Image processing | **OpenCV + scikit-image** | Preprocessing + skeletonization |

### 4.2 Geospatial Processing

| Component | Choice | Reason |
|---|---|---|
| Skeletonization | `skimage.morphology.skeletonize` | Medial axis thinning to 1px centerline |
| OSM data download | **OSMnx** | `ox.graph_from_place("Bengaluru, India")` — 2 lines |
| Coordinate systems | **Pyproj + GDAL** | CRS transformations (WGS84 ↔ UTM) |
| Vector data | **Geopandas + Shapely** | GeoDataFrame operations, spatial joins |
| GeoJSON export | **Geopandas `.to_file(driver="GeoJSON")`** | Dashboard-ready output |

### 4.3 Graph Analysis

| Component | Choice | Reason |
|---|---|---|
| Graph library | **NetworkX** | Standard, all centrality metrics built-in |
| Betweenness centrality | `nx.betweenness_centrality(G)` | Core metric for bottleneck identification |
| Bridge detection | `nx.bridges(G)` | Finds edges whose removal disconnects graph |
| MST healing | `nx.minimum_spanning_tree(G_candidates)` | Connect broken components |
| Connected components | `nx.connected_components(G)` | Identify disconnected subgraphs |
| Shortest paths | `nx.average_shortest_path_length(G)` | Resilience Index calculation |
| Advanced (optional) | **PyTorch Geometric (PyG)** | GNN approach for learned graph features |

### 4.4 Backend

| Component | Choice |
|---|---|
| Language | Python 3.11 |
| API framework | **FastAPI** |
| Server | **Uvicorn** |
| Task queue | **Celery + Redis** (for long-running graph computation) |
| Data serialization | **Pydantic v2** |
| File storage | Local filesystem (hackathon scope) |

### 4.5 Frontend / Dashboard

| Component | Choice | Reason |
|---|---|---|
| Map rendering | **Leaflet.js** | Lightweight, GeoJSON native, free |
| Base map tiles | **OpenStreetMap / CartoDB Dark Matter** | Free, no API key |
| Graph viz | **D3.js** | Force-directed graph for network view |
| Heatmap layer | **Leaflet.heat plugin** | Colour-coded criticality overlay |
| UI framework | **Next.js 14 (App Router)** | Your existing stack from Mumbai Through Rails |
| Styling | **Tailwind CSS** | Rapid dark-mode dashboard |
| Charts | **Recharts** | Resilience Index over time chart |
| State | **React useState / useReducer** | Simple enough, no Redux needed |

### 4.6 Dev Environment

| Tool | Purpose |
|---|---|
| VS Code | Primary IDE |
| Python venv | Isolated ML environment |
| Node 18+ | Frontend |
| CUDA 12.x | GPU training (if local GPU available) |
| Google Colab Pro / Kaggle | Free GPU for model training if no local GPU |
| Docker | Containerize FastAPI backend |

---

## 5. FOLDER STRUCTURE

```
route-resilience/
│
├── README.md
├── PRD.md                          ← this file
├── .env.example
├── docker-compose.yml
│
├── ml/                             ← All ML code
│   ├── data/
│   │   ├── raw/                    ← Downloaded satellite tiles
│   │   ├── processed/              ← Tiled 512×512 patches
│   │   ├── masks/                  ← Ground truth binary masks
│   │   └── augmented/              ← Synthetically occluded training data
│   │
│   ├── preprocessing/
│   │   ├── tile_images.py          ← Slice large GeoTIFF → 512×512 patches
│   │   ├── normalize.py            ← Band normalization + CLAHE
│   │   ├── generate_masks.py       ← OSM → binary road mask via rasterio
│   │   └── synthetic_occlusions.py ← Paste tree/shadow textures over roads
│   │
│   ├── models/
│   │   ├── segformer.py            ← SegFormer-B2 fine-tuning wrapper
│   │   ├── deeplabv3.py            ← Fallback model
│   │   └── losses.py               ← Dice + IoU + Boundary + Connectivity loss
│   │
│   ├── training/
│   │   ├── train.py                ← Main training loop
│   │   ├── evaluate.py             ← IoU, Dice, Occlusion-Recall metrics
│   │   └── config.yaml             ← Hyperparameters
│   │
│   ├── inference/
│   │   ├── predict.py              ← Run model on new satellite tile
│   │   └── stitch_tiles.py         ← Merge tiled predictions back to full image
│   │
│   └── checkpoints/                ← Saved model weights (.pth files)
│
├── graph/                          ← Graph processing pipeline
│   ├── skeletonize.py              ← Binary mask → 1px centerline
│   ├── build_graph.py              ← Centerline → NetworkX graph
│   ├── heal_topology.py            ← MST + Union-Find gap bridging
│   ├── centrality.py               ← Betweenness + bridge detection
│   ├── stress_test.py              ← Node ablation + Resilience Index
│   └── export_geojson.py           ← Graph → GeoJSON for dashboard
│
├── backend/                        ← FastAPI server
│   ├── main.py                     ← App entry point
│   ├── routers/
│   │   ├── predict.py              ← POST /predict → run full pipeline
│   │   ├── graph.py                ← GET /graph → return GeoJSON
│   │   ├── simulate.py             ← POST /simulate → node removal
│   │   └── resilience.py           ← GET /resilience → index scores
│   ├── schemas.py                  ← Pydantic models
│   └── utils.py
│
├── frontend/                       ← Next.js dashboard
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx                ← Main dashboard
│   │   └── api/                    ← Next.js API routes (proxy to FastAPI)
│   ├── components/
│   │   ├── Map.tsx                 ← Leaflet map component
│   │   ├── CriticalityHeatmap.tsx  ← Heatmap layer
│   │   ├── NodeSimulator.tsx       ← Click-to-remove UI
│   │   ├── ResiliencePanel.tsx     ← Index display + chart
│   │   └── GraphViewer.tsx         ← D3 network graph
│   └── public/
│
└── scripts/
    ├── download_sentinel2.py       ← Copernicus API download
    ├── download_osm.py             ← OSMnx Bengaluru download
    └── run_pipeline.sh             ← End-to-end single command
```

---

## 6. DATA PIPELINE

### 6.1 Data Sources (in order of priority for 30hrs)

**Step 1 — OSM for Bengaluru (immediate, 5 minutes)**
```python
import osmnx as ox
G = ox.graph_from_place("Bengaluru, India", network_type="drive")
ox.save_graphml(G, "data/osm_bengaluru.graphml")
```
Use this as: ground truth for evaluation + mock input for graph pipeline during parallel dev.

**Step 2 — SpaceNet Roads Dataset (pretrain ML model)**
- URL: https://spacenet.ai/roads/
- Contains: Satellite imagery + road mask pairs
- Cities: Las Vegas, Paris, Shanghai, Khartoum
- Download via AWS CLI: `aws s3 cp s3://spacenet-dataset/spacenet/SN3_roads/...`
- Use for: pretraining SegFormer-B2

**Step 3 — Sentinel-2 for Bengaluru (fine-tuning + demo)**
- URL: https://browser.dataspace.copernicus.eu/
- How to download: Create free account → search "Bengaluru" → filter by cloud cover <10% → download L2A product
- Bands needed: B02 (Blue), B03 (Green), B04 (Red), B08 (NIR)
- Resolution: 10m/pixel
- Convert to GeoTIFF using GDAL

**Step 4 — Cartosat-3 (provided by ISRO during hackathon)**
- 0.25m resolution — extremely high detail
- Plug into inference pipeline after training is done
- Primary evaluation dataset

### 6.2 Ground Truth Mask Generation
```python
# Generate binary road mask from OSM vectors
import rasterio
import geopandas as gpd
from rasterio.features import rasterize

# Load OSM road vectors
roads = gpd.read_file("osm_roads.geojson")

# Rasterize onto same grid as satellite image
with rasterio.open("sentinel2_bengaluru.tif") as src:
    mask = rasterize(
        [(geom, 1) for geom in roads.geometry],
        out_shape=src.shape,
        transform=src.transform,
        fill=0,
        dtype="uint8"
    )
```

---

## 7. ML MODEL DETAILS

### 7.1 SegFormer-B2 Architecture

SegFormer is a Transformer-based segmentation model that:
- Uses hierarchical Transformer encoder (Mix Transformer / MiT)
- No positional encoding → works on arbitrary image sizes
- Lightweight MLP decoder → fast inference
- Pre-trained on ImageNet-22K → excellent feature extraction

**Why better than U-Net for this problem:**
U-Net uses local convolutions. It classifies each pixel based on a limited receptive field. If a road disappears under a tree canopy for 50 pixels, U-Net has no way to know the road continues. SegFormer's self-attention mechanism can relate pixels across the entire image — so it can "see" road on both sides of the canopy and infer the connection.

### 7.2 Training Configuration
```yaml
# config.yaml
model:
  name: "nvidia/mit-b2"
  num_classes: 2
  image_size: 512

training:
  batch_size: 8
  learning_rate: 6e-5
  epochs: 50
  optimizer: "AdamW"
  scheduler: "CosineAnnealingLR"
  warmup_steps: 500

loss:
  dice_weight: 0.4
  iou_weight: 0.4
  boundary_weight: 0.2

augmentation:
  horizontal_flip: true
  vertical_flip: true
  rotation: 90
  color_jitter: true
  synthetic_occlusion_prob: 0.4   # 40% of training images get synthetic occlusions
  occlusion_types:
    - "tree_canopy"                # paste green texture patches
    - "shadow"                    # darken random road segments
    - "cloud"                     # paste white/grey blobs

data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
```

### 7.3 Synthetic Occlusion Generation
```python
def add_synthetic_occlusion(image, mask, occlusion_type="tree"):
    """
    Paste synthetic occlusions over known road pixels.
    Forces model to learn road completion from context.
    """
    occluded_image = image.copy()
    road_pixels = np.where(mask == 1)
    
    if len(road_pixels[0]) == 0:
        return occluded_image
    
    # Pick random contiguous road segment to occlude
    start_idx = np.random.randint(0, len(road_pixels[0]))
    segment_length = np.random.randint(20, 80)  # pixels
    
    if occlusion_type == "tree":
        # Green texture with noise
        occluded_image[y:y+h, x:x+w] = generate_vegetation_texture(h, w)
    elif occlusion_type == "shadow":
        # Darken the road segment
        occluded_image[y:y+h, x:x+w] = (image[y:y+h, x:x+w] * 0.2).astype(np.uint8)
    elif occlusion_type == "cloud":
        # White/grey blob
        occluded_image[y:y+h, x:x+w] = np.random.randint(200, 255, (h, w, 3))
    
    return occluded_image
```

### 7.4 Loss Functions
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, dice_w=0.4, iou_w=0.4, boundary_w=0.2):
        super().__init__()
        self.dice_w = dice_w
        self.iou_w = iou_w
        self.boundary_w = boundary_w

    def dice_loss(self, pred, target):
        smooth = 1e-6
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    def iou_loss(self, pred, target):
        smooth = 1e-6
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        return 1 - (intersection + smooth) / (union + smooth)

    def boundary_loss(self, pred, target):
        # Penalize errors near road edges more heavily
        pred = torch.sigmoid(pred)
        boundary = self._get_boundary(target)
        boundary_error = F.binary_cross_entropy(pred * boundary, target * boundary)
        return boundary_error

    def _get_boundary(self, mask):
        # Dilate - erode = boundary ring
        from kornia.morphology import dilation, erosion
        kernel = torch.ones(5, 5).to(mask.device)
        dilated = dilation(mask.unsqueeze(1).float(), kernel)
        eroded = erosion(mask.unsqueeze(1).float(), kernel)
        return (dilated - eroded).squeeze(1)

    def forward(self, pred, target):
        return (self.dice_w * self.dice_loss(pred, target) +
                self.iou_w * self.iou_loss(pred, target) +
                self.boundary_w * self.boundary_loss(pred, target))
```

---

## 8. GRAPH PIPELINE DETAILS

### 8.1 Skeletonization
```python
from skimage.morphology import skeletonize
import numpy as np

def mask_to_skeleton(binary_mask):
    """
    Reduce fat road pixels to 1-pixel wide centerline.
    Input: binary mask (H x W), values 0 or 1
    Output: skeleton (H x W), values 0 or 1
    """
    skeleton = skeletonize(binary_mask.astype(bool))
    return skeleton.astype(np.uint8)
```

### 8.2 Graph Construction from Skeleton
```python
import networkx as nx
import numpy as np

def skeleton_to_graph(skeleton):
    """
    Convert 1px skeleton to NetworkX graph.
    Nodes = intersections + endpoints
    Edges = road segments between nodes
    """
    G = nx.Graph()
    h, w = skeleton.shape
    
    # Find all skeleton pixels
    ys, xs = np.where(skeleton == 1)
    
    def get_neighbours(y, x):
        neighbours = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx_ < w and skeleton[ny, nx_] == 1:
                    neighbours.append((ny, nx_))
        return neighbours
    
    # Classify pixels: endpoint (1 neighbour), intersection (3+ neighbours)
    node_pixels = []
    for y, x in zip(ys, xs):
        n = len(get_neighbours(y, x))
        if n == 1 or n >= 3:
            node_pixels.append((y, x))
            G.add_node((y, x), pos=(x, y), node_type="endpoint" if n == 1 else "intersection")
    
    # Trace edges between nodes along skeleton
    # ... (path tracing logic)
    
    return G
```

### 8.3 Topological Healing
```python
def heal_topology(G, max_bridge_distance=30, max_angle_deviation=45):
    """
    Connect disconnected components using MST + geometric constraints.
    """
    components = list(nx.connected_components(G))
    
    if len(components) == 1:
        return G  # already connected
    
    # Build candidate bridge graph between component endpoints
    G_candidates = nx.Graph()
    
    for i, comp_a in enumerate(components):
        for j, comp_b in enumerate(components):
            if i >= j:
                continue
            
            endpoints_a = [n for n in comp_a if G.degree(n) == 1]
            endpoints_b = [n for n in comp_b if G.degree(n) == 1]
            
            for ep_a in endpoints_a:
                for ep_b in endpoints_b:
                    dist = euclidean_distance(ep_a, ep_b)
                    angle = compute_angle(ep_a, ep_b, G)
                    
                    if dist < max_bridge_distance and angle < max_angle_deviation:
                        G_candidates.add_edge(ep_a, ep_b, weight=dist)
    
    # MST on candidates to get minimum set of bridges
    if G_candidates.number_of_edges() > 0:
        mst_bridges = nx.minimum_spanning_tree(G_candidates)
        for u, v, data in mst_bridges.edges(data=True):
            G.add_edge(u, v, weight=data['weight'], edge_type='healed')
    
    return G
```

### 8.4 Criticality Analysis
```python
def compute_criticality(G):
    """
    Compute betweenness centrality and identify critical nodes.
    """
    # Betweenness centrality (normalized)
    bc = nx.betweenness_centrality(G, normalized=True, weight='weight')
    nx.set_node_attributes(G, bc, 'betweenness')
    
    # Bridge edges (removal disconnects graph)
    bridges = list(nx.bridges(G))
    for u, v in bridges:
        G[u][v]['is_bridge'] = True
    
    # Rank nodes
    ranked_nodes = sorted(bc.items(), key=lambda x: x[1], reverse=True)
    
    return G, ranked_nodes

def compute_resilience_index(G, nodes_to_remove):
    """
    Resilience Index R = L_base / L_perturbed
    R = 1 → no impact
    R < 1 → network degraded (lower = worse)
    """
    # Baseline (only on largest connected component for valid computation)
    lcc = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    L_base = nx.average_shortest_path_length(lcc, weight='weight')
    
    results = []
    G_test = G.copy()
    
    for node in nodes_to_remove:
        G_test.remove_node(node)
        lcc_test = G_test.subgraph(max(nx.connected_components(G_test), key=len)).copy()
        
        if len(lcc_test) < 2:
            R = 0  # network completely fragmented
        else:
            L_perturbed = nx.average_shortest_path_length(lcc_test, weight='weight')
            R = L_base / L_perturbed
        
        results.append({
            "node": node,
            "resilience_index": R,
            "path_length_increase_pct": ((L_perturbed - L_base) / L_base) * 100
        })
    
    return results
```

---

## 9. API ENDPOINTS

### Backend (FastAPI — port 8000)

| Method | Endpoint | Input | Output | Description |
|---|---|---|---|---|
| `POST` | `/api/predict` | GeoTIFF satellite image | Binary road mask (PNG) | Run segmentation model |
| `POST` | `/api/pipeline` | GeoTIFF | Full GeoJSON graph | Run end-to-end pipeline |
| `GET` | `/api/graph` | `?city=bengaluru` | GeoJSON FeatureCollection | Return precomputed graph |
| `GET` | `/api/criticality` | `?city=bengaluru` | JSON node scores | Betweenness centrality scores |
| `POST` | `/api/simulate` | `{node_id, city}` | JSON resilience result | Remove node + compute impact |
| `GET` | `/api/resilience` | `?city=bengaluru&top_n=10` | JSON resilience report | Top-N node ablation results |
| `GET` | `/api/health` | — | `{"status": "ok"}` | Health check |

### Example API Response — `/api/criticality`
```json
{
  "city": "bengaluru",
  "total_nodes": 12847,
  "top_bottlenecks": [
    {
      "node_id": "node_4521",
      "lat": 12.9716,
      "lon": 77.5946,
      "betweenness_score": 0.847,
      "is_bridge": true,
      "criticality_rank": 1,
      "label": "Silk Board Junction"
    }
  ]
}
```

### Example API Response — `/api/simulate`
```json
{
  "node_removed": "node_4521",
  "baseline_avg_path_length": 42.3,
  "perturbed_avg_path_length": 67.8,
  "resilience_index": 0.624,
  "path_length_increase_pct": 60.3,
  "affected_routes": 8432,
  "largest_connected_component_before": 12847,
  "largest_connected_component_after": 9103,
  "rerouting_geojson": { "type": "FeatureCollection", "features": [] }
}
```

---

## 10. DASHBOARD UI SPECIFICATION

### 10.1 Layout

```
┌─────────────────────────────────────────────────────────┐
│  ROUTE RESILIENCE   [City: Bengaluru ▼]  [Export ↓]    │
├──────────────────────────────────┬──────────────────────┤
│                                  │  RESILIENCE PANEL    │
│                                  │  ┌────────────────┐  │
│                                  │  │ City Score     │  │
│         LEAFLET MAP              │  │    0.74        │  │
│         (fullscreen)             │  └────────────────┘  │
│                                  │                      │
│    [road network overlay]        │  TOP BOTTLENECKS     │
│    [criticality heatmap]         │  1. Silk Board  0.85 │
│    [rerouting paths]             │  2. Hebbal      0.79 │
│                                  │  3. KR Puram    0.71 │
│                                  │                      │
│                                  │  SIMULATION MODE     │
│                                  │  [Click node on map] │
│                                  │  to simulate removal │
│                                  │                      │
│                                  │  [Reset Simulation]  │
└──────────────────────────────────┴──────────────────────┘
│  LAYER CONTROLS: [Roads] [Heatmap] [Simulate] [OSM]     │
└─────────────────────────────────────────────────────────┘
```

### 10.2 Map Layers

| Layer | Description | Colour Scheme |
|---|---|---|
| Road Network | GeoJSON LineString overlay | White on dark basemap |
| Criticality Heatmap | Node betweenness → colour | Green (low) → Yellow → Red (high) |
| Bridge Edges | Edges that disconnect graph if removed | Pulsing orange |
| Rerouting Paths | After simulation, show alternate routes | Blue dashed |
| Disabled Nodes | User-removed nodes | Red X marker |

### 10.3 Interaction Flow

1. User loads dashboard → sees Bengaluru with road network + criticality heatmap
2. User hovers node → tooltip shows "Silk Board Junction — Criticality: 0.85 — 8,432 routes pass through"
3. User clicks node → node turns red → loading spinner → panel updates with simulation result
4. Panel shows: "Removing this junction increases average travel time by 60.3%. 3,744 routes disconnected."
5. Map shows rerouted paths in blue
6. User can click "Reset Simulation" to restore network
7. User can toggle layers on/off
8. User can export: GeoJSON graph, PNG heatmap, CSV report

---

## 11. EVALUATION METRICS IMPLEMENTATION

```python
def evaluate_all_metrics(pred_mask, gt_mask, pred_graph, gt_graph_osm):
    
    # 1. IoU Score
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    iou = intersection / union
    
    # 2. Dice Score
    dice = (2 * intersection) / (pred_mask.sum() + gt_mask.sum())
    
    # 3. Occlusion-Recall (most important for this PS)
    # Only evaluate on pixels that were occluded in input but present in GT
    occluded_road_pixels = np.where((occluded_input == 0) & (gt_mask == 1))
    occlusion_recall = pred_mask[occluded_road_pixels].mean()
    
    # 4. Connectivity Ratio
    cc_before = nx.number_connected_components(raw_graph)
    cc_after = nx.number_connected_components(healed_graph)
    connectivity_ratio = 1 - (cc_after / cc_before)  # higher = better healing
    
    # 5. Topological Accuracy vs OSM
    # Sample 100 random point pairs, compare path lengths
    errors = []
    node_list = list(pred_graph.nodes())
    for _ in range(100):
        u, v = random.sample(node_list, 2)
        try:
            pred_path = nx.shortest_path_length(pred_graph, u, v, weight='weight')
            osm_path = get_osm_path_length(u, v, gt_graph_osm)
            errors.append(abs(pred_path - osm_path) / osm_path)
        except nx.NetworkXNoPath:
            errors.append(1.0)  # penalise disconnected
    topological_accuracy = 1 - np.mean(errors)
    
    # 6. Relaxed IoU (3px buffer)
    from scipy.ndimage import binary_dilation
    dilated_gt = binary_dilation(gt_mask, iterations=3)
    relaxed_tp = np.logical_and(pred_mask, dilated_gt).sum()
    relaxed_iou = relaxed_tp / np.logical_or(pred_mask, gt_mask).sum()
    
    return {
        "iou": iou,
        "dice": dice,
        "occlusion_recall": occlusion_recall,
        "connectivity_ratio": connectivity_ratio,
        "topological_accuracy": topological_accuracy,
        "relaxed_iou_3px": relaxed_iou
    }
```

---

## 12. 30-HOUR EXECUTION PLAN

### Team Split: Sub-team A (ML, 2 people) | Sub-team B (Graph + UI, 2 people)

```
HOUR 00-02  [BOTH TEAMS]
  - Setup repos, environments, install dependencies
  - Sub-A: Download SpaceNet Roads dataset, verify CUDA
  - Sub-B: Download OSM Bengaluru via osmnx, verify graph

HOUR 02-08  [PARALLEL]
  Sub-A: Train baseline U-Net on SpaceNet Roads
         Target: IoU > 0.60 on validation set
         Use: smp.Unet("resnet50", encoder_weights="imagenet")
  
  Sub-B: Build skeletonization pipeline on OSM vectors
         OSM graph → GeoJSON → skeleton → NetworkX graph
         This is the "mock" pipeline Sub-A will plug into later

HOUR 08-14  [PARALLEL]
  Sub-A: Switch to SegFormer-B2
         Add synthetic occlusion augmentation
         Add Dice + IoU + Boundary loss
         Re-train, target: IoU > 0.70, Occlusion-Recall > 0.60
  
  Sub-B: Implement centrality + stress testing
         Betweenness centrality on Bengaluru OSM graph
         Node ablation + Resilience Index calculation
         FastAPI backend with /api/graph and /api/simulate endpoints

HOUR 14-20  [PARALLEL]
  Sub-A: Fine-tune on Sentinel-2 Bengaluru imagery
         Evaluate on occluded test patches
         Export: model weights + inference script
  
  Sub-B: Build Next.js dashboard
         Leaflet map + GeoJSON road overlay
         Criticality heatmap layer
         Simulation panel (click → API call → update)

HOUR 20-24  [INTEGRATION]
  - Connect Sub-A model output to Sub-B pipeline
  - Run full end-to-end: Sentinel-2 image → road mask → graph → dashboard
  - Fix integration bugs
  - Load Bengaluru data into dashboard

HOUR 24-28  [POLISH + EVALUATION]
  - Run all evaluation metrics, record numbers
  - Fix critical bugs
  - Improve UI (criticality colours, tooltips, export)
  - If Cartosat-3 data available: run inference on it

HOUR 28-30  [DEMO PREP]
  - Prepare 5-minute demo flow
  - Screenshot final results
  - Write summary of metrics achieved
  - Prepare "what-if Silk Board floods" demo scenario
```

---

## 13. DEPENDENCIES — COMPLETE LIST

### Python (requirements.txt)
```
# ML
torch==2.2.0
torchvision==0.17.0
transformers==4.38.0
segmentation-models-pytorch==0.3.3
albumentations==1.3.1
timm==0.9.12

# Geospatial
rasterio==1.3.9
gdal==3.8.4
geopandas==0.14.3
shapely==2.0.3
pyproj==3.6.1
osmnx==1.9.1
fiona==1.9.5

# Image processing
opencv-python==4.9.0.80
scikit-image==0.22.0
Pillow==10.2.0
numpy==1.26.4

# Graph
networkx==3.2.1
scipy==1.12.0

# Backend
fastapi==0.110.0
uvicorn==0.27.1
pydantic==2.6.1
python-multipart==0.0.9

# Utils
tqdm==4.66.2
matplotlib==3.8.3
pandas==2.2.1
PyYAML==6.0.1
```

### Node.js (package.json)
```json
{
  "dependencies": {
    "next": "14.1.0",
    "react": "18.2.0",
    "react-dom": "18.2.0",
    "leaflet": "1.9.4",
    "react-leaflet": "4.2.1",
    "leaflet.heat": "0.2.0",
    "d3": "7.9.0",
    "recharts": "2.12.2",
    "tailwindcss": "3.4.1",
    "axios": "1.6.7"
  }
}
```

---

## 14. KEY CONCEPTS GLOSSARY

| Term | Definition |
|---|---|
| **Spectral Blindness** | When a satellite model fails to detect a road because the road's spectral signature is replaced by another material (trees, shadow) |
| **Topological Connectivity** | Whether every node in a road network can reach every other node via some path |
| **Skeletonization** | Morphological operation that reduces thick road pixels to 1-pixel wide centerlines while preserving topology |
| **Betweenness Centrality** | For each node, the fraction of all shortest paths in the network that pass through it |
| **Gatekeeper Node** | A node with very high betweenness centrality — its removal causes maximum disruption |
| **Bridge Edge** | An edge whose removal disconnects the graph into two or more components |
| **MST Healing** | Using Minimum Spanning Tree to connect disconnected graph components with minimum total added edge weight |
| **Union-Find** | Data structure to efficiently track and merge disjoint sets — used to verify all components are connected after healing |
| **Resilience Index** | R = L_base / L_perturbed. Ratio of average shortest path length before vs after node removal. R=1 means no impact. |
| **Node Ablation** | Systematically removing nodes from a graph to measure the impact on network performance |
| **Occlusion-Recall** | The model's ability to correctly predict roads specifically in regions that were occluded in the input image |
| **GSD** | Ground Sampling Distance — the real-world size (in metres) represented by one pixel |

---

## 15. DEMO SCRIPT (5 MINUTES)

```
[0:00 - 0:30] — The Problem
Show a raw Sentinel-2 satellite image of Bengaluru.
Highlight a road disappearing under a tree canopy.
"Standard AI sees no road here. Our model does."

[0:30 - 1:30] — The Model
Show split screen: input image (occluded) vs output mask (complete road network).
Show Occlusion-Recall metric on screen.
"We trained on synthetic occlusions so the model learns to infer, not just see."

[1:30 - 2:30] — The Graph
Animate: binary mask → skeleton → graph nodes/edges.
Show connected graph on Bengaluru map.
"We converted 847,000 road pixels into 12,847 nodes and 31,204 edges."

[2:30 - 3:30] — Criticality Heatmap
Show dashboard with heatmap layer.
"Red nodes are bottlenecks. Silk Board Junction handles 8,432 routes."
Hover over red nodes to show tooltip scores.

[3:30 - 4:30] — Simulation
Click Silk Board Junction on the map.
Watch rerouting paths appear in blue.
Panel shows: "Removing this junction increases average city travel time by 60.3%"
"This is real-time disaster impact prediction."

[4:30 - 5:00] — Resilience Index
Show the city's overall Resilience Index: 0.74
"Bengaluru scores 0.74. The top 3 junctions together bring it to 0.41 — near collapse."
"This tool tells planners exactly where to invest in infrastructure redundancy."
```

---

## 16. WHAT MAKES THIS SUBMISSION WIN

1. **Occlusion-specific evaluation** — explicitly compute and show Occlusion-Recall, not just IoU. Most teams won't.
2. **Topology healing is visible** — show before/after connectivity ratio. Quantify the improvement.
3. **Resilience Index is a single number** — judges can understand it instantly. "Bengaluru is 0.74 resilient."
4. **Named real junctions** — "Silk Board, Hebbal, KR Puram" in your demo. Shows you used real Bengaluru data.
5. **Interactive simulation** — clicking a node and seeing the city respond in real-time is memorable.
6. **ISRO data used** — even if just Sentinel-2, mention Cartosat-3 compatibility. Shows ISRO alignment.
7. **Parallel team workflow** — shows technical maturity. Mention it in your presentation.

---

*End of PRD — Feed this entire document into your coding assistant as project context before writing any code.*