# Route Resilience — ISRO PS-04 Prototype

Route Resilience is an end-to-end prototype for occlusion-aware road extraction and graph-based resilience analysis. The project combines computer vision, graph theory, and a lightweight web dashboard to turn satellite imagery into actionable insights about road network robustness.

This repository is designed as a hackathon-style prototype for the ISRO PS-04 problem statement. It demonstrates a workflow that:

- segments roads from imagery,
- converts the resulting mask into a graph,
- identifies critical junctions and bottlenecks,
- and simulates the impact of node removal on network resilience.

---

## What this project does

The system is built around a simple pipeline:

1. A user uploads an image or mask.
2. The backend runs inference and produces a binary road mask.
3. The mask is converted into a skeletal road network.
4. A graph is built from the skeleton.
5. Topology healing and centrality analysis are applied.
6. The resulting graph can be visualized and simulated through the web interface.

This makes the project useful for exploring questions such as:

- Which road segments are most critical to connectivity?
- What happens if a major junction is disrupted?
- How resilient is a road network under stress or failure?

---

## Project architecture

The repository is organized into four main parts:

- Backend: FastAPI API for prediction, graph construction, simulation, and model management.
- Frontend: Next.js + Leaflet dashboard for visualization.
- ML: training and inference helpers for segmentation models.
- Graph pipeline: skeletonization, graph building, topology healing, and centrality analysis.

A high-level flow looks like this:

```text
Image input
  -> segmentation / mask generation
  -> skeletonization
  -> graph construction
  -> topology healing
  -> criticality analysis
  -> API / dashboard output
```

---

## Repository structure

```text
backend/              # FastAPI application and routers
frontend/             # Next.js dashboard
ml/                   # Model code, training, preprocessing, and inference
graph/                # Skeletonization, graph construction, healing, and metrics
tests/                # Unit and integration tests
requirements.txt      # Python dependencies
frontend/package.json # Frontend dependencies
```

---

## Prerequisites

Before running the project locally, ensure you have:

- Python 3.10+ recommended
- Node.js 18+ and npm
- A terminal with access to the repository root

On Windows, PowerShell is used in the examples below.

---

## Setup

### 1. Create a Python environment

```powershell
cd C:\Users\Maitree\OneDrive\Desktop\ISRO_Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2. Install Python dependencies

```powershell
pip install -r requirements.txt
```

If you want to work with the training and inference pieces more fully, make sure the ML packages from the requirements file are available. The repository also expects PyTorch and transformers-style dependencies to be present for model-based inference.

### 3. Install frontend dependencies

```powershell
cd frontend
npm install
```

---

## Run the application

### Start the backend

From the repository root:

```powershell
uvicorn backend.main:app --reload --port 8000
```

The API will be available at:

- http://localhost:8000/api/health
- http://localhost:8000/docs

### Start the frontend

In a second terminal:

```powershell
cd frontend
npm run dev
```

Then open the frontend at:

- http://localhost:3000

The frontend is designed to talk to the backend at http://localhost:8000 by default.

---

## API overview

The FastAPI backend exposes several routes:

- POST /api/predict: runs inference on an uploaded image and returns a PNG mask.
- POST /api/pipeline: runs the full graph workflow on a mask upload.
- GET /api/graph: retrieves the stored graph payload.
- POST /api/simulate: simulates node removal and returns resilience metrics.
- GET /api/health: basic health endpoint.
- POST /api/models/preload: preload a model wrapper for repeated inference.

Example health check:

```powershell
curl http://localhost:8000/api/health
```

Example prediction request:

```powershell
curl -X POST -F "image=@C:\path\to\image.png" http://localhost:8000/api/predict --output mask.png
```

---

## ML workflow

The ML portion of the repository contains training and inference scaffolding under the ml/ folder.

Main areas:

- ml/training/: training scripts and config
- ml/inference/: prediction helpers
- ml/models/: segmentation model implementations
- ml/data/: dataset loading and augmentation helpers

A smoke test entrypoint is available:

```powershell
python -m ml.training.smoke_test
```

For a full training run, prepare data as needed and run:

```powershell
python -m ml.training.train
```

Note that the current API inference path includes a fallback heuristic path when a full model checkpoint is not available, so the backend can still return a mask in a lightweight demo mode.

---

## Graph workflow

The graph pipeline uses a series of steps:

1. Convert a binary mask into a skeleton.
2. Build a graph from the skeleton pixels.
3. Heal topology by connecting loosely disconnected components.
4. Compute centrality and resilience metrics.
5. Export the final graph for visualization.

Key modules:

- graph/skeletonize.py
- graph/build_graph.py
- graph/heal_topology.py
- graph/centrality.py
- graph/export_geojson.py

---

## Testing

Unit tests are placed in the tests/ folder.

Run the test suite with:

```powershell
pytest -q
```

Some integration-style tests are also included for model-related flows. They may require heavier dependencies and are best used when you want to exercise the full inference path.

---

## Development notes

This repository is intentionally structured as a prototype. Some areas are still lightweight or heuristic, especially around full geospatial alignment and production-grade model training. The current implementation is suitable for:

- local demos,
- hackathon presentations,
- API prototyping,
- and experimentation with road-network resilience ideas.

Suggested next improvements include:

- better georeferencing of graph outputs,
- stronger training data and real satellite tiles,
- improved occlusion augmentation,
- and containerization for deployment.

---

## Summary

Route Resilience demonstrates how a road-network analysis pipeline can be built around a practical stack of FastAPI, Next.js, PyTorch-compatible ML tooling, and graph-based resilience metrics. It is a strong starting point for turning geospatial imagery into interpretable network intelligence.
