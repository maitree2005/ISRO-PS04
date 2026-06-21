# Route Resilience — End-to-end prototype

This repository contains an end-to-end prototype for "Route Resilience": an occlusion-aware
road segmentation and graph-based resilience analysis pipeline. It includes:

- A FastAPI backend serving inference, graph pipeline, and simulation endpoints.
- ML training and inference scaffolding (SegFormer integration + tiny UNet fallback).
- Graph pipeline: skeletonization, pixel→graph conversion, topology healing, centrality and resilience analysis.
- Next.js frontend with a Leaflet dashboard for visualization and simulation.
- Tests and CI workflows (unit + optional integration smoke test using CPU PyTorch).

This README explains how to set up, run, train, test and deploy the project locally.

## Quickstart (developer)

1. Create a Python environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install --upgrade pip
```

2. Install Python dependencies (project uses a minimal set; add extras as needed):

```bash
pip install -r requirements.txt
# If you want to run training/integration tests, also install:
pip install torch torchvision transformers albumentations pillow tqdm pyyaml
```

3. Run the backend (development):

```bash
# from repository root
uvicorn backend.main:app --reload --port 8000
```

4. Open the frontend (if present) — the Next.js app lives in `frontend/`.

```bash
cd frontend
npm install
npm run dev
```

Frontend development expects the backend at `http://localhost:8000` by default.

## ML: Training

Training scaffolding is in `ml/training/`.

- `ml/data/dataset.py`: dataset loader that returns HWC images and HxW masks.
- `ml/training/train.py`: processor-aware training loop using `transformers` SegFormer when available
	and falling back to a small `tiny_unet` if not. Check `ml/training/config.yaml` for hyperparameters.

Quick training (example smoke test that generates synthetic data and runs 1 epoch):

```bash
python -m ml.training.smoke_test
```

To run full training, prepare your dataset under the paths in `ml/training/config.yaml` and run:

```bash
python -m ml.training.train
```

Notes:
- The training loop uses a combined CrossEntropy + Dice loss and an optional continuity/topology loss
	(configured via `training.topology_weight` in `ml/training/config.yaml`).
- Augmentations (including synthetic occlusions) are defined in `ml/data/augmentations.py`.

## ML: Inference

Inference helpers are in `ml/inference/predict.py` and `ml/models/segformer.py`.

- `SegFormerWrapper` will attempt to load a transformers SegFormer model if available. If not, it remains in
	mock mode or can load a tiny UNet checkpoint as a fallback.
- You can call the backend endpoint `/api/predict` with optional form fields `checkpoint` and `model_name`.
	You may also preload a model into memory using the model management endpoints (faster repeated inference).

Example curl (checkpoint path relative to repo):

```bash
curl -X POST -F "image=@/path/to/img.png" -F "checkpoint=ml/checkpoints/segformer_best.pt" http://localhost:8000/api/predict --output mask.png
```

Preload a model (store wrapper in memory) and then use its handle for fast inference:

```bash
curl -X POST -F "checkpoint=ml/checkpoints/segformer_best.pt" http://localhost:8000/api/models/preload
# returns {"handle":"segformer_best.pt"}

curl -X POST -F "image=@/path/to/img.png" -F "handle=segformer_best.pt" http://localhost:8000/api/predict --output mask.png
```

## Graph pipeline & API

Key graph code lives in the `graph/` package:

- `graph/skeletonize.py` — converts binary mask to skeleton (skimage).
- `graph/build_graph.py` — converts skeleton pixels to a NetworkX graph.
- `graph/heal_topology.py` — connects components using candidate bridges and MST for topology healing.
- `graph/centrality.py` — computes betweenness and resilience index under node removals.
- `graph/export_geojson.py` — exports graphs to GeoJSON (includes numpy→native coercion).

Backend routes of interest:

- `POST /api/predict` — runs segmentation inference on image uploads.
- `POST /api/pipeline` — runs the graph pipeline (skeletonize → build_graph → heal → centrality) on a mask.
- `GET /api/graph` — returns sample or stored graph GeoJSON.
- `POST /api/simulate` — simulate node removals and compute resilience.

## Tests and CI

Unit tests live under `tests/` and there is an integration smoke test `tests/test_models_integration.py` that
preloads a tiny UNet checkpoint, runs `/predict` against it, and unloads it. Integration tests are disabled by
default and controlled by the `RUN_INTEGRATION_TESTS` environment variable.

Run unit tests locally:

```bash
pytest -q
```

Run the integration test (requires `torch` and more time/downloads):

```bash
SET RUN_INTEGRATION_TESTS=1
pytest -q tests/test_models_integration.py
```

CI workflows (in `.github/workflows`):

- `python-tests.yml` — runs pytest on push/PR with integration tests disabled.
- `python-integration.yml` — manual workflow to run the PyTorch CPU integration smoke test (use Actions → Run workflow).

## Contributing & next steps

Suggested next work items (pick and implement):

- Georeference pixel coordinates using rasterio transforms so graphs export accurate lat/lon.
- Improve occlusion augmentation using real-object cutouts (clouds, vehicles) for realism.
- Add a task queue (Celery/RQ) for long pipeline jobs and background model scoring.
- Dockerize backend + frontend and add GitHub Actions to build/push images.

If you'd like, I can:
- Wire model preloading into the frontend so the dashboard can request inference from a preloaded handle.
- Add a small dataset of real tiles and run a short training job to produce a demo checkpoint.

## Project layout (high level)

```
backend/                 # FastAPI app and routers
frontend/                # Next.js dashboard
ml/                      # ML code: models, training, inference, data helpers
graph/                   # Skeleton → graph → healing → metrics
tests/                   # Unit and integration tests
.github/workflows/       # CI workflows
```

---

If anything is unclear or you want me to change the README tone/format, tell me which section to expand.