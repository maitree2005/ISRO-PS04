from fastapi import APIRouter, HTTPException
from typing import List
import os
from ml.models.segformer import SegFormerWrapper

router = APIRouter()

# Simple in-memory registry for preloaded models
_PRELOADED = {}


@router.get("/list")
async def list_checkpoints(models_dir: str = "ml/checkpoints") -> List[str]:
    """List checkpoint files under `models_dir`."""
    if not os.path.exists(models_dir):
        return []
    files = [f for f in os.listdir(models_dir) if os.path.isfile(os.path.join(models_dir, f))]
    return files


@router.post("/preload")
async def preload_model(checkpoint: str, model_name: str = None):
    """Preload a model from a checkpoint into memory for faster inference.

    Returns a simple handle (the checkpoint filename) on success.
    """
    models_dir = os.path.dirname(checkpoint) if os.path.dirname(checkpoint) else 'ml/checkpoints'
    ckpt_path = checkpoint if os.path.isabs(checkpoint) else os.path.join(models_dir, os.path.basename(checkpoint))
    if not os.path.exists(ckpt_path):
        raise HTTPException(status_code=404, detail=f'Checkpoint not found: {ckpt_path}')

    # create wrapper and load
    wrapper = SegFormerWrapper(device='cpu')
    wrapper.load(model_name or 'nvidia/mit-b2', checkpoint=ckpt_path)
    if wrapper.mock:
        raise HTTPException(status_code=500, detail='Failed to load model from checkpoint')

    _PRELOADED[os.path.basename(ckpt_path)] = wrapper
    return {"handle": os.path.basename(ckpt_path)}


@router.get("/status")
async def status():
    return {"preloaded": list(_PRELOADED.keys())}


@router.post("/unload")
async def unload_model(handle: str):
    if handle in _PRELOADED:
        del _PRELOADED[handle]
        return {"unloaded": handle}
    raise HTTPException(status_code=404, detail='Handle not found')


def get_preloaded(handle: str):
    """Return a preloaded model wrapper or None."""
    return _PRELOADED.get(handle)
