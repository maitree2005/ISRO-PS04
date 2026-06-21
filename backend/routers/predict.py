from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image
import numpy as np

router = APIRouter()


@router.post("/predict")
async def predict_mask(image: UploadFile = File(...), checkpoint: str = Form(None), model_name: str = Form(None), handle: str = Form(None)):
    """Inference stub:
    - If the uploaded file is a GeoTIFF (rasterio can open), attempt a simple NDVI-based heuristic mask.
    - Otherwise, fall back to grayscale thresholding for common image formats.
    Returns a binary PNG mask.
    """
    try:
        contents = await image.read()

        # Prefer ML inference wrapper if available
        try:
            # If a preloaded handle is provided, try to use that wrapper directly
            if handle:
                from . import models as models_router
                wrapper = models_router.get_preloaded(handle)
                if wrapper is not None and not wrapper.mock:
                    pil = Image.open(BytesIO(contents))
                    arr = np.array(pil)
                    mask = wrapper.predict(arr)
                else:
                    # fallback to standard inference
                    from ml.inference.predict import predict_mask_from_image
                    pil = Image.open(BytesIO(contents))
                    mask = predict_mask_from_image(pil, checkpoint=checkpoint, model_name=model_name)
            else:
                from ml.inference.predict import predict_mask_from_image
                pil = Image.open(BytesIO(contents))
                mask = predict_mask_from_image(pil, checkpoint=checkpoint, model_name=model_name)
        except Exception:
            # Fallback: try GeoTIFF NDVI heuristic, otherwise grayscale threshold
            try:
                import rasterio
                from rasterio.io import MemoryFile
                mem = MemoryFile(contents)
                with mem.open() as ds:
                    if ds.count >= 4:
                        try:
                            red = ds.read(3).astype('float32')
                            nir = ds.read(4).astype('float32')
                            eps = 1e-6
                            ndvi = (nir - red) / (nir + red + eps)
                            mask = (ndvi < 0.2).astype(np.uint8) * 255
                        except Exception:
                            rgb = np.mean(ds.read([1, 2, 3]), axis=0)
                            mask = (rgb > rgb.mean()).astype(np.uint8) * 255
                    else:
                        arr = np.mean([ds.read(i + 1) for i in range(ds.count)], axis=0)
                        mask = (arr > arr.mean()).astype(np.uint8) * 255
            except Exception:
                img = Image.open(BytesIO(contents)).convert("L")
                arr = np.array(img)
                mask = (arr > 127).astype(np.uint8) * 255

        out = Image.fromarray(mask.astype('uint8'))
        buf = BytesIO()
        out.save(buf, format='PNG')
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
