from fastapi.testclient import TestClient
from backend.main import app
from io import BytesIO
from PIL import Image
import numpy as np

client = TestClient(app)


def test_pipeline_endpoint_with_synthetic_mask():
    # Create a simple synthetic binary mask (50x50) with a line
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, 10:40] = 255
    img = Image.fromarray(mask)
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)

    files = {'mask': ('mask.png', buf, 'image/png')}
    resp = client.post('/api/pipeline', files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data, dict)
    assert data.get('type') == 'FeatureCollection'
    assert len(data.get('features', [])) >= 1
