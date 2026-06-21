from fastapi.testclient import TestClient
from backend.main import app
from io import BytesIO
from PIL import Image
import numpy as np

client = TestClient(app)


def test_predict_endpoint_returns_mask():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[10:22, 5:27] = [200, 200, 200]
    pil = Image.fromarray(img)
    buf = BytesIO()
    pil.save(buf, format='PNG')
    buf.seek(0)

    files = {'image': ('img.png', buf, 'image/png')}
    resp = client.post('/api/predict', files=files)
    assert resp.status_code == 200
    assert resp.headers['content-type'] == 'image/png'
    # Basic check: response body is non-empty
    assert len(resp.content) > 0
