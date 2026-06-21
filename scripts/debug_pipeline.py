import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from backend.main import app
from io import BytesIO
from PIL import Image
import numpy as np

client = TestClient(app)

mask = np.zeros((50, 50), dtype=np.uint8)
mask[25, 10:40] = 255
img = Image.fromarray(mask)
buf = BytesIO()
img.save(buf, format='PNG')
buf.seek(0)
files = {'mask': ('mask.png', buf, 'image/png')}
resp = client.post('/api/pipeline', files=files)
print('STATUS', resp.status_code)
try:
    print('JSON:', resp.json())
except Exception:
    print('TEXT:', resp.text)
    print('RAW:', resp.content)
