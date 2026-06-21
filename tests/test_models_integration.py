import os
import tempfile
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app


@pytest.mark.skipif(not os.environ.get('RUN_INTEGRATION_TESTS'), reason='Integration tests disabled')
def test_preload_predict_unload():
    # require torch for creating a checkpoint
    try:
        import torch
    except Exception:
        pytest.skip('torch not available')

    client = TestClient(app)

    # create checkpoints dir
    ckpt_dir = os.path.join('ml', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'test_unet.pt')

    # build tiny unet state and save
    from ml.models.tiny_unet import get_tiny_unet
    model = get_tiny_unet(num_classes=2, in_channels=3)
    torch.save(model.state_dict(), ckpt_path)

    # create a small test image
    fd, img_path = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    img = Image.new('RGB', (128, 128), (100, 100, 100))
    img.save(img_path)

    try:
        # preload
        resp = client.post('/api/models/preload', data={'checkpoint': ckpt_path})
        assert resp.status_code == 200, resp.text
        handle = resp.json().get('handle')
        assert handle is not None

        # predict using handle
        with open(img_path, 'rb') as f:
            files = {'image': ('img.png', f, 'image/png')}
            resp2 = client.post('/api/predict', files=files, data={'handle': handle})
        assert resp2.status_code == 200
        assert resp2.headers.get('content-type') in ('image/png', 'image/png; charset=binary')

        # unload
        resp3 = client.post('/api/models/unload', data={'handle': handle})
        assert resp3.status_code == 200
    finally:
        # cleanup
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        if os.path.exists(img_path):
            os.remove(img_path)
