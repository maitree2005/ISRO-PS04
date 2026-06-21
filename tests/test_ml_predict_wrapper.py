import numpy as np
from ml.models.segformer import SegFormerWrapper


def test_segformer_wrapper_mock():
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[16, 5:27] = 255
    w = SegFormerWrapper()
    # intentionally not calling load() to force mock mode
    mask = w.predict(arr)
    assert mask.dtype == np.uint8
    assert mask.shape[0] == arr.shape[0]
    assert mask.shape[1] == arr.shape[1]
    # ensure mask has at least one non-zero pixel
    assert mask.sum() > 0
