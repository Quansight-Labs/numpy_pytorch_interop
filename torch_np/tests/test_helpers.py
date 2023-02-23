import torch

import torch_np as tnp
from torch_np import _helpers


def test_ndarrays_to_tensors():
    out = _helpers.ndarrays_to_tensors(((tnp.asarray(42), 7), 3))
    assert len(out) == 2
    assert isinstance(out[0], tuple) and len(out[0]) == 2
    assert isinstance(out[0][0], torch.Tensor)
