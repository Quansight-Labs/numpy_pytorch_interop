import torch


def tensor_equal(a1_t, a2_t, equal_nan=False):
    """Implementation of array_equal/array_equiv."""
    if a1_t.shape != a2_t.shape:
        return False
    if equal_nan:
        nan_loc = (torch.isnan(a1_t) == torch.isnan(a2_t)).all()
        if nan_loc:
            # check the values
            result = a1_t[~torch.isnan(a1_t)] == a2_t[~torch.isnan(a2_t)]
        else:
            return False
    else:
        result = a1_t == a2_t
    return bool(result.all())
