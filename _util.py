
# https://github.com/numpy/numpy/blob/v1.23.0/numpy/distutils/misc_util.py#L497-L504
def is_sequence(seq):
    if isinstance(seq, str):
        return False
    try:
        len(seq)
    except Exception:
        return False
    return True


def subok_not_ok(like=None, subok=False):
    if like is not None:
        raise ValueError("like=... parameter is not supported.")
    if subok:
        raise ValueError("subok parameter is not supported.")

