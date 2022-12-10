from collections import defaultdict
import inspect

import numpy as np


def grab_namespace(np=np):
    """grab the numpy namespace content"""
    dct = defaultdict(list)
    for name in dir(np):
        obj = getattr(np, name)
        dct[type(obj).__name__].append(obj)
    return dct


def get_signature(obj):
    """Get a text signature of an object."""
    try:
        return obj.__name__ + str(inspect.signature(obj))
    except Exception:
        # builtins don't have it, try the first line of the docstring
        d = obj.__doc__.split('\n')
        if d[0]:
            return d[0].strip()
        else:
            # empty line, maybe the is lines 2-3 (np.vectorize)
            return '\n'.join((d[1], d[2])).strip()


def dump_signatures(keys, replace=None):
    """Dump a pseudo-python source for a subset of signatures."""
    dct = grab_namespace(np)

    out = "\n"
    for key in keys:
        lst = dct[key]
        for obj in lst:
            sig = get_signature(obj)

            if replace:
                for old in replace:
                    sig = sig.replace(old, replace[old])

            out += f"def {sig}:\n    raise NotImlementedError\n\n"
    return out

if __name__ == "__main__":

#    dct = grab_namespace(np)
#    print(dct.keys())

#    for obj in dct['function']:
#        print( get_signature(obj) )

    keys = ["builtin_function_or_method", "function"]
    replace = {"<no value>": "NoValue"}

    print(dump_signatures(keys, replace))
