import inspect
from collections import defaultdict

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
        d = obj.__doc__.split("\n")
        if d[0]:
            return d[0].strip()
        else:
            # empty line, maybe the is lines 2-3 (np.vectorize)
            return "\n".join((d[1], d[2])).strip()


def dump_signatures(keys, namespace=None, replace=None):
    """Dump a pseudo-python source for a subset of signatures."""
    if namespace is None:
        namespace = np
    dct = grab_namespace(namespace)

    out = "\n"
    for key in keys:
        lst = dct[key]
        for obj in lst:
            sig = get_signature(obj)

            if replace:
                for old in replace:
                    sig = sig.replace(old, replace[old])

            out += f"def {sig}:\n    raise NotImplementedError\n\n"
    return out


def dump_difference(namespace):
    import torch_np

    dct_wrapper = grab_namespace(torch_np)
    wrapper_funcs = set([obj.__name__ for obj in dct_wrapper["function"]])

    dct_api = grab_namespace(namespace)
    namespace_funcs = set(obj.__name__ for obj in dct_api["function"])

    missing_names = namespace_funcs.difference(wrapper_funcs)

    for name in sorted(missing_names):
        print("- [ ]", name)

    breakpoint()

    extras = wrapper_funcs.difference(namespace_funcs)
    print("\n\n")
    for name in sorted(extras):
        print("- [ ]", name)


if __name__ == "__main__":

    #    dct = grab_namespace(np)
    #    print(dct.keys())

    #    for obj in dct['function']:
    #        print( get_signature(obj) )

    # dump array_api, full_signatures
    #  from numpy import array_api

    #   keys = ["builtin_function_or_method", "function"]
    #   replace = {"<no value>": "NoValue"}

    #   print(dump_signatures(keys, namespace=array_api, replace=replace))

    # dump the difference
    from numpy import array_api

    dump_difference(array_api)

# keys = ["builtin_function_or_method", "function"]
# replace = {"<no value>": "NoValue"}
# print(dump_signatures(keys, namespace=array_api, replace=replace))
