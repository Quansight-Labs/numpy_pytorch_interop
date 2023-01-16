# Generate the mapping between numpy names and replacement names from the wrapper module.
#
# The default is autopopulated, apply manual tweaks on the result, if needed.

import numpy as np
import wrapper

_all = set(name for name in dir(wrapper) if not name.startswith("_"))

_all.remove("ndarray")
_all.remove("NoValue")
# _all.remove("mapping")

pieces = [
    "     np.{np_name}: {wrapper_name}, ".format(np_name=name, wrapper_name=name)
    for name in sorted(_all)
]

# XXX: apply additional manual surgery here, if neeeded.

with open("_mapping.py", "w") as f:
    f.write("import numpy as np\n\n")
    f.write("mapping = {\n")
    f.write("\n".join(pieces))
    f.write("\n}\n")
