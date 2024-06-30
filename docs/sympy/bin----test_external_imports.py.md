# `D:\src\scipysrc\sympy\bin\test_external_imports.py`

```
#!/usr/bin/env python
"""
Test that

from sympy import *

doesn't import anything other than SymPy, it's hard dependencies (mpmath), and
hard optional dependencies (gmpy2). Importing unnecessary libraries
can accidentally add hard dependencies to SymPy in the worst case, or at best
slow down the SymPy import time when they are installed.

Note, for this test to be effective, every external library that could
potentially be imported by SymPy must be installed.

TODO: Monkeypatch the importer to detect non-standard library imports even
when they aren't installed.

Based on code from
https://stackoverflow.com/questions/22195382/how-to-check-if-a-module-library-package-is-part-of-the-python-standard-library.
"""

# These libraries will always be imported with SymPy
hard_dependencies = ['mpmath']

# These libraries are optional, but are always imported at SymPy import time
# when they are installed. External libraries should only be added to this
# list if they are required for core SymPy functionality.
hard_optional_dependencies = ['gmpy', 'gmpy2', 'pycosat', 'python-sat']

import sys
import os

# Function to check if a path belongs to Python standard library
def is_stdlib(p):
    return ((p.startswith((sys.prefix, sys.base_prefix)))
            and 'site-packages' not in p)

# Set of paths that belong to the Python standard library
stdlib = {p for p in sys.path if is_stdlib(p)}

# List of currently loaded module names
existing_modules = list(sys.modules.keys())


# hook in-tree SymPy into Python path, if possible

# Get absolute path and directory of this script
this_path = os.path.abspath(__file__)
this_dir = os.path.dirname(this_path)
# Determine top level directory of SymPy
sympy_top = os.path.split(this_dir)[0]
sympy_dir = os.path.join(sympy_top, 'sympy')

# If the SymPy directory exists, add it to the beginning of sys.path
if os.path.isdir(sympy_dir):
    sys.path.insert(0, sympy_top)

# Function to test for unexpected external module imports
def test_external_imports():
    # Execute the import statement 'from sympy import *' in an empty namespace
    exec("from sympy import *", {})

    # List to store names of unexpected external modules
    bad = []
    for mod in sys.modules:
        # Skip modules that are submodules of other modules
        if '.' in mod and mod.split('.')[0] in sys.modules:
            continue

        # Skip modules that were already imported before this test
        if mod in existing_modules:
            continue

        # Skip modules that are part of SymPy, its hard dependencies,
        # or hard optional dependencies
        if any(mod == i or mod.startswith(i + '.') for i in ['sympy'] +
            hard_dependencies + hard_optional_dependencies):
            continue

        # Skip modules that are built-in Python modules
        if mod in sys.builtin_module_names:
            continue

        # Get the filename associated with the module
        fname = getattr(sys.modules[mod], "__file__", None)
        if fname is None:
            bad.append(mod)
            continue

        # If the filename is an __init__.py file, get its directory
        if fname.endswith(('__init__.py', '__init__.pyc', '__init__.pyo')):
            fname = os.path.dirname(fname)

        # Check if the directory of the module's file is in the Python standard library
        if os.path.dirname(fname) in stdlib:
            continue

        bad.append(mod)

    # If there are any unexpected external modules, raise an error
    if bad:
        raise RuntimeError("""Unexpected external modules found when running 'from sympy import *':
    """ + '\n    '.join(bad))

    # If no unexpected external modules were found, print a success message
    print("No unexpected external modules were imported with 'from sympy import *'!")

if __name__ == '__main__':
    # Run the test function when this script is executed directly
    test_external_imports()
```