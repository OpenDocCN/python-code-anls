# `D:\src\scipysrc\sympy\bin\test_optional_dependencies.py`

```
#!/usr/bin/env python
"""
Run tests for specific packages that use optional dependencies.

The optional dependencies need to be installed before running this.
"""

import pytest

# Add the local sympy to sys.path (needed for CI)
from get_sympy import path_hack
path_hack()

# Custom exception class to signal test failures
class TestsFailedError(Exception):
    pass

# List of directories and files to test
test_list = [
    # numpy
    '*numpy*',  # Test all modules containing 'numpy' in their path
    'sympy/core/',  # Test the core module of sympy
    'sympy/matrices/',  # Test the matrices module of sympy
    'sympy/physics/quantum/',  # Test the quantum module of sympy physics
    'sympy/utilities/tests/test_lambdify.py',  # Test the lambdify utility function
    'sympy/physics/control/',  # Test the control module of sympy physics
    
    # scipy
    '*scipy*',  # Test all modules containing 'scipy' in their path
    
    # matplotlib
    'sympy/plotting/',  # Test the plotting module of sympy
    
    # llvmlite
    '*llvm*',  # Test all modules containing 'llvm' in their path
    
    # aesara
    '*aesara*',  # Test all modules containing 'aesara' in their path
    
    # jax
    '*jax*',  # Test all modules containing 'jax' in their path
    
    # gmpy
    'sympy/ntheory',  # Test the number theory module of sympy
    'sympy/polys',  # Test the polynomials module of sympy
    
    # gmpy, numpy, scipy, autowrap, matplotlib
    'sympy/external',  # Test the external module of sympy
    
    # autowrap
    '*autowrap*',  # Test all modules containing 'autowrap' in their path
    
    # ipython
    '*ipython*',  # Test all modules containing 'ipython' in their path
    
    # antlr, lfortran, clang
    'sympy/parsing/',  # Test the parsing module of sympy
    
    # codegen
    'sympy/codegen/',  # Test the code generation module of sympy
    'sympy/utilities/tests/test_codegen',  # Test the codegen utility functions
    'sympy/utilities/_compilation/tests/test_compilation',  # Test compilation utilities
    'sympy/external/tests/test_codegen.py',  # Test code generation in external tests
    
    # cloudpickle
    'pickling',  # Test pickling functionality
    
    # pycosat
    'sympy/logic',  # Test the logic module of sympy
    'sympy/assumptions',  # Test the assumptions module of sympy
    
    # stats
    'sympy/stats',  # Test the statistics module of sympy
    
    # lxml
    "sympy/utilities/tests/test_mathml.py",  # Test the mathml utility functions
]

# List of specific files to exclude from tests
blacklist = [
    'sympy/physics/quantum/tests/test_circuitplot.py',  # Exclude the circuitplot test
]

# List of directories and files for doctests
doctest_list = [
    # numpy
    'sympy/matrices/',  # Test doctests in the matrices module of sympy
    'sympy/utilities/lambdify.py',  # Test doctests in the lambdify utility script
    
    # scipy
    '*scipy*',  # Test all doctests in modules containing 'scipy' in their path
    
    # matplotlib
    'sympy/plotting/',  # Test doctests in the plotting module of sympy
    
    # llvmlite
    '*llvm*',  # Test all doctests in modules containing 'llvm' in their path
    
    # aesara
    '*aesara*',  # Test all doctests in modules containing 'aesara' in their path
    
    # gmpy
    'sympy/ntheory',  # Test doctests in the number theory module of sympy
    'sympy/polys',  # Test doctests in the polynomials module of sympy
    
    # autowrap
    '*autowrap*',  # Test all doctests in modules containing 'autowrap' in their path
    
    # ipython
    '*ipython*',  # Test all doctests in modules containing 'ipython' in their path
    
    # antlr, lfortran, clang
    'sympy/parsing/',  # Test doctests in the parsing module of sympy
    
    # codegen
    'sympy/codegen/',  # Test doctests in the code generation module of sympy
    
    # pycosat
    'sympy/logic',  # Test doctests in the logic module of sympy
    'sympy/assumptions',  # Test doctests in the assumptions module of sympy
    
    # stats
    'sympy/stats',  # Test doctests in the statistics module of sympy
    
    # lxml
    "sympy/utilities/mathml/",  # Test doctests in the mathml utility directory
]

# Check if matplotlib is available and extend doctest_list if so
try:
    import matplotlib
    doctest_list.extend([
        'doc/src/tutorials/biomechanics/biomechanical-model-example.rst',  # Additional doctests
        'doc/src/tutorials/biomechanics/biomechanics.rst',  # Additional doctests
    ])
except ImportError:
    pass

# Print message indicating testing of optional dependencies
print('Testing optional dependencies')

# Import test and doctest functions from sympy module
from sympy import test, doctest

# Run tests and assign result to tests_passed
tests_passed = test(*test_list, blacklist=blacklist, force_colors=True)

# Convert boolean result to pytest ExitCode enumeration
if tests_passed is True:
    tests_passed = pytest.ExitCode.OK

# Run doctests and assign result to doctests_passed
doctests_passed = doctest(*doctest_list, force_colors=True)

# Check test and doctest results and raise appropriate error messages if failed
if (tests_passed != pytest.ExitCode.OK) and not doctests_passed:
    raise TestsFailedError('Tests and doctests failed')
elif tests_passed != pytest.ExitCode.OK:
    raise TestsFailedError('Doctests passed but tests failed')
elif not doctests_passed:
    raise TestsFailedError('Tests passed but doctests failed')
```