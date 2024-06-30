# `D:\src\scipysrc\sympy\sympy\core\multidimensional.py`

```
"""
Provides functionality for multidimensional usage of scalar-functions.

Read the vectorize docstring for more details.
"""

from functools import wraps


def apply_on_element(f, args, kwargs, n):
    """
    Returns a structure with the same dimension as the specified argument,
    where each basic element is replaced by the function f applied on it. All
    other arguments stay the same.
    """
    # Determine the specified argument (either from positional args or kwargs).
    if isinstance(n, int):
        structure = args[n]      # Get the argument from positional args
        is_arg = True            # Flag indicating it's from args
    elif isinstance(n, str):
        structure = kwargs[n]    # Get the argument from kwargs
        is_arg = False           # Flag indicating it's from kwargs

    # Define a nested function to apply f recursively on all elements of structure.
    def f_reduced(x):
        if hasattr(x, "__iter__"):  # If x is iterable, apply recursively
            return list(map(f_reduced, x))
        else:
            if is_arg:
                args[n] = x        # Replace the original arg with x
            else:
                kwargs[n] = x      # Replace the original kwarg with x
            return f(*args, **kwargs)  # Apply function f with updated args

    # Apply f_reduced recursively on structure to replace each element with f's result.
    return list(map(f_reduced, structure))


def iter_copy(structure):
    """
    Returns a copy of an iterable object (also copying all embedded iterables).
    """
    return [iter_copy(i) if hasattr(i, "__iter__") else i for i in structure]


def structure_copy(structure):
    """
    Returns a copy of the given structure (numpy-array, list, iterable, ..).
    """
    if hasattr(structure, "copy"):
        return structure.copy()   # If structure supports .copy(), use it
    return iter_copy(structure)   # Otherwise, recursively copy iterables


class vectorize:
    """
    Generalizes a function taking scalars to accept multidimensional arguments.

    Examples
    ========

    >>> from sympy import vectorize, diff, sin, symbols, Function
    >>> x, y, z = symbols('x y z')
    >>> f, g, h = list(map(Function, 'fgh'))

    >>> @vectorize(0)
    ... def vsin(x):
    ...     return sin(x)

    >>> vsin([1, x, y])
    [sin(1), sin(x), sin(y)]

    >>> @vectorize(0, 1)
    ... def vdiff(f, y):
    ...     return diff(f, y)

    >>> vdiff([f(x, y, z), g(x, y, z), h(x, y, z)], [x, y, z])
    [[Derivative(f(x, y, z), x), Derivative(f(x, y, z), y), Derivative(f(x, y, z), z)],
     [Derivative(g(x, y, z), x), Derivative(g(x, y, z), y), Derivative(g(x, y, z), z)],
     [Derivative(h(x, y, z), x), Derivative(h(x, y, z), y), Derivative(h(x, y, z), z)]]
    """
    def __init__(self, *mdargs):
        """
        The given numbers and strings characterize the arguments that will be
        treated as data structures, where the decorated function will be applied
        to every single element.
        If no argument is given, everything is treated multidimensional.
        """
        for a in mdargs:
            if not isinstance(a, (int, str)):
                raise TypeError("a is of invalid type")
        self.mdargs = mdargs
    def __call__(self, f):
        """
        Returns a wrapper for the one-dimensional function that can handle
        multidimensional arguments.
        """
        @wraps(f)
        def wrapper(*args, **kwargs):
            # 获取需要处理多维参数的参数列表
            if self.mdargs:
                mdargs = self.mdargs
            else:
                mdargs = range(len(args)) + kwargs.keys()

            arglength = len(args)

            for n in mdargs:
                if isinstance(n, int):
                    if n >= arglength:
                        continue
                    entry = args[n]
                    is_arg = True
                elif isinstance(n, str):
                    try:
                        entry = kwargs[n]
                    except KeyError:
                        continue
                    is_arg = False
                if hasattr(entry, "__iter__"):
                    # 现在创建给定数组的副本，然后直接操作其条目。
                    if is_arg:
                        args = list(args)
                        args[n] = structure_copy(entry)
                    else:
                        kwargs[n] = structure_copy(entry)
                    # 应用于元素的处理函数，递归调用包装器处理参数
                    result = apply_on_element(wrapper, args, kwargs, n)
                    return result
            # 如果没有多维参数需要处理，则调用原始函数
            return f(*args, **kwargs)
        return wrapper
```