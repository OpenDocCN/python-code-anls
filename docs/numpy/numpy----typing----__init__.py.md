# `.\numpy\numpy\typing\__init__.py`

```py
"""
============================
Typing (:mod:`numpy.typing`)
============================

.. versionadded:: 1.20

Large parts of the NumPy API have :pep:`484`-style type annotations. In
addition a number of type aliases are available to users, most prominently
the two below:

- `ArrayLike`: objects that can be converted to arrays
- `DTypeLike`: objects that can be converted to dtypes

.. _typing-extensions: https://pypi.org/project/typing-extensions/

Mypy plugin
-----------

.. versionadded:: 1.21

.. automodule:: numpy.typing.mypy_plugin

.. currentmodule:: numpy.typing

Differences from the runtime NumPy API
--------------------------------------

NumPy is very flexible. Trying to describe the full range of
possibilities statically would result in types that are not very
helpful. For that reason, the typed NumPy API is often stricter than
the runtime NumPy API. This section describes some notable
differences.

ArrayLike
~~~~~~~~~

The `ArrayLike` type tries to avoid creating object arrays. For
example,

.. code-block:: python

    >>> np.array(x**2 for x in range(10))
    array(<generator object <genexpr> at ...>, dtype=object)

is valid NumPy code which will create a 0-dimensional object
array. Type checkers will complain about the above example when using
the NumPy types however. If you really intended to do the above, then
you can either use a ``# type: ignore`` comment:

.. code-block:: python

    >>> np.array(x**2 for x in range(10))  # type: ignore

or explicitly type the array like object as `~typing.Any`:

.. code-block:: python

    >>> from typing import Any
    >>> array_like: Any = (x**2 for x in range(10))
    >>> np.array(array_like)
    array(<generator object <genexpr> at ...>, dtype=object)

ndarray
~~~~~~~

It's possible to mutate the dtype of an array at runtime. For example,
the following code is valid:

.. code-block:: python

    >>> x = np.array([1, 2])
    >>> x.dtype = np.bool

This sort of mutation is not allowed by the types. Users who want to
write statically typed code should instead use the `numpy.ndarray.view`
method to create a view of the array with a different dtype.

DTypeLike
~~~~~~~~~

The `DTypeLike` type tries to avoid creation of dtype objects using
dictionary of fields like below:

.. code-block:: python

    >>> x = np.dtype({"field1": (float, 1), "field2": (int, 3)})

Although this is valid NumPy code, the type checker will complain about it,
since its usage is discouraged.
Please see : :ref:`Data type objects <arrays.dtypes>`

Number precision
~~~~~~~~~~~~~~~~

The precision of `numpy.number` subclasses is treated as a invariant generic
parameter (see :class:`~NBitBase`), simplifying the annotating of processes
involving precision-based casting.

.. code-block:: python

    >>> from typing import TypeVar
    >>> import numpy as np
    >>> import numpy.typing as npt

    >>> T = TypeVar("T", bound=npt.NBitBase)
    >>> def func(a: "np.floating[T]", b: "np.floating[T]") -> "np.floating[T]":
"""
    # 定义一个名为 `matrix_transpose` 的函数，用于计算矩阵的转置
    def matrix_transpose(matrix):
        # 获取矩阵的行数
        rows = len(matrix)
        # 获取矩阵的列数（假设矩阵是规则的，每行列数相同）
        cols = len(matrix[0])
        
        # 使用嵌套的列表推导式，生成转置后的矩阵
        transpose = [[matrix[j][i] for j in range(rows)] for i in range(cols)]
        
        # 返回转置后的矩阵
        return transpose
# 如果存在文档字符串，则将其扩展为包含 `_docstrings` 变量的内容
if __doc__ is not None:
    # 导入 numpy._typing._add_docstring 模块中的 _docstrings 变量
    from numpy._typing._add_docstring import _docstrings
    # 将 _docstrings 添加到当前模块的文档字符串末尾
    __doc__ += _docstrings
    # 添加一个额外的类到文档字符串中
    __doc__ += '\n.. autoclass:: numpy.typing.NBitBase\n'
    # 删除 _docstrings 变量，以避免在全局命名空间中存在多余的引用
    del _docstrings

# 导入 PytestTester 类
from numpy._pytesttester import PytestTester
# 创建一个名为 test 的 PytestTester 实例，其模块名称为当前模块的名称
test = PytestTester(__name__)
# 删除 PytestTester 类，以避免在全局命名空间中存在多余的引用
del PytestTester
```