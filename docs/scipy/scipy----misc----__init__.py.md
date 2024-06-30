# `D:\src\scipysrc\scipy\scipy\misc\__init__.py`

```
"""
==========================================
Miscellaneous routines (:mod:`scipy.misc`)
==========================================

.. currentmodule:: scipy.misc

.. deprecated:: 1.10.0

   This module is deprecated and will be completely
   removed in SciPy v2.0.0.

Various utilities that don't have another home.

.. autosummary::
   :toctree: generated/

   ascent - Get example image for processing
   central_diff_weights - Weights for an n-point central mth derivative
   derivative - Find the nth derivative of a function at a point
   face - Get example image for processing
   electrocardiogram - Load an example of a 1-D signal

"""


from ._common import *  # 导入 _common 模块中的所有内容
from . import _common  # 导入 _common 模块
import warnings  # 导入警告模块


# Deprecated namespaces, to be removed in v2.0.0
from . import common, doccer  # 导入 common 和 doccer 子模块

__all__ = _common.__all__  # 将 _common 模块中的 __all__ 列表赋值给当前模块的 __all__


dataset_methods = ['ascent', 'face', 'electrocardiogram']  # 定义数据集方法列表


def __dir__():
    return __all__  # 返回当前模块的 __all__ 列表作为可用属性


def __getattr__(name):
    if name not in __all__:  # 如果请求的属性不在 __all__ 列表中
        raise AttributeError(
            "scipy.misc is deprecated and has no attribute "
            f"{name}.")  # 抛出属性错误，提示模块已经被弃用且没有该属性

    if name in dataset_methods:  # 如果请求的属性在数据集方法列表中
        msg = ("The module `scipy.misc` is deprecated and will be "
               "completely removed in SciPy v2.0.0. "
               f"All dataset methods including {name}, must be imported "
               "directly from the new `scipy.datasets` module.")
    else:  # 否则
        msg = (f"The method `{name}` from the `scipy.misc` namespace is"
               " deprecated, and will be removed in SciPy v1.12.0.")

    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)  # 发出警告，提示方法已弃用

    return getattr(name)  # 返回请求的属性


del _common  # 删除 _common 模块的引用

from scipy._lib._testutils import PytestTester  # 导入 PytestTester 类
test = PytestTester(__name__)  # 创建 PytestTester 类的实例，并赋值给 test 变量
del PytestTester  # 删除 PytestTester 类的引用
```