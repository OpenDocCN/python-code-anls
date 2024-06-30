# `D:\src\scipysrc\scipy\scipy\_lib\__init__.py`

```
"""
Module containing private utility functions
===========================================

The ``scipy._lib`` namespace is empty (for now). Tests for all
utilities in submodules of ``_lib`` can be run with::

    from scipy import _lib
    _lib.test()

"""

# 从 scipy._lib._testutils 模块中导入 PytestTester 类
from scipy._lib._testutils import PytestTester
# 创建一个 PytestTester 的实例 test，用于测试当前模块
test = PytestTester(__name__)
# 删除 PytestTester 类的引用，避免在当前命名空间中保留不必要的类
del PytestTester
```