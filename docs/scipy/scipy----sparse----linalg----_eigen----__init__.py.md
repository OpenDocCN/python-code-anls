# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_eigen\__init__.py`

```
"""
Sparse Eigenvalue Solvers
-------------------------

The submodules of sparse.linalg._eigen:
    1. lobpcg: Locally Optimal Block Preconditioned Conjugate Gradient Method

"""

# 从当前目录的arpack模块中导入所有内容
from .arpack import *

# 从当前目录的lobpcg模块中导入所有内容
from .lobpcg import *

# 从当前目录的_svds模块中导入svds函数
from ._svds import svds

# 从当前目录中导入arpack模块
from . import arpack

# 定义__all__列表，包含需要公开的模块和异常
__all__ = [
    'ArpackError', 'ArpackNoConvergence',
    'eigs', 'eigsh', 'lobpcg', 'svds'
]

# 从scipy._lib._testutils模块中导入PytestTester类
from scipy._lib._testutils import PytestTester

# 创建一个PytestTester对象test，用于测试当前模块
test = PytestTester(__name__)

# 删除PytestTester类的引用，以避免全局范围内存在不必要的引用
del PytestTester
```