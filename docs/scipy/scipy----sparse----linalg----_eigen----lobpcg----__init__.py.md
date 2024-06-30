# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_eigen\lobpcg\__init__.py`

```
"""
Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG)

LOBPCG is a preconditioned eigensolver for large symmetric positive definite
(SPD) generalized eigenproblems.

Call the function lobpcg - see help for lobpcg.lobpcg.

"""
# 导入 lobpcg 模块中的所有内容
from .lobpcg import *

# 将不以下划线开头的所有变量和函数名添加到 __all__ 列表中，以便导出
__all__ = [s for s in dir() if not s.startswith('_')]

# 导入 PytestTester 类
from scipy._lib._testutils import PytestTester

# 创建一个 PytestTester 对象，用于测试当前模块
test = PytestTester(__name__)

# 删除 PytestTester 类的引用，确保不会在当前命名空间中保留该类的引用
del PytestTester
```